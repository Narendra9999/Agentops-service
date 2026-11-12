"""Databricks clients used by customer workers; no AgentOPS dependency."""
import json


def chat_model(settings, workspace, model_name):
    """Keep LangChain tool/usage handling while selecting the Gateway API route."""
    from databricks_langchain import ChatDatabricks
    class AsyncChatDatabricks(ChatDatabricks):
        # Pinned databricks-langchain 0.8.0 has no native _agenerate. Reuse its
        # tool/response conversion, with a request-scoped async transport.
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            import httpx
            from openai.types.chat import ChatCompletion
            from openai.types.responses import Response
            from framework.inference.async_runtime import run_blocking
            headers = await run_blocking(workspace.config.authenticate)
            data = self._prepare_inputs(messages, stop, **kwargs)
            # LangChain already traces this model invocation and its usage.
            # Use HTTP transport rather than a second instrumented OpenAI call
            # so native MLflow totals count each provider request once.
            async with httpx.AsyncClient(timeout=self.timeout or 240) as client:
                path = "responses" if self.use_responses_api else "chat/completions"
                response = await client.post(
                    str(self.client.base_url).rstrip("/") + "/" + path,
                    headers=headers, json=data,
                )
                response.raise_for_status()
                if self.use_responses_api:
                    return self._convert_responses_api_response_to_chat_result(
                        Response.model_validate(response.json()))
                return self._convert_response_to_chat_result(
                    ChatCompletion.model_validate(response.json()))

    model = AsyncChatDatabricks(model=model_name, workspace_client=workspace,
                                temperature=0, max_retries=0)
    if settings.model_transport == "ai_gateway":
        # 0.8.0 builds an SDK-authenticated OpenAI client on the serving route.
        # Reuse its rotating authentication; change only the public base_url.
        model.client.base_url = workspace.config.host.rstrip('/') + '/ai-gateway/mlflow/v1/'
    return model


class GenieWaitTimeout(TimeoutError):
    def __init__(self, details):
        self.details = details
        super().__init__("Genie did not finish within the 180-second wait limit "
                         "(last status: " + details.get("status", "unknown") + "). "
                         "No result was returned. Open the Genie conversation to check its status; "
                         "this request was not automatically restarted.")


def genie_presentation(space_id, message, attachments):
    """Bounded query evidence for the UI; no SQL thoughts or executable viz specs."""
    tables = []
    for item in attachments[:10]:
        statement = (item.get("query_result") or {}).get("statement_response") or {}
        columns = ((statement.get("manifest") or {}).get("schema") or {}).get("columns") or []
        result = statement.get("result") or {}
        rows = result.get("data_array") or []
        if columns:
            tables.append({"title": (item.get("query") or {}).get("title") or "Genie query result",
                           "columns": [str(c.get("name", ""))[:200] for c in columns[:30]],
                           "rows": [[None if v is None else str(v)[:2000] for v in row[:30]] for row in rows[:100]],
                           "truncated": bool(statement.get("manifest", {}).get("truncated")) or
                                        len(rows) > 100 or len(columns) > 30 or bool(result.get("next_chunk_index"))})
    return {"space_id": space_id, "conversation_id": message.conversation_id,
            "message_id": message.id or message.message_id, "tables": tables,
            "has_visualization": any(item.get("viz") for item in attachments)}


class GenieCostService:
    def __init__(self, workspace, space_id):
        self.workspace, self.space_id = workspace, space_id

    def ask(self, question):
        from .genie_agent import ask_agent
        return ask_agent(self.workspace, self.space_id, question)


class ExistingResponsesWorker:
    """Call an existing ResponsesAgent endpoint once; never deploy a worker."""
    def __init__(self, workspace, endpoint, agent):
        self.workspace, self.endpoint, self.agent = workspace, endpoint, agent

    def invoke(self, state):
        import mlflow
        import requests
        from langchain_core.messages import AIMessage
        from urllib.parse import quote
        from .progress import operation
        from .usage import current_meter
        payload = {"input": [{"role": "user", "content": m.content} for m in state["messages"]]}
        # Propagation requires the worker endpoint to support MLflow's incoming
        # trace headers. A client span alone does not prove a joined server trace.
        with operation(self.agent, "worker_invocation", "Calling existing specialist endpoint"), \
             mlflow.start_span(name="worker.remote", span_type="TOOL") as span:
            span.set_attribute("agentops.worker.endpoint", self.endpoint)
            headers = {**self.workspace.config.authenticate(),
                       **mlflow.tracing.get_tracing_context_headers_for_http_request()}
            response = requests.post(
                self.workspace.config.host.rstrip('/') + '/serving-endpoints/' + quote(self.endpoint, safe='') + '/invocations',
                headers=headers, json=payload, timeout=(20,240))
            response.raise_for_status()
            result = response.json()
            text = "\n".join(p.get('text','') for item in result.get('output',[])
                             for p in item.get('content',[]) if p.get('type')=='output_text')
            if not text:
                raise RuntimeError("Existing worker returned no ResponsesAgent answer")
            custom = result.get('custom_outputs') or {}
            remote_trace = custom.get('trace_id')
            if remote_trace:
                span.set_attribute('agentops.worker.trace_id', str(remote_trace))
            if current_meter():
                current_meter().observe_remote(self.agent, self.endpoint, result, invocation_id=span.span_id)
            update = {'messages':[AIMessage(content=text,name=self.agent)]}
            if self.agent=='cost_estimation' and custom.get('genie_result'):
                update['genie_result']=custom['genie_result']
            if self.agent=='platform_docs':
                # Only preserve explicit worker-provided source metadata.
                from urllib.parse import urlparse
                update['source_urls']=[url for url in custom.get('source_urls',[]) if isinstance(url,str)
                                       and urlparse(url).scheme=='https'
                                       and urlparse(url).netloc=='docs.databricks.com']
            return update


class ExistingGenieWorker:
    def __init__(self, workspace, space_id):
        self.service = GenieCostService(workspace, space_id)

    def invoke(self, state):
        from langchain_core.messages import AIMessage
        from .progress import operation
        from .usage import mark_unmetered
        question = "\n\n".join(str(m.content) for m in state['messages'])
        mark_unmetered('genie')
        with operation('cost_estimation','genie_query','Calling existing Genie agent in STANDARD mode'):
            result = self.service.ask(question)
        return {'messages':[AIMessage(content=json.dumps(result,default=str),name='cost_estimation')],
                'genie_result':result['presentation']}
