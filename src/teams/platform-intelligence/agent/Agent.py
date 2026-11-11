"""Standard AgentOPS team entry point; customer graph owns all routing."""
from copy import deepcopy
import os
import json
import time
from pathlib import Path

import mlflow
from mlflow.pyfunc import ResponsesAgent
from framework.agent_base import AgentOPSBase
from fe_multi_agent.supervisor_agent import SupervisorAgent, astream_prediction
from framework.inference.async_runtime import AsyncResponsesAgent, call_async_method, iterate_async_sync, run_sync
from contextlib import aclosing
from fe_multi_agent.config import Settings


class _GuardedTurn(AgentOPSBase):
    """One instance per request isolates mutable guardrail/session context."""
    def __init__(self, config, graph_agent):
        super().__init__(config)
        self.graph_agent = graph_agent

    async def _aprocess_request(self, request):
        response = await self.graph_agent.apredict(request)
        artifact = (response.custom_outputs or {}).get('genie_result')
        if artifact and self.guardrails_enabled:
            from framework.inference import extract_request_text
            checked = await call_async_method(self.post_llm_guardrails, "acheck", "check", extract_request_text(request),json.dumps(artifact),self._request_context)
            if checked.get('blocked'):
                mlflow.update_current_trace(tags={'agentops.guardrail.artifacts.blocked':'true'})
                payload = response.model_dump(exclude_none=True)
                payload['custom_outputs'].pop('genie_result',None)
                payload['output']=[self.create_text_output_item(text=checked.get('message','Result filtered for safety.'),id='filtered-result')]
                return type(response)(**payload)
        return response


class ChatModel(AsyncResponsesAgent):
    def load_context(self, context):
        self.config = deepcopy(context.model_config or {})
        block = self.config.get('multi_agent') or {}
        if block.get('pattern') != 'langgraph_supervisor':
            raise ValueError('multi_agent.pattern must be langgraph_supervisor')
        if set(block.get('agents', {})) != {'supervisor', 'platform_docs', 'cost_estimation'}:
            raise ValueError('Configure supervisor, platform_docs and cost_estimation only')
        values = dict(block.get('settings') or {})
        llm = self.config.get('llm') or {}
        values['supervisor_model'] = llm['endpoint']
        values['model_transport'] = llm.get('api', 'model_serving')
        values['prompt_manifest'] = str(Path(__file__).resolve().parent.parent / 'configs/prompts.json')
        values['experiment_id'] = os.getenv('MLFLOW_EXPERIMENT_ID') or self.config.get('tracing', {}).get('experiment_id', '')
        self.graph_agent = SupervisorAgent(Settings(**values))
        self.graph_agent.settings.validate()
        self.graph_agent.load_context(context)
        mlflow.langchain.autolog()

    def predict(self, request):
        return run_sync(self.apredict(request))

    @mlflow.trace(name='agentops.platform_intelligence', span_type='AGENT')
    async def apredict(self, request):
        # Apply existing AgentOPS pre/post guardrails around the entire graph.
        # No shared _request_context across concurrent serving requests.
        start=time.monotonic()
        turn=_GuardedTurn(self.config, self.graph_agent)
        response=await turn.apredict(request)
        custom=dict(response.custom_outputs or {})
        active=mlflow.get_current_active_span()
        custom.setdefault('trace_id', active.trace_id if active else mlflow.get_last_active_trace_id())
        tags={**turn.default_tags, **custom.get('tags', {}),
              'agentops.latency_ms':str(round((time.monotonic()-start)*1000,2)),
              'agentops.thread_id':str(custom.get('thread_id',''))}
        custom['tags']=tags
        response.custom_outputs=custom
        mlflow.update_current_trace(tags=tags)
        return response

    @mlflow.trace(
        name="platform_intelligence.stream", span_type="AGENT",
        output_reducer=ResponsesAgent.responses_agent_output_reducer,
    )
    def predict_stream(self, request):
        # Public activity streams immediately; final output/Genie tables/usage are
        # released only after AgentOPS post-guardrails accept the final answer.
        yield from iterate_async_sync(lambda: self.apredict_stream(request))

    async def apredict_stream(self, request):
        async with aclosing(astream_prediction(self.apredict, request)) as stream:
            async for event in stream:
                yield event


mlflow.models.set_model(ChatModel())
