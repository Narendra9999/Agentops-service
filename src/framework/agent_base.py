"""
AgentOPS Framework — Base Agent Class
Context-aware agent hooks and shared callback lifecycle. Provides standardized guardrails,
MLflow tracing with tags, session history, and long-term user memory.
"""

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from typing import Generator, Optional
import logging
import uuid
import time
import json
from contextlib import AsyncExitStack, aclosing, asynccontextmanager

# Public helpers remain importable here for existing team integrations.
from framework.observability.tracing import configure_trace_destination
from framework.observability.usage_tracking import (
    emit_standard_trace_metrics,
    invoke_with_usage_tracking,
    record_response_usage,
)

from framework.inference.async_runtime import AsyncResponsesAgent, call_async_method, iterate_blocking, iterate_async_sync, request_deadline, run_blocking, run_sync

from framework.inference.request_context import RequestContext

from framework.guardrails.pre_llm import PreLLMGuardrails
from framework.guardrails.post_llm import PostLLMGuardrails
from framework.guardrails.middleware import PIIRedactionMiddleware
from framework.session.session_store import SessionStore
from framework.inference import extract_request_text, extract_response_text

logger = logging.getLogger(__name__)


class AgentOPSBase(AsyncResponsesAgent):
    """
    Base agent class for the AgentOPS framework.
    Wraps agent logic with pre/post LLM guardrails and MLflow tracing.
    Subclasses must implement _process_request().

    Memory support (via Lakebase DatabricksStore):
      - Short-term: per-thread conversation history (auto-loaded/saved)
      - Long-term: cross-session user memory recalled into RequestContext.state["user_memories"]

    Clients pass thread_id and user_id via custom_inputs:
      {"thread_id": "abc-123", "user_id": "user@example.com"}
    Response includes custom_outputs with thread_id for follow-up requests.
    """

    def __init__(self, config: dict):
        self.config = config
        configure_trace_destination(config)
        agent_cfg = config.get("agent", {})
        self.agent_name = agent_cfg.get("name", "unnamed_agent")
        self.agent_version = agent_cfg.get("version", "0.0.0")

        # Initialize guardrails from config
        gr_config = config.get("guardrails", {})
        self.guardrails_enabled = gr_config.get("enabled", True)
        self.guardrail_middleware = PIIRedactionMiddleware(gr_config)
        self.pre_llm_guardrails = PreLLMGuardrails(gr_config.get("pre_llm", {}))
        self.post_llm_guardrails = PostLLMGuardrails(gr_config.get("post_llm", {}))

        # Session history + long-term memory
        self.session_store = SessionStore(config)

        # Tracing tags — set on every trace automatically
        # Environment is resolved at runtime, not from config.yaml
        tracing_config = config.get("tracing", {})
        environment = self._resolve_environment(tracing_config)
        self.default_tags = {
            "agentops.agent_name": self.agent_name,
            "agentops.agent_version": self.agent_version,
            "agentops.framework_version": "0.1.0",
            "agentops.environment": environment,
            **tracing_config.get("default_tags", {}),
        }

    @staticmethod
    def _resolve_environment(tracing_config: dict) -> str:
        """
        Resolve environment dynamically. Priority:
        1. AGENTOPS_ENVIRONMENT env var (set by DABs/Jenkins)
        2. tracing.default_tags.agentops.environment in config.yaml
        3. Detect from workspace URL
        4. Default: "unknown"
        """
        import os

        # 1. Env var (set by CI/CD or DABs target)
        env = os.environ.get("AGENTOPS_ENVIRONMENT")
        if env:
            return env

        # 2. Config.yaml value
        config_env = tracing_config.get("default_tags", {}).get("agentops.environment")
        if config_env:
            return config_env

        # 3. Try to detect from workspace
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            host = w.config.host or ""
            if "stage" in host.lower() or "staging" in host.lower():
                return "stage"
            elif "prod" in host.lower():
                return "prod"
            else:
                return "dev"
        except Exception:
            pass

        return "unknown"

    def _set_trace_tags(self, extra_tags: Optional[dict] = None):
        tags = {**self.default_tags}
        if extra_tags:
            tags.update(extra_tags)
        for key, value in tags.items():
            mlflow.update_current_trace(tags={key: str(value)})

    def _llm_request_body(self, request: ResponsesAgentRequest) -> dict:
        body = request.model_dump(exclude_none=True, exclude={"custom_inputs", "context", "stream", "store"})
        instructions = []
        input_items = []
        for item in body["input"]:
            if item.get("role") in ("system", "developer"):
                instructions.append(extract_request_text({"input": [{**item, "role": "user"}]}))
            else:
                input_items.append(item)
        body["input"] = input_items
        if instructions:
            body["instructions"] = "\n\n".join(instructions)
        body["model"] = self.llm_endpoint
        body.setdefault("max_output_tokens", self.max_tokens)
        body.setdefault("temperature", self.temperature)
        return body

    def _responses_api_path(self) -> str:
        api = self.config.get("llm", {}).get("responses_api", "open-responses")
        if api not in {"responses", "open-responses"}:
            raise ValueError("llm.responses_api must be 'responses' or 'open-responses'.")
        return f"/{api}"

    @mlflow.trace(name="agentops.model.responses", span_type="LLM")
    def _call_llm(self, request: ResponsesAgentRequest, context: RequestContext) -> ResponsesAgentResponse:
        """Call the Responses API without converting native output items."""
        from databricks.sdk import WorkspaceClient
        from openai.types.responses import Response

        client = WorkspaceClient().serving_endpoints.get_open_ai_client()
        response = client.post(
            self._responses_api_path(), cast_to=Response, body=self._llm_request_body(request),
        )
        if response.error:
            raise RuntimeError(f"Responses API failed: {response.error.message}")
        record_response_usage(response, context.state, self.llm_endpoint,
                              span=mlflow.get_current_active_span())
        return ResponsesAgentResponse(output=[item.model_dump(exclude_none=True) for item in response.output])

    @mlflow.trace(name="agentops.model.responses.stream", span_type="LLM")
    def _call_llm_stream(
        self, request: ResponsesAgentRequest, context: RequestContext,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Emit serving-compatible output events and capture provider usage."""
        from databricks.sdk import WorkspaceClient
        from openai import Stream
        from openai.types.responses import Response, ResponseStreamEvent

        client = WorkspaceClient().serving_endpoints.get_open_ai_client()
        stream = client.post(
            self._responses_api_path(), cast_to=Response,
            body={**self._llm_request_body(request), "stream": True},
            stream=True, stream_cls=Stream[ResponseStreamEvent],
        )
        completed = False
        with stream:
            for event in stream:
                if event.type in {"error", "response.failed", "response.incomplete"}:
                    raise RuntimeError(f"Responses stream failed: {event.model_dump(exclude_none=True)}")
                if event.type == "response.completed":
                    record_response_usage(event.response, context.state, self.llm_endpoint,
                                          span=mlflow.get_current_active_span())
                    completed = True
                # Playground accepts output events, not provider lifecycle envelopes.
                # MLflow's permissive event model does not enforce that UI contract.
                elif event.type in {
                    "response.output_text.delta", "response.output_text.annotation.added",
                    "response.output_item.done",
                }:
                    yield ResponsesAgentStreamEvent(**event.model_dump(exclude_none=True))
        if not completed:
            raise RuntimeError("Responses stream ended without a response.completed event.")

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Synchronous MLflow boundary over the same callback lifecycle."""
        return run_sync(self.apredict(request))

    async def apredict(self, request):
        return await self.execute(request, invoke=self._aprocess_request)

    async def _aprocess_request(self, request, context):
        """Override for native async hooks; synchronous hooks run off-loop."""
        return await run_blocking(self._process_request, request, context)

    @asynccontextmanager
    async def _callback_lifecycle(self, request, *, context, invoke, stream=None):
        async def normalized_invoke(prepared):
            from framework.inference.agent_middleware import wrap_target_result
            return wrap_target_result(await invoke(prepared, context))

        # The only scope is resource/deadline ownership, never an agent clone.
        async with context.scope(self.config.get('middleware', {}).get('timeout_seconds')):
            async with aclosing(self._run_lifecycle(request, streaming=stream is not None,
                context=context.state, invoke=normalized_invoke,
                stream=(lambda req: stream(req, context)) if stream is not None else None)) as lifecycle:
                yield lifecycle

    async def execute(self, request, *, invoke, context=None):
        """Run one prediction callback inside the shared middleware lifecycle."""
        context = context if context is not None else RequestContext()
        async with self._callback_lifecycle(request, context=context, invoke=invoke) as lifecycle:
            async for result in lifecycle:
                return result

    async def execute_stream(self, request, *, invoke, stream=None, context=None):
        """Use a stream callback or one prediction fallback, with the same checks."""
        context = context if context is not None else RequestContext()
        async with self._callback_lifecycle(request, context=context, invoke=invoke, stream=stream) as lifecycle:
            async for result in lifecycle:
                for event in self._stream_events(result):
                    yield event

    async def _run_lifecycle(self, request, *, streaming, context, invoke, stream=None):
        """One ordered middleware lifecycle for prediction and streaming."""
        start_time = time.time()
        messages = request.model_dump(exclude_none=True)["input"]
        custom_inputs = request.custom_inputs
        user_message = extract_request_text(request)
        state = context
        context_args = {"context": state}

        # Extract per-request tags, thread_id, and user_id
        request_tags = {}
        thread_id = ""
        user_id = ""
        if custom_inputs and isinstance(custom_inputs, dict):
            request_tags = custom_inputs.get("tags", {})
            # Support both thread_id (new) and session_id (legacy) naming
            thread_id = custom_inputs.get("thread_id", custom_inputs.get("session_id", ""))
            user_id = custom_inputs.get("user_id", "")
        if not thread_id:
            try:
                trace = mlflow.get_current_active_span()
                thread_id = trace.request_id if trace else ""
            except Exception:
                pass
        if not thread_id:
            thread_id = str(uuid.uuid4())
        self._set_trace_tags(request_tags)

        # ── Load session history FIRST (before guardrails) ──
        # History must load before guardrails so the intent check can see
        # conversation context from prior turns. Without this, follow-up
        # messages like "What happens if I set it too low?" get blocked
        # because they lack Databricks keywords on their own.
        has_client_history = len(messages) > 1
        if self.session_store.enabled and thread_id and not has_client_history:
            await self._aload_session_history(thread_id, messages, **context_args)
            messages = state["messages"]
            request = ResponsesAgentRequest.model_validate({**request.model_dump(exclude_none=True), "input": messages})
        elif has_client_history:
            mlflow.update_current_trace(tags={
                "agentops.session.history_turns": str(len(messages) - 1),
                "agentops.session.source": "client",
            })

        if self.guardrails_enabled:
            request = await self.guardrail_middleware.process_request(request)
            messages = request.model_dump(exclude_none=True)["input"]
            user_message = extract_request_text(request)
            conversation_context = " ".join(
                extract_request_text({"input": [message]})
                for message in messages[:-1] if message.get("role") == "user"
            )

        # ── Pre-LLM Guardrails ──
        # Runs AFTER session history load so the intent check can see
        # conversation context from prior turns via conversation_context.
        if self.guardrails_enabled:
            pre_result = await call_async_method(self.pre_llm_guardrails, "acheck", "check",
                user_message, conversation_context=conversation_context)
            if pre_result.get("blocked"):
                mlflow.update_current_trace(tags={
                    "agentops.guardrail.pre_llm.blocked": "true",
                    "agentops.guardrail.pre_llm.blocked_by": pre_result.get("blocked_by", "unknown"),
                })
                yield self._build_response(
                    pre_result.get("message", "Request blocked by safety filters."),
                    thread_id, user_id, **context_args,
                )
                return
            mlflow.update_current_trace(tags={
                "agentops.guardrail.pre_llm.blocked": "false",
                "agentops.guardrail.pre_llm.intent": pre_result.get("checks", {}).get("intent", {}).get("intent", "unknown"),
            })

        # ── Recall long-term user memories ──
        # If user_id is provided, search Lakebase for relevant facts about this user
        # and pass them to the agent in explicit request state.
        if self.session_store.memory_enabled and user_id and user_message:
            await self._arecall_long_term_memory(user_id, user_message, **context_args)

        # ── Agent Processing (subclass implements this) ──
        # Callbacks can populate context.state for shared postchecks.
        state["thread_id"] = thread_id
        state["user_id"] = user_id
        if streaming:
            full_output = []
            stream_custom = {}
            stream_usage = None
            async with aclosing(stream(request)) as upstream:
                async for event in upstream:
                    payload = event.model_dump(exclude_none=True)
                    if event.type in {"error", "response.failed", "response.incomplete"}:
                        raise RuntimeError(payload.get("message", "Target agent stream failed"))
                    if event.type == "response.in_progress":
                        # Status is a public target capability, separate from final output.
                        # Never release a target-supplied response body before checks.
                        if (request.custom_inputs or {}).get("stream_activity") is True:
                            activity = (event.custom_outputs or {}).get("activity")
                            yield ResponsesAgentStreamEvent(type="response.in_progress",
                                response={"id": "activity", "object": "response", "status": "in_progress", "output": []},
                                custom_outputs={"activity": activity} if activity else None)
                        continue
                    if event.type == "response.output_item.done":
                        stream_custom.update(event.custom_outputs or {})
                    envelope = payload.get("response") or {}
                    stream_usage = payload.get("usage") or envelope.get("usage") or stream_usage
                    if event.type == "response.output_item.done":
                        full_output.append(event.model_dump(exclude_none=True)["item"])
                        event = event.model_copy(update={"custom_outputs": {
                            **self._session_outputs(thread_id, user_id, **context_args),
                            **(event.custom_outputs or {}),
                        }})
                    if not self._buffers_stream_output():
                        yield event
            response = ResponsesAgentResponse(output=full_output, custom_outputs=stream_custom,
                **({"usage": stream_usage} if stream_usage is not None else {}))
        else:
            response = await invoke(request)

        # ── Post-LLM Guardrails ──
        if self.guardrails_enabled:
            response = await self.guardrail_middleware.process_response(response)
            response_text = extract_response_text(response)
            # User-visible structured artifacts are checked at the same outer
            # boundary as answer text, independent of which target produced them.
            artifact = (response.custom_outputs or {}).get("genie_result")
            checked_text = response_text + ("\n" + json.dumps(artifact, default=str) if artifact else "")
            post_result = await call_async_method(self.post_llm_guardrails, "acheck", "check", user_message, checked_text, state)
            if post_result.get("blocked"):
                mlflow.update_current_trace(tags={
                    "agentops.guardrail.post_llm.blocked": "true",
                    "agentops.guardrail.post_llm.blocked_by": post_result.get("blocked_by", "unknown"),
                })
                if not streaming or self._buffers_stream_output():
                    yield self._build_response(
                        post_result.get("message", "Response filtered for safety."),
                        thread_id, user_id, **context_args,
                    )
                    return
                logger.warning("Post-LLM guardrail flagged streamed response: %s", post_result.get("blocked_by"))
            else:
                mlflow.update_current_trace(tags={"agentops.guardrail.post_llm.blocked": "false"})

        # ── Tag with latency + token usage ──
        latency_ms = (time.time() - start_time) * 1000
        state["latency_ms"] = latency_ms
        state["trace_tags"] = emit_standard_trace_metrics(start_time, state,
            extra_tags={"agentops.streaming": "true"} if streaming else None)

        # ── Save session history ──
        if self.session_store.enabled:
            response_text = extract_response_text(response)
            trace_id = ""
            try:
                trace = mlflow.get_current_active_span()
                trace_id = trace.request_id if trace else ""
            except Exception:
                pass

            model_endpoint = getattr(self, "llm_endpoint", "")
            await call_async_method(self.session_store, "asave_full_session", "save_full_session",
                session_id=thread_id,
                messages=messages,
                response_text=response_text,
                response_items=[item.model_dump(exclude_none=True) for item in response.output],
                response_time_ms=latency_ms,
                model_endpoint=model_endpoint,
                trace_id=trace_id,
            )

        # ── Add custom_outputs (thread_id for follow-up requests) ──
        custom_outputs = self._session_outputs(thread_id, user_id, **context_args)
        # Diagnostic: surface store status for debugging (prefix with _ to avoid conflicts)
        store_err = getattr(self.session_store, '_store_error', None)
        if store_err:
            custom_outputs["_store_error"] = store_err[:200]
        uc_err = getattr(self.session_store, '_uc_last_error', None)
        if uc_err:
            custom_outputs["_uc_error"] = uc_err[:200]
        if response.custom_outputs:
            merged_tags = {**custom_outputs.get("tags", {}), **response.custom_outputs.get("tags", {})}
            custom_outputs.update(response.custom_outputs)
            if merged_tags:
                custom_outputs["tags"] = merged_tags
        response.custom_outputs = custom_outputs

        if not streaming or self._buffers_stream_output():
            yield response

    @mlflow.trace(name="load_session_history", span_type="RETRIEVER")
    def _load_session_history(
        self, thread_id: str, messages: list[dict], *, context
    ) -> dict:
        """Load conversation history from Lakebase and prepend to messages.

        Traced as a RETRIEVER span — returns a dict with diagnostic info
        so the loaded history is clearly visible in the MLflow trace output.
        """
        prior_turns = self.session_store.get_history(thread_id, max_turns=10)
        return self._apply_session_history(thread_id, messages, prior_turns, context=context)

    async def _aload_session_history(self, thread_id, messages, *, context):
        from inspect import iscoroutinefunction
        method = getattr(self.session_store, "aget_history", None)
        if iscoroutinefunction(method):
            prior_turns = await method(thread_id, max_turns=10)
            return self._apply_session_history(thread_id, messages, prior_turns, context=context)
        return await run_blocking(self._load_session_history, thread_id, messages,
            context=context)

    def _apply_session_history(self, thread_id, messages, prior_turns, *, context):
        # Surface store connection status
        store_ok = self.session_store._store is not None
        store_err = getattr(self.session_store, '_store_error', None)
        mlflow.update_current_trace(tags={
            "agentops.session.store_connected": str(store_ok),
        })
        if store_err:
            mlflow.update_current_trace(tags={
                "agentops.session.store_error": store_err[:200],
            })

        if prior_turns:
            history_msgs = list(prior_turns)
            context["messages"] = history_msgs + list(messages)
            mlflow.update_current_trace(tags={
                "agentops.session.history_turns": str(len(prior_turns)),
                "agentops.session.source": "server_store",
            })
            return {
                "thread_id": thread_id,
                "store_connected": store_ok,
                "turns_loaded": len(prior_turns),
                "messages_loaded": [
                    {"role": turn.get("role", turn.get("type", "")), "content": str(turn.get("content", ""))[:100]}
                    for turn in prior_turns
                ],
            }
        else:
            context["messages"] = list(messages)
            mlflow.update_current_trace(tags={
                "agentops.session.history_turns": "0",
                "agentops.session.source": "server_store",
            })
            return {
                "thread_id": thread_id,
                "store_connected": store_ok,
                "store_error": store_err,
                "turns_loaded": 0,
            }

    @mlflow.trace(name="recall_user_memory", span_type="RETRIEVER")
    def _recall_long_term_memory(self, user_id: str, query: str, *, context) -> dict:
        """Search long-term user memory in Lakebase.

        Traced as a RETRIEVER span — returns a dict with diagnostic info
        so recalled memories are clearly visible in the MLflow trace output.
        """
        memories = self.session_store.recall_user_memories(user_id, query)
        return self._apply_recalled_memory(user_id, query, memories, context=context)

    async def _arecall_long_term_memory(self, user_id, query, *, context):
        from inspect import iscoroutinefunction
        method = getattr(self.session_store, "arecall_user_memories", None)
        if iscoroutinefunction(method):
            memories = await method(user_id, query)
            return self._apply_recalled_memory(user_id, query, memories, context=context)
        return await run_blocking(self._recall_long_term_memory, user_id, query,
            context=context)

    def _apply_recalled_memory(self, user_id, query, memories, *, context):
        if memories:
            context["user_memories"] = memories
            mlflow.update_current_trace(tags={
                "agentops.memory.recalled_count": str(len(memories)),
            })
        return {
            "user_id": user_id,
            "query": query[:100],
            "memories_found": len(memories),
            "memories": memories,
        }

    @staticmethod
    def _session_outputs(thread_id, user_id="", *, context):
        return {"thread_id": thread_id, **({"user_id": user_id} if user_id else {})}

    def _build_response(
        self, content: str, thread_id: str, user_id: str = "", *, context
    ) -> ResponsesAgentResponse:
        """Build a ResponsesAgentResponse with standard custom_outputs."""
        custom_outputs = self._session_outputs(thread_id, user_id, context=context)
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=content, id=str(uuid.uuid4()))],
            custom_outputs=custom_outputs,
        )

    def _process_request(
        self, request: ResponsesAgentRequest, context: RequestContext,
    ) -> ResponsesAgentResponse:
        """Override this in your agent subclass."""
        raise NotImplementedError("Subclasses must implement _process_request")

    def _process_request_stream(
        self, request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Override this to emit Responses deltas and completed output items."""
        raise NotImplementedError("Subclasses must implement _process_request_stream for streaming")

    async def _aprocess_request_stream(self, request, context):
        """Native async integrations override this hook; legacy streams run off-loop."""
        async with aclosing(iterate_blocking(self._process_request_stream, request, context)) as upstream:
            async for event in upstream:
                yield event

    def _supports_streaming(self):
        return (
            type(self)._aprocess_request_stream is not AgentOPSBase._aprocess_request_stream
            or type(self)._process_request_stream is not AgentOPSBase._process_request_stream
        )

    def _buffers_stream_output(self):
        return self.guardrails_enabled and self.guardrail_middleware.buffers_output

    @staticmethod
    def _stream_events(result):
        if isinstance(result, ResponsesAgentResponse):
            for item in result.output:
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=item.model_dump(exclude_none=True),
                    custom_outputs=result.custom_outputs,
                    **({"usage": result.usage.model_dump(exclude_none=True)} if result.usage else {}),
                )
        else:
            yield result

    async def apredict_stream(self, request):
        async with aclosing(self.execute_stream(request, invoke=self._aprocess_request,
            stream=self._aprocess_request_stream if self._supports_streaming() else None)) as events:
            async for event in events:
                yield event

    def predict_stream(
        self, request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Thin synchronous MLflow adapter over the shared async middleware."""
        yield from iterate_async_sync(lambda: self.apredict_stream(request))
