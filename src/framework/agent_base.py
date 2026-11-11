"""
AgentOPS Framework — Base Agent Class
All agents must extend this. Provides standardized guardrails,
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

# Public helpers remain importable here for existing team integrations.
from framework.observability.usage_tracking import (
    emit_standard_trace_metrics,
    invoke_with_usage_tracking,
    record_response_usage,
)

from copy import copy
from framework.inference.async_runtime import AsyncResponsesAgent, call_async_method, run_blocking, run_sync

from framework.guardrails.pre_llm import PreLLMGuardrails
from framework.guardrails.post_llm import PostLLMGuardrails
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
      - Long-term: cross-session user memory recalled into _request_context["user_memories"]

    Clients pass thread_id and user_id via custom_inputs:
      {"thread_id": "abc-123", "user_id": "user@example.com"}
    Response includes custom_outputs with thread_id for follow-up requests.
    """

    def __init__(self, config: dict):
        self.config = config
        agent_cfg = config.get("agent", {})
        self.agent_name = agent_cfg.get("name", "unnamed_agent")
        self.agent_version = agent_cfg.get("version", "0.0.0")

        # Initialize guardrails from config
        gr_config = config.get("guardrails", {})
        self.guardrails_enabled = gr_config.get("enabled", True)
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

    def _async_request_agent(self):
        # The legacy lifecycle assigns request context on self. Clone the turn
        # so concurrent async requests cannot overwrite each other's state.
        turn = copy(self)
        turn._request_context = {}
        return turn

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

    def _call_llm(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Call the Responses API without converting native output items."""
        from databricks.sdk import WorkspaceClient
        from openai.types.responses import Response

        client = WorkspaceClient().serving_endpoints.get_open_ai_client()
        response = client.post(
            self._responses_api_path(), cast_to=Response, body=self._llm_request_body(request),
        )
        if response.error:
            raise RuntimeError(f"Responses API failed: {response.error.message}")
        record_response_usage(response, self._request_context, self.llm_endpoint)
        return ResponsesAgentResponse(output=[item.model_dump(exclude_none=True) for item in response.output])

    def _call_llm_stream(
        self, request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Forward native Responses events and capture final response usage."""
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
                    record_response_usage(event.response, self._request_context, self.llm_endpoint)
                    completed = True
                elif event.type.startswith("response."):
                    yield ResponsesAgentStreamEvent(**event.model_dump(exclude_none=True))
        if not completed:
            raise RuntimeError("Responses stream ended without a response.completed event.")

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Synchronous MLflow contract over the shared async request lifecycle."""
        # Direct synchronous callers retain the historical request context.
        return run_sync(self._apredict_turn(request))

    async def apredict(self, request):
        turn = self._async_request_agent()
        return await turn._apredict_turn(request)

    async def _aprocess_request(self, request):
        """Override for native async customer agents; legacy hooks run off-loop."""
        return await run_blocking(self._process_request, request)

    async def _apredict_turn(
        self, request: ResponsesAgentRequest,
    ) -> ResponsesAgentResponse:
        """
        Main prediction method with guardrail wrapping.
        Subclasses should NOT override this — override _process_request() instead.
        """
        start_time = time.time()
        messages = request.model_dump(exclude_none=True)["input"]
        custom_inputs = request.custom_inputs
        user_message = extract_request_text(request)
        self._request_context = {}

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
        conversation_context = ""
        if self.session_store.enabled and thread_id and not has_client_history:
            await run_blocking(self._load_session_history, thread_id, messages)
            messages = self._augmented_messages
            request = ResponsesAgentRequest.model_validate({**request.model_dump(exclude_none=True), "input": messages})
            # Build context string from prior user messages for intent detection
            if len(messages) > 1:
                conversation_context = " ".join(
                    extract_request_text({"input": [message]})
                    for message in messages[:-1] if message.get("role") == "user"
                )
        elif has_client_history:
            mlflow.update_current_trace(tags={
                "agentops.session.history_turns": str(len(messages) - 1),
                "agentops.session.source": "client",
            })

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
                return self._build_response(
                    pre_result.get("message", "Request blocked by safety filters."),
                    thread_id, user_id,
                )
            mlflow.update_current_trace(tags={
                "agentops.guardrail.pre_llm.blocked": "false",
                "agentops.guardrail.pre_llm.intent": pre_result.get("checks", {}).get("intent", {}).get("intent", "unknown"),
            })

        # ── Recall long-term user memories ──
        # If user_id is provided, search Lakebase for relevant facts about this user
        # and make them available to the agent via _request_context.
        if self.session_store.memory_enabled and user_id and user_message:
            await run_blocking(self._recall_long_term_memory, user_id, user_message)

        # ── Agent Processing (subclass implements this) ──
        # _request_context is a plain dict that subclasses can populate
        # with data for post-LLM guardrails (e.g., retrieved_docs).
        self._request_context["thread_id"] = thread_id
        self._request_context["user_id"] = user_id
        response = await self._aprocess_request(request)

        # ── Post-LLM Guardrails ──
        if self.guardrails_enabled:
            response_text = extract_response_text(response)
            post_result = await call_async_method(self.post_llm_guardrails, "acheck", "check", user_message, response_text, self._request_context)
            if post_result.get("blocked"):
                mlflow.update_current_trace(tags={
                    "agentops.guardrail.post_llm.blocked": "true",
                    "agentops.guardrail.post_llm.blocked_by": post_result.get("blocked_by", "unknown"),
                })
                return self._build_response(
                    post_result.get("message", "Response filtered for safety."),
                    thread_id, user_id,
                )
            mlflow.update_current_trace(tags={"agentops.guardrail.post_llm.blocked": "false"})

        # ── Tag with latency + token usage ──
        latency_ms = (time.time() - start_time) * 1000
        self._request_context["latency_ms"] = latency_ms
        emit_standard_trace_metrics(start_time, self._request_context)

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
            await run_blocking(self.session_store.save_full_session,
                session_id=thread_id,
                messages=messages,
                response_text=response_text,
                response_items=[item.model_dump(exclude_none=True) for item in response.output],
                response_time_ms=latency_ms,
                model_endpoint=model_endpoint,
                trace_id=trace_id,
            )

        # ── Add custom_outputs (thread_id for follow-up requests) ──
        custom_outputs = {"thread_id": thread_id}
        if user_id:
            custom_outputs["user_id"] = user_id
        # Diagnostic: surface store status for debugging (prefix with _ to avoid conflicts)
        store_err = getattr(self.session_store, '_store_error', None)
        if store_err:
            custom_outputs["_store_error"] = store_err[:200]
        uc_err = getattr(self.session_store, '_uc_last_error', None)
        if uc_err:
            custom_outputs["_uc_error"] = uc_err[:200]
        if response.custom_outputs:
            custom_outputs.update(response.custom_outputs)
        response.custom_outputs = custom_outputs

        return response

    @mlflow.trace(name="load_session_history", span_type="RETRIEVER")
    def _load_session_history(
        self, thread_id: str, messages: list[dict]
    ) -> dict:
        """Load conversation history from Lakebase and prepend to messages.

        Traced as a RETRIEVER span — returns a dict with diagnostic info
        so the loaded history is clearly visible in the MLflow trace output.
        """
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

        prior_turns = self.session_store.get_history(thread_id, max_turns=10)
        if prior_turns:
            history_msgs = list(prior_turns)
            self._augmented_messages = history_msgs + list(messages)
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
            self._augmented_messages = list(messages)
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
    def _recall_long_term_memory(self, user_id: str, query: str) -> dict:
        """Search long-term user memory in Lakebase.

        Traced as a RETRIEVER span — returns a dict with diagnostic info
        so recalled memories are clearly visible in the MLflow trace output.
        """
        memories = self.session_store.recall_user_memories(user_id, query)
        if memories:
            self._request_context["user_memories"] = memories
            mlflow.update_current_trace(tags={
                "agentops.memory.recalled_count": str(len(memories)),
            })
        return {
            "user_id": user_id,
            "query": query[:100],
            "memories_found": len(memories),
            "memories": memories,
        }

    def _build_response(
        self, content: str, thread_id: str, user_id: str = ""
    ) -> ResponsesAgentResponse:
        """Build a ResponsesAgentResponse with standard custom_outputs."""
        custom_outputs = {"thread_id": thread_id}
        if user_id:
            custom_outputs["user_id"] = user_id
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=content, id=str(uuid.uuid4()))],
            custom_outputs=custom_outputs,
        )

    def _process_request(
        self, request: ResponsesAgentRequest,
    ) -> ResponsesAgentResponse:
        """Override this in your agent subclass."""
        raise NotImplementedError("Subclasses must implement _process_request")

    def _process_request_stream(
        self, request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Override this to emit Responses deltas and completed output items."""
        raise NotImplementedError("Subclasses must implement _process_request_stream for streaming")

    def predict_stream(
        self, request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        if type(self)._process_request_stream is AgentOPSBase._process_request_stream:
            response = self.predict(request)
            for item in response.output:
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done", item=item.model_dump(exclude_none=True),
                    custom_outputs=response.custom_outputs,
                )
            return

        """
        Streaming prediction — same pre/post processing as predict(),
        but yields chunks as the LLM generates tokens.

        Pre-processing (history, guardrails, memory) runs fully before streaming.
        Post-processing (guardrails, save history) runs after streaming completes.
        """
        start_time = time.time()
        messages = request.model_dump(exclude_none=True)["input"]
        custom_inputs = request.custom_inputs
        user_message = extract_request_text(request)
        self._request_context = {}

        # Extract thread_id, user_id
        request_tags = {}
        thread_id = ""
        user_id = ""
        if custom_inputs and isinstance(custom_inputs, dict):
            request_tags = custom_inputs.get("tags", {})
            thread_id = custom_inputs.get("thread_id", custom_inputs.get("session_id", ""))
            user_id = custom_inputs.get("user_id", "")
        if not thread_id:
            thread_id = str(uuid.uuid4())
        self._set_trace_tags(request_tags)

        # ── Load session history ──
        self._request_context["thread_id"] = thread_id
        self._request_context["user_id"] = user_id
        has_client_history = len(messages) > 1
        conversation_context = ""
        if self.session_store.enabled and thread_id and not has_client_history:
            self._load_session_history(thread_id, messages)
            messages = self._augmented_messages
            request = ResponsesAgentRequest.model_validate({**request.model_dump(exclude_none=True), "input": messages})
            if len(messages) > 1:
                conversation_context = " ".join(
                    extract_request_text({"input": [message]})
                    for message in messages[:-1] if message.get("role") == "user"
                )

        # ── Pre-LLM Guardrails ──
        if self.guardrails_enabled:
            pre_result = self.pre_llm_guardrails.check(
                user_message, conversation_context=conversation_context)
            if pre_result.get("blocked"):
                mlflow.update_current_trace(tags={
                    "agentops.guardrail.pre_llm.blocked": "true",
                    "agentops.guardrail.pre_llm.blocked_by": pre_result.get("blocked_by", "unknown"),
                    "agentops.streaming": "true",
                })
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_text_output_item(
                        text=pre_result.get("message", "Request blocked."), id=str(uuid.uuid4()),
                    ),
                    custom_outputs={"thread_id": thread_id, **({"user_id": user_id} if user_id else {})},
                )
                return
            mlflow.update_current_trace(tags={
                "agentops.guardrail.pre_llm.blocked": "false",
                "agentops.guardrail.pre_llm.intent": pre_result.get("checks", {}).get("intent", {}).get("intent", "unknown"),
            })

        # ── Recall long-term memory ──
        if self.session_store.memory_enabled and user_id and user_message:
            self._recall_long_term_memory(user_id, user_message)

        # ── Stream agent response ──
        self._request_context["thread_id"] = thread_id
        self._request_context["user_id"] = user_id
        full_output = []

        for event in self._process_request_stream(request):
            if event.type == "response.output_item.done":
                full_output.append(event.model_dump(exclude_none=True)["item"])
                event = event.model_copy(update={"custom_outputs": {
                    "thread_id": thread_id, **({"user_id": user_id} if user_id else {}),
                    **(event.custom_outputs or {}),
                }})
            yield event

        response_text = extract_response_text({"output": full_output})

        # ── Post-LLM Guardrails ──
        # Runs after full response is assembled. Can't un-stream tokens already
        # sent, but tags the trace for monitoring/alerting and logs the violation.
        if self.guardrails_enabled:
            post_result = self.post_llm_guardrails.check(
                user_message, response_text, self._request_context)
            if post_result.get("blocked"):
                mlflow.update_current_trace(tags={
                    "agentops.guardrail.post_llm.blocked": "true",
                    "agentops.guardrail.post_llm.blocked_by": post_result.get("blocked_by", "unknown"),
                })
                logger.warning(f"Post-LLM guardrail blocked streamed response: {post_result.get('blocked_by')}")
            else:
                mlflow.update_current_trace(tags={"agentops.guardrail.post_llm.blocked": "false"})

        # ── Save session history ──
        latency_ms = (time.time() - start_time) * 1000
        if self.session_store.enabled:
            trace_id = ""
            try:
                trace = mlflow.get_current_active_span()
                trace_id = trace.request_id if trace else ""
            except Exception:
                pass
            model_endpoint = getattr(self, "llm_endpoint", "")
            self.session_store.save_full_session(
                session_id=thread_id,
                messages=messages,
                response_text=response_text,
                response_items=full_output,
                response_time_ms=latency_ms,
                model_endpoint=model_endpoint,
                trace_id=trace_id,
            )

        self._request_context["latency_ms"] = latency_ms
        emit_standard_trace_metrics(
            start_time,
            self._request_context,
            extra_tags={"agentops.streaming": "true"},
        )
