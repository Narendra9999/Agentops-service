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

try:
    from langchain_core.callbacks import get_usage_metadata_callback
except Exception:
    get_usage_metadata_callback = None

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except Exception:
    BaseCallbackHandler = object

try:
    from tools.token_tracker import TokenTracker
except Exception:
    try:
        from token_tracker import TokenTracker
    except Exception:
        TokenTracker = None

from framework.guardrails.pre_llm import PreLLMGuardrails
from framework.guardrails.post_llm import PostLLMGuardrails
from framework.session.session_store import SessionStore
from framework.inference import extract_request_text, extract_response_text

logger = logging.getLogger(__name__)


class _UsageCaptureCallback(BaseCallbackHandler):
    """Capture token usage from raw LangChain callback payloads.

    This is a fallback for providers/chains where usage metadata callback is empty,
    especially for ChatDatabricks chains that end with output parsers.
    """

    def __init__(self):
        self.usages = []

    def on_llm_end(self, response, **kwargs):
        self._capture_from_callback_response(response)

    def on_chat_model_end(self, response, **kwargs):
        self._capture_from_callback_response(response)

    def _capture_from_callback_response(self, response):
        usage = _extract_usage_from_llm_callback_response(response)
        if usage:
            self.usages.append(usage)


def emit_standard_trace_metrics(
    start_time: float,
    request_context: Optional[dict] = None,
    status: str = "success",
    extra_tags: Optional[dict] = None,
) -> dict:
    """Emit the standard AgentOPS trace metrics in one MLflow update call.

    This is the shared entry point for team agents that are not direct subclasses
    of AgentOPSBase. It centralizes latency, status, and token-cost tags so team
    notebooks only need to populate request_context["token_usage"].
    """
    trace_tags = {
        "agentops.latency_ms": str(round((time.time() - start_time) * 1000, 2)),
        "agentops.status": status,
    }

    if request_context:
        thread_id = request_context.get("thread_id", "")
        user_id = request_context.get("user_id", "")
        if thread_id:
            trace_tags["agentops.thread_id"] = str(thread_id)
        if user_id:
            trace_tags["agentops.user_id"] = str(user_id)

        token_history = request_context.get("token_usage_history")
        token_usage = request_context.get("token_usage")

        if token_history:
            if not isinstance(token_history, (list, tuple)):
                token_history = [token_history]

            total_prompt = total_completion = total_reasoning = total_cached = total_tokens = 0
            total_cost = 0.0
            llm_requests = 0
            for usage in token_history:
                if hasattr(usage, "to_dict"):
                    usage = usage.to_dict()
                if isinstance(usage, dict):
                    total_prompt += int(usage.get("prompt_tokens", 0) or 0)
                    total_completion += int(usage.get("completion_tokens", 0) or 0)
                    total_reasoning += int(usage.get("reasoning_tokens", usage.get("thinking_tokens", 0)) or 0)
                    total_cached += int(usage.get("cached_tokens", 0) or 0)
                    total_tokens += int(usage.get("total_tokens", 0) or 0)
                    total_cost += float(usage.get("total_cost_usd", usage.get("cost_usd", 0.0)) or 0.0)
                    llm_requests += int(usage.get("requests", 1) or 1)

            trace_tags.update({
                "agentops.tokens.prompt": str(total_prompt),
                "agentops.tokens.completion": str(total_completion),
                "agentops.tokens.total": str(total_tokens),
                "agentops.cost_usd": f"{total_cost:.6f}",
                "agentops.llm_requests": str(llm_requests),
            })
            if total_reasoning > 0:
                trace_tags["agentops.tokens.reasoning"] = str(total_reasoning)
            if total_cached > 0:
                trace_tags["agentops.tokens.cached"] = str(total_cached)
        elif token_usage and hasattr(token_usage, "to_tags"):
            trace_tags.update(token_usage.to_tags())

    if extra_tags:
        trace_tags.update(extra_tags)

    try:
        mlflow.update_current_trace(tags=trace_tags)
    except Exception as exc:
        logger.warning("Could not attach standard trace metrics: %s", exc)

    return trace_tags


def invoke_with_usage_tracking(runnable, payload, request_context: Optional[dict] = None, model_name: str = ""):
    """Invoke a LangChain runnable and record usage in request_context.

    Token counts are captured via LangChain callbacks regardless of whether
    TokenTracker is available. Without TokenTracker, raw prompt/completion/total
    counts are stored as plain dicts so emit_standard_trace_metrics can still
    emit agentops.tokens.* and agentops.cost_usd trace tags.
    """
    result = None
    usage_recorded = False
    # Do NOT bail out when TokenTracker is None — raw dict tracking still works.
    
    capture_cb = _UsageCaptureCallback()
    invoke_config = {"callbacks": [capture_cb]}

    if get_usage_metadata_callback is not None:
        # 1) Primary path: ambient usage metadata callback
        try:
            with get_usage_metadata_callback() as callback:
                try:
                    result = runnable.invoke(payload, config=invoke_config)
                except TypeError:
                    result = runnable.invoke(payload)

            usage_recorded = _track_llm_usage_from_usage_metadata(
                callback=callback,
                request_context=request_context,
                model_name=model_name,
            )
        except Exception as exc:
            logger.debug("Usage metadata callback failed: %s. Falling back to handler/response extraction.", exc)
            if result is None:
                try:
                    result = runnable.invoke(payload, config=invoke_config)
                except TypeError:
                    result = runnable.invoke(payload)
    else:
        # No ambient usage callback available, invoke with direct callback handler.
        try:
            result = runnable.invoke(payload, config=invoke_config)
        except TypeError:
            result = runnable.invoke(payload)

    # 2) Fallback path: callback-level extraction from LLM/chat model events
    if request_context is not None and not usage_recorded and capture_cb.usages:
        usage_recorded = _track_llm_usage_from_usage_list(
            usage_list=capture_cb.usages,
            request_context=request_context,
            model_name=model_name,
        )

    # 3) Final fallback: direct response extraction if result carries usage metadata
    if request_context is not None and not usage_recorded:
        _track_llm_usage_from_response(result, request_context, model_name)
    
    return result


def _normalize_usage_dict(usage: dict) -> dict:
    """Normalize Responses/OpenAI and LangChain counts to TokenTracker fields."""
    from framework.observability.token_usage import normalize_usage
    normalized = normalize_usage(usage)
    if normalized is None:
        return {}
    return {"prompt_tokens": normalized["input_tokens"],
            "completion_tokens": normalized["output_tokens"],
            "total_tokens": normalized["total_tokens"],
            "cached_tokens": normalized["cached_tokens"],
            "reasoning_tokens": normalized["reasoning_tokens"]}


def _track_llm_usage_from_usage_metadata(callback, request_context: Optional[dict], model_name: str) -> bool:
    """Track usage from langchain_core get_usage_metadata_callback()."""
    if callback is None or request_context is None:
        return False

    usage_by_model = getattr(callback, "usage_metadata", None) or {}
    if not usage_by_model:
        return False

    if TokenTracker is not None:
        tracker = TokenTracker(model_name=model_name)
        for usage in usage_by_model.values():
            normalized = _normalize_usage_dict(usage if isinstance(usage, dict) else {})
            if normalized.get("total_tokens", 0) > 0:
                tracker.track({"usage": normalized})
        if tracker.requests <= 0:
            return False
        request_context.setdefault("token_usage_history", []).append(tracker.to_dict())
        request_context["token_usage"] = tracker
        return True

    # TokenTracker unavailable — accumulate raw dict counts so emit_standard_trace_metrics
    # can still emit agentops.tokens.* tags without a cost calculation.
    agg = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
           "total_cost_usd": 0.0, "requests": 0}
    for usage in usage_by_model.values():
        norm = _normalize_usage_dict(usage if isinstance(usage, dict) else {})
        agg["prompt_tokens"]    += norm.get("prompt_tokens", 0)
        agg["completion_tokens"] += norm.get("completion_tokens", 0)
        agg["total_tokens"]      += norm.get("total_tokens", 0) or (
            norm.get("prompt_tokens", 0) + norm.get("completion_tokens", 0))
        agg["requests"] += 1
    if agg["requests"] == 0 or (agg["prompt_tokens"] == 0 and agg["completion_tokens"] == 0):
        return False
    request_context.setdefault("token_usage_history", []).append(agg)
    return True


def _track_llm_usage_from_usage_list(usage_list: list, request_context: Optional[dict], model_name: str) -> bool:
    """Track usage from callback handler captured usage payloads."""
    if request_context is None or not usage_list:
        return False

    if TokenTracker is not None:
        tracker = TokenTracker(model_name=model_name)
        for usage in usage_list:
            normalized = _normalize_usage_dict(usage if isinstance(usage, dict) else {})
            if normalized.get("total_tokens", 0) > 0:
                tracker.track({"usage": normalized})
        if tracker.requests <= 0:
            return False
        request_context.setdefault("token_usage_history", []).append(tracker.to_dict())
        request_context["token_usage"] = tracker
        return True

    # TokenTracker unavailable — raw dict fallback.
    agg = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
           "total_cost_usd": 0.0, "requests": 0}
    for usage in usage_list:
        norm = _normalize_usage_dict(usage if isinstance(usage, dict) else {})
        agg["prompt_tokens"]    += norm.get("prompt_tokens", 0)
        agg["completion_tokens"] += norm.get("completion_tokens", 0)
        agg["total_tokens"]      += norm.get("total_tokens", 0) or (
            norm.get("prompt_tokens", 0) + norm.get("completion_tokens", 0))
        agg["requests"] += 1
    if agg["requests"] == 0 or (agg["prompt_tokens"] == 0 and agg["completion_tokens"] == 0):
        return False
    request_context.setdefault("token_usage_history", []).append(agg)
    return True


def _extract_usage_from_llm_callback_response(response) -> Optional[dict]:
    """Extract usage payload from LangChain callback response objects (LLMResult/ChatResult)."""
    if response is None:
        return None

    # Most providers attach usage to llm_output (OpenAI-compatible token_usage/usage)
    llm_output = getattr(response, "llm_output", None) or {}
    if isinstance(llm_output, dict):
        usage = llm_output.get("usage") or llm_output.get("token_usage")
        if isinstance(usage, dict):
            return usage
        # langchain_community ChatDatabricks puts counts directly at the top level
        # e.g. {"prompt_tokens": 106, "completion_tokens": 10, "total_tokens": 116}
        if "prompt_tokens" in llm_output or "completion_tokens" in llm_output:
            return {k: v for k, v in llm_output.items()
                    if k in ("prompt_tokens", "completion_tokens", "total_tokens",
                             "reasoning_tokens", "cached_tokens")}

    # Chat providers can attach usage to generation message response_metadata.
    # ChatDatabricks often uses response_metadata["token_usage"].
    generations = getattr(response, "generations", None) or []
    for generation_group in generations:
        candidates = generation_group if isinstance(generation_group, list) else [generation_group]
        for generation in candidates:
            message = getattr(generation, "message", None)
            if message is None:
                continue
            usage_metadata = getattr(message, "usage_metadata", None)
            if isinstance(usage_metadata, dict) and usage_metadata:
                return usage_metadata
            response_metadata = getattr(message, "response_metadata", None) or {}
            if isinstance(response_metadata, dict):
                usage = response_metadata.get("usage") or response_metadata.get("token_usage")
                if isinstance(usage, dict):
                    return usage
                # Direct top-level format (langchain_community ChatDatabricks)
                if "prompt_tokens" in response_metadata or "completion_tokens" in response_metadata:
                    return {k: v for k, v in response_metadata.items()
                            if k in ("prompt_tokens", "completion_tokens", "total_tokens",
                                     "reasoning_tokens", "cached_tokens")}

    return None


def _track_llm_usage_from_response(
    response,
    request_context: Optional[dict] = None,
    model_name: str = "",
) -> bool:
    """Extract usage from response payload and append it to request_context.

    Supports Databricks/OpenAI-like payloads that expose usage as either:
    - response.usage
    - response.response_metadata["usage"]
    - response.response_metadata["token_usage"]
    - response["usage"]
    - response["token_usage"]
    """
    if request_context is None or response is None:
        return False

    from framework.observability.token_usage import response_usage
    usage = response_usage(response)

    if usage is None and hasattr(response, "usage"):
        usage = getattr(response, "usage")

    if usage is None and hasattr(response, "response_metadata"):
        response_metadata = getattr(response, "response_metadata") or {}
        if isinstance(response_metadata, dict):
            usage = response_metadata.get("usage") or response_metadata.get("token_usage")

    if usage is None and isinstance(response, dict):
        usage = response.get("usage") or response.get("token_usage")
        if usage is None:
            response_metadata = response.get("response_metadata") or {}
            if isinstance(response_metadata, dict):
                usage = response_metadata.get("usage") or response_metadata.get("token_usage")

    if usage is None:
        return False

    if hasattr(usage, "model_dump"):
        usage = usage.model_dump(exclude_none=True)
    elif hasattr(usage, "as_dict"):
        usage = usage.as_dict()
    elif hasattr(usage, "__dict__") and not isinstance(usage, dict):
        usage = dict(vars(usage))

    if not isinstance(usage, dict):
        return False

    usage = _normalize_usage_dict(usage)
    if usage.get("total_tokens", 0) <= 0:
        return False

    if TokenTracker is None:
        request_context.setdefault("token_usage_history", []).append({
            **usage, "requests": 1, "total_cost_usd": 0.0,
        })
        return True

    tracker = TokenTracker(model_name=model_name)
    tracker.track({"usage": usage})

    if tracker.requests <= 0:
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        if total_tokens <= 0 and (prompt_tokens > 0 or completion_tokens > 0):
            total_tokens = prompt_tokens + completion_tokens
        if total_tokens <= 0:
            return False

        tracker.prompt_tokens = prompt_tokens
        tracker.completion_tokens = completion_tokens
        tracker.total_tokens = total_tokens
        tracker.requests = 1

    request_context.setdefault("token_usage_history", []).append(tracker.to_dict())
    request_context["token_usage"] = tracker
    return True


class AgentOPSBase(ResponsesAgent):
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

    def _record_token_usage(self, tracker) -> None:
        if tracker is None:
            return

        if not hasattr(self, "_request_context") or self._request_context is None:
            self._request_context = {}

        self._request_context.setdefault("token_usage_history", []).append(tracker.to_dict())
        self._request_context["token_usage"] = tracker

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
        _track_llm_usage_from_response(response, self._request_context, self.llm_endpoint)
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
                    _track_llm_usage_from_response(event.response, self._request_context, self.llm_endpoint)
                    completed = True
                elif event.type.startswith("response."):
                    yield ResponsesAgentStreamEvent(**event.model_dump(exclude_none=True))
        if not completed:
            raise RuntimeError("Responses stream ended without a response.completed event.")

    def predict(
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
            self._load_session_history(thread_id, messages)
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
            pre_result = self.pre_llm_guardrails.check(
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
            self._recall_long_term_memory(user_id, user_message)

        # ── Agent Processing (subclass implements this) ──
        # _request_context is a plain dict that subclasses can populate
        # with data for post-LLM guardrails (e.g., retrieved_docs).
        self._request_context["thread_id"] = thread_id
        self._request_context["user_id"] = user_id
        response = self._process_request(request)

        # ── Post-LLM Guardrails ──
        if self.guardrails_enabled:
            response_text = extract_response_text(response)
            post_result = self.post_llm_guardrails.check(user_message, response_text, self._request_context)
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
            self.session_store.save_full_session(
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
