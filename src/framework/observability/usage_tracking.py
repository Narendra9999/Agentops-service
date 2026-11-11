"""Request usage capture and trace metrics shared by agent runtimes.

Capture adapters choose one source per invocation; all sources use the same
normalization and accounting path. Model invocation errors are never retried here.
"""
import logging

import mlflow
import time
from typing import Optional

from framework.observability.token_usage import as_dict, normalize_usage, response_usage

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


logger = logging.getLogger(__name__)

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



def record_usage(usages, request_context, model_name=""):
    """Normalize and record one batch of model-call usage exactly once."""
    if request_context is None:
        return False
    records = []
    for usage in usages:
        normalized = normalize_usage(usage)
        if normalized and normalized["total_tokens"] > 0:
            records.append({"prompt_tokens": normalized["input_tokens"],
                            "completion_tokens": normalized["output_tokens"],
                            "total_tokens": normalized["total_tokens"],
                            "cached_tokens": normalized["cached_tokens"],
                            "reasoning_tokens": normalized["reasoning_tokens"]})
    if not records:
        return False
    if TokenTracker is not None:
        tracker = TokenTracker(model_name=model_name)
        for record in records:
            tracker.track({"usage": record})
        summary = tracker.to_dict()
        request_context["token_usage"] = tracker
    else:
        summary = {key: sum(record[key] for record in records) for key in records[0]}
        summary.update(requests=len(records), total_cost_usd=0.0)
    request_context.setdefault("token_usage_history", []).append(summary)
    return True


def record_response_usage(response, request_context=None, model_name=""):
    return record_usage([response_usage(response)], request_context, model_name)


class _UsageCaptureCallback(BaseCallbackHandler):
    """Capture every model result, including chains ending in output parsers."""

    def __init__(self):
        self.usages = []

    def on_llm_end(self, response, **kwargs):
        output = as_dict(getattr(response, "llm_output", None))
        usage = response_usage(output) or normalize_usage(output)
        if usage is not None:
            self.usages.append(usage)
            return
        for group in getattr(response, "generations", None) or []:
            for generation in group if isinstance(group, list) else [group]:
                message = getattr(generation, "message", None)
                usage = response_usage(message)
                if usage is None:
                    usage = normalize_usage(as_dict(getattr(message, "response_metadata", None)))
                if usage is not None:
                    self.usages.append(usage)
                    return


def _record_invocation_usage(callback, capture, result, request_context, model_name):
    metadata = getattr(callback, "usage_metadata", None) or {}
    for usages in (metadata.values(), capture.usages, [response_usage(result)]):
        if record_usage(usages, request_context, model_name):
            break


def invoke_with_usage_tracking(runnable, payload, request_context=None, model_name=""):
    """Invoke once; prefer standard metadata, then callback, then response usage."""
    from contextlib import nullcontext

    capture = _UsageCaptureCallback()
    manager = get_usage_metadata_callback() if get_usage_metadata_callback else nullcontext(None)
    with manager as callback:
        result = runnable.invoke(payload, config={"callbacks": [capture]})
    _record_invocation_usage(callback, capture, result, request_context, model_name)
    return result


async def ainvoke_with_usage_tracking(runnable, payload, request_context=None, model_name=""):
    """Use LangChain's native ainvoke with the same single-source accounting."""
    from contextlib import nullcontext

    capture = _UsageCaptureCallback()
    manager = get_usage_metadata_callback() if get_usage_metadata_callback else nullcontext(None)
    with manager as callback:
        result = await runnable.ainvoke(payload, config={"callbacks": [capture]})
    _record_invocation_usage(callback, capture, result, request_context, model_name)
    return result
