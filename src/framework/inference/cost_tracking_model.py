"""
Cost-Tracking wrapper model — framework entry point for all team agents.

This file is the MLflow python_model. It is always staged at the root of the
MLflow artifact temp directory, while the team's Agent.py is staged inside
code_paths (e.g. <temp>/code/agent/Agent.py).

AgentOPS owns the outer lifecycle for every target. Supported targets:

  - Plain ResponsesAgent / AsyncResponsesAgent: invoke public agent methods.
  - Legacy AgentOPSBase: invoke raw hooks, applying the middleware only once.
  - PythonModel: adapt predict(context, model_input) and normalize its result.

Guardrails, history, request tracing and persistence belong to the shared
AgentOPSMiddleware, not to a supervisor-specific team wrapper.

No team changes required to switch between patterns — the class name
"ChatModel" and the file name "Agent.py" are the only contracts.
"""

import os
import logging
import importlib.util

import mlflow
import mlflow.pyfunc
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from typing import Generator

logger = logging.getLogger(__name__)


def _resolve_inner_model():
    """Locate the team's Agent.py via sys.path and return a ChatModel instance."""
    import sys as _sys

    failures = []
    for sp in _sys.path:
        for rel in ("Agent.py", os.path.join("agent", "Agent.py")):
            fp = os.path.join(sp, rel)
            if not os.path.isfile(fp):
                continue
            try:
                spec = importlib.util.spec_from_file_location("_team_agent", fp)
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                cls = getattr(mod, "ChatModel", None)
                if cls is not None:
                    logger.info("Resolved inner agent from: %s", fp)
                    return cls()
            except Exception as exc:
                failures.append(f"{fp}: {type(exc).__name__}: {exc}")
                logger.warning("Could not load agent from %s: %s", fp, exc)
                continue

    failure_details = "\n".join(f"  - {failure}" for failure in failures)
    raise ImportError(
        "Could not find Agent.py with a ChatModel class in any code_paths directory. "
        "Ensure the team agent directory is included in code_paths when logging the model."
        + (f" Candidate load failures:\n{failure_details}" if failure_details else "")
    )


# Compatibility import for integrations using the old result normalizer.
from framework.inference.agent_middleware import AgentOPSMiddleware, wrap_target_result as _wrap_pyfunc_result


def _record_usage(response):
    try:
        from framework.observability.token_usage import emit_response_usage
        emit_response_usage(response)
    except Exception:
        logger.warning("Response usage tag export failed", exc_info=False)


from framework.inference.async_runtime import AsyncResponsesAgent


class CostTrackingResponsesModel(AsyncResponsesAgent):
    """Responses serving wrapper for native agents and PythonModel implementations.

    At load_context time it detects which pattern the team's ChatModel uses and
    routes predict / predict_stream accordingly — no team changes required.
    """

    def load_context(self, context):
        self._inner = _resolve_inner_model()
        if hasattr(self._inner, "load_context"):
            self._inner.load_context(context)
        config = getattr(context, "model_config", None)
        if not isinstance(config, dict):
            config = getattr(self._inner, "config", {})
        self._middleware = AgentOPSMiddleware(config, self._inner)

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        from framework.inference.async_runtime import run_sync
        return run_sync(self.apredict(request))

    async def apredict(self, request):
        response = await self._middleware.apredict(request)
        _record_usage(response)
        return response

    @mlflow.trace(
        name="agentops.request.stream", span_type="AGENT",
        output_reducer=ResponsesAgent.responses_agent_output_reducer,
    )
    def predict_stream(
        self, request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        from framework.inference.async_runtime import iterate_async_sync
        yield from iterate_async_sync(lambda: self.apredict_stream(request))

    async def apredict_stream(self, request):
        from contextlib import aclosing
        from framework.observability.token_usage import response_usage
        last_usage = None
        try:
            async with aclosing(self._middleware.apredict_stream(request)) as stream:
                async for event in stream:
                    if response_usage(event) is not None:
                        last_usage = event
                    yield event
        finally:
            if last_usage is not None:
                _record_usage(last_usage)


mlflow.models.set_model(model=CostTrackingResponsesModel())
