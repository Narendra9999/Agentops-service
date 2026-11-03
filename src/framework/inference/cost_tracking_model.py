"""
Cost-Tracking wrapper model — framework entry point for all team agents.

This file is the MLflow python_model. It is always staged at the root of the
MLflow artifact temp directory, while the team's Agent.py is staged inside
code_paths (e.g. <temp>/code/agent/Agent.py).

Supports two ChatModel patterns — the framework auto-detects which is used:

  Pattern A — AgentOPSBase (recommended):
      class ChatModel(AgentOPSBase): ...
    Full framework: guardrails, token tracking, session history, trace tags.
    Team calls invoke_with_usage_tracking() for automatic token capture.

  Pattern B — Plain pyfunc (custom / non-LangChain models):
      class ChatModel(mlflow.pyfunc.PythonModel): ...
    predict(context, model_input) signature. Framework wraps the raw output
    into ResponsesAgentResponse so serving gets a standard response.
    Token tracking is NOT automatic — team must populate
    request_context["token_usage_history"] manually if desired.

No team changes required to switch between patterns — the class name
"ChatModel" and the file name "Agent.py" are the only contracts.
"""

import os
import logging
import importlib.util
import uuid

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


def _wrap_pyfunc_result(result):
    """Convert plain pyfunc output into a ResponsesAgentResponse.

    Handles the common return types a PythonModel.predict() might produce:
    - Already a ResponsesAgentResponse    → returned as-is
      - str                                 → wrapped in a single assistant message
      - dict with "content" / "output" key  → content extracted, then wrapped
      - pandas DataFrame (first cell)       → first cell stringified, then wrapped
      - anything else                       → str() conversion, then wrapped
    """
    if isinstance(result, ResponsesAgentResponse):
        return result

    if isinstance(result, str):
        content = result
    elif isinstance(result, dict):
        if isinstance(result.get("output"), list):
            return ResponsesAgentResponse(**result)
        content = (
            result.get("content")
            or result.get("output")
            or result.get("answer")
            or result.get("response")
            or str(result)
        )
    else:
        try:
            import pandas as pd
            if isinstance(result, pd.DataFrame) and not result.empty:
                content = str(result.iloc[0, 0])
            else:
                content = str(result)
        except ImportError:
            content = str(result)

    return ResponsesAgentResponse(
        output=[ResponsesAgent.create_text_output_item(text=str(content), id=str(uuid.uuid4()))]
    )


class CostTrackingResponsesModel(ResponsesAgent):
    """Responses serving wrapper for native agents and PythonModel implementations.

    At load_context time it detects which pattern the team's ChatModel uses and
    routes predict / predict_stream accordingly — no team changes required.
    """

    def load_context(self, context):
        self._inner = _resolve_inner_model()
        self._inner_is_responses_agent = isinstance(self._inner, ResponsesAgent)
        logger.info(
            "Inner model type: %s (%s)",
            type(self._inner).__name__,
            "ResponsesAgent" if self._inner_is_responses_agent else "PythonModel/pyfunc",
        )
        if hasattr(self._inner, "load_context"):
            self._inner.load_context(context)

    # ── Inference entry points ────────────────────────────────────────────────

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        if self._inner_is_responses_agent:
            return self._inner.predict(request)
        result = self._inner.predict(
            context=None, model_input=request.model_dump(exclude_none=True),
        )
        return _wrap_pyfunc_result(result)

    def predict_stream(
        self, request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        if self._inner_is_responses_agent:
            yield from self._inner.predict_stream(request)
        else:
            response = self.predict(request)
            for item in response.output:
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done", item=item.model_dump(exclude_none=True),
                    custom_outputs=response.custom_outputs,
                )


mlflow.models.set_model(model=CostTrackingResponsesModel())
