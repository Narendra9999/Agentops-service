"""
Cost-Tracking wrapper model — framework entry point for all team agents.

This file is the MLflow python_model. It is always staged at the root of the
MLflow artifact temp directory, while the team's Agent.py is staged inside
code_paths (e.g. <temp>/code/agent/Agent.py).

AgentOPS owns the outer lifecycle for every target. Supported targets:

  - Plain ResponsesAgent / AsyncResponsesAgent: invoke public agent methods.
  - Context-aware AgentOPSBase: invoke raw hooks, applying the middleware only once.
  - PythonModel: adapt predict(context, model_input) and normalize its result.

Guardrails, history, request tracing and persistence belong to the shared
AgentOPSMiddleware, not to a supervisor-specific team wrapper.

The team entry file is Agent.py, or inference.target_module when absent.
ChatModel is the default class;
inference.target_class can select an explicit alternate implementation.
inference.target_protocol selects responses or chat_completions.
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


def _resolve_inner_model(config=None):
    """Prefer packaged Agent.py; use target_module only when Agent.py is absent."""
    import sys
    from pathlib import Path
    from framework.inference.target_loader import team_target_entry

    config = config or {}
    options = config.get('inference') or {}
    class_name = options.get('target_class', 'ChatModel')
    roots = []
    for entry in sys.path:
        for root in (Path(entry), Path(entry) / 'agent'):
            if root not in roots:
                roots.append(root)
    # Search all staged roots for the conventional entry point first.
    conventional = next(((root, root / 'Agent.py') for root in roots
                         if (root / 'Agent.py').is_file()), None)
    selected = conventional
    if selected is None:
        for root in roots:
            path = team_target_entry(root, config)
            if path is not None:
                selected = root, path
                break
    if selected is None:
        raise ImportError('No packaged Agent.py or inference.target_module found. '
                          'Include the team agent directory in code_paths.')
    root, path = selected
    module_name = '_team_agent' if conventional else options['target_module']
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load team entry point: {path}')
    module = importlib.util.module_from_spec(spec)
    # Support sibling imports and package-relative imports from staged modules.
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(module)
        cls = getattr(module, class_name, None)
        if cls is None:
            raise ImportError(f'{path} does not expose inference.target_class={class_name!r}')
        logger.info('Resolved target %s from %s', class_name, path)
        return cls()
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    finally:
        sys.path.remove(str(root))


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
        from framework.observability.tracing import configure_native_instrumentation
        config = getattr(context, "model_config", None)
        if isinstance(config, dict):
            configure_native_instrumentation(config)
        self._inner = _resolve_inner_model(config)
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
        response = await self._middleware.execute(request, invoke=self._middleware.adapter.invoke)
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
            adapter = self._middleware.adapter
            async with aclosing(self._middleware.execute_stream(request, invoke=adapter.invoke,
                stream=adapter.stream if adapter.supports_streaming else None)) as stream:
                async for event in stream:
                    if response_usage(event) is not None:
                        last_usage = event
                    yield event
        finally:
            if last_usage is not None:
                _record_usage(last_usage)


mlflow.models.set_model(model=CostTrackingResponsesModel())
