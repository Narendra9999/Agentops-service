"""AgentOPS owns inference; adapters invoke target logic without a second lifecycle."""
from contextlib import aclosing
from copy import copy
from inspect import isasyncgenfunction
import uuid
import mlflow

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from framework.agent_base import AgentOPSBase
from framework.inference.async_runtime import AsyncResponsesAgent, call_async_method, iterate_blocking


def wrap_target_result(result):
    """Normalize a PythonModel result at the adapter boundary."""
    if isinstance(result, ResponsesAgentResponse):
        return result
    if isinstance(result, dict):
        if isinstance(result.get('output'), list):
            return ResponsesAgentResponse(**result)
        content = (result.get('content') or result.get('output') or result.get('answer')
                   or result.get('response') or str(result))
    elif isinstance(result, str):
        content = result
    else:
        try:
            import pandas as pd
            content = str(result.iloc[0, 0]) if isinstance(result, pd.DataFrame) and not result.empty else str(result)
        except ImportError:
            content = str(result)
    return ResponsesAgentResponse(output=[ResponsesAgent.create_text_output_item(
        text=str(content), id=str(uuid.uuid4()))])


class TargetAgentAdapter:
    """Only invocation/format adaptation. No guardrails, persistence or routing."""

    def __init__(self, target):
        self.target = target
        self.legacy = isinstance(target, AgentOPSBase)
        self.responses = isinstance(target, ResponsesAgent)

    def for_request(self):
        # Never attach per-request state to the model shared by serving threads.
        target = self.target._async_request_agent() if self.legacy else copy(self.target)
        return type(self)(target)

    @property
    def supports_streaming(self):
        if self.legacy:
            return self.target._supports_streaming()
        if not self.responses:
            return False
        native = getattr(type(self.target), 'apredict_stream', None)
        sync = getattr(type(self.target), 'predict_stream', None)
        return ((native is not None and native is not AsyncResponsesAgent.apredict_stream)
                or (sync is not None and sync is not ResponsesAgent.predict_stream))

    def _bind(self, request, context, resources):
        if self.legacy:
            # Legacy hooks already use these fields. Do not invoke their public
            # predict methods, which would execute the middleware a second time.
            self.target._request_context = context
            self.target._request_resources = resources
            return request
        custom = dict(request.custom_inputs or {})
        custom['thread_id'] = context['thread_id']
        custom.pop('agentops_context', None)
        if context.get('user_id'):
            custom.setdefault('user_id', context['user_id'])
        if context.get('user_memories'):
            # Optional context contract; target agents need not inherit AgentOPS.
            custom['agentops_context'] = {'user_memories': context['user_memories']}
        return request.model_copy(update={'custom_inputs': custom})

    async def invoke(self, request, context, resources):
        request = self._bind(request, context, resources)
        if self.legacy:
            result = await self.target._aprocess_request(request)
        elif self.responses:
            result = await call_async_method(self.target, 'apredict', 'predict', request)
        else:
            result = await call_async_method(self.target, 'apredict', 'predict',
                context=None, model_input=request.model_dump(exclude_none=True))
        return wrap_target_result(result)

    async def stream(self, request, context, resources):
        request = self._bind(request, context, resources)
        if self.legacy:
            stream = self.target._aprocess_request_stream(request)
        else:
            native = getattr(self.target, 'apredict_stream', None)
            stream = (native(request) if isasyncgenfunction(native)
                      else iterate_blocking(self.target.predict_stream, request))
        async with aclosing(stream):
            async for event in stream:
                yield event


class AgentOPSMiddleware(AgentOPSBase):
    """Reusable outer lifecycle around any configured ResponsesAgent/PythonModel.

    Teams supply target behavior; the serving wrapper constructs this middleware.
    Existing AgentOPSBase targets are adapted at their raw processing hooks.
    """

    def __init__(self, config, target):
        super().__init__(config)
        self.adapter = TargetAgentAdapter(target)
        # Compatibility integrations can customize storage/guard instances at load.
        if self.adapter.legacy:
            self.session_store = target.session_store
            self.pre_llm_guardrails = target.pre_llm_guardrails
            self.post_llm_guardrails = target.post_llm_guardrails
            self.guardrail_middleware = target.guardrail_middleware
            self.guardrails_enabled = target.guardrails_enabled
        self._buffers_stream_output()  # Validate delivery policy before invocation.

    def _async_request_agent(self):
        turn = super()._async_request_agent()
        turn.adapter = self.adapter.for_request()
        return turn

    def _session_outputs(self, thread_id, user_id=""):
        output = super()._session_outputs(thread_id, user_id)
        active = mlflow.get_current_active_span()
        output["trace_id"] = active.trace_id if active else mlflow.get_last_active_trace_id()
        output["tags"] = {**self.default_tags,
                          **getattr(self, "_request_context", {}).get("trace_tags", {}),
                          "agentops.thread_id": str(thread_id)}
        return output

    def _supports_streaming(self):
        return self.adapter.supports_streaming

    def _buffers_stream_output(self):
        mode = self.config.get('middleware', {}).get('streaming', {}).get('output_mode', 'buffered')
        if mode not in {'buffered', 'passthrough'}:
            raise ValueError('middleware.streaming.output_mode must be buffered or passthrough')
        return mode == 'buffered' or super()._buffers_stream_output()

    async def _aprocess_request(self, request):
        return await self.adapter.invoke(request, self._request_context, self._request_resources)

    async def _aprocess_request_stream(self, request):
        async with aclosing(self.adapter.stream(request, self._request_context, self._request_resources)) as stream:
            async for event in stream:
                yield event
