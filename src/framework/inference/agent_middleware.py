"""AgentOPS owns inference; adapters invoke target logic without a second lifecycle."""
from contextlib import aclosing
from inspect import isasyncgenfunction
import uuid
import mlflow

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from framework.agent_base import AgentOPSBase
from framework.inference.request_context import RequestContext
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
    """Provide invocation callbacks, binding explicit context to context-aware agents."""

    def __init__(self, target, config=None):
        self.target = target
        self.contextual = isinstance(target, AgentOPSBase)
        self.responses = isinstance(target, ResponsesAgent)
        protocol = (config or {}).get('inference', {}).get('target_protocol', 'responses')
        if protocol not in {'responses', 'chat_completions'}:
            raise ValueError('inference.target_protocol must be responses or chat_completions')
        self.chat = None
        if protocol == 'chat_completions':
            if self.contextual:
                raise ValueError('Chat Completions targets expose public callbacks, not AgentOPSBase hooks')
            from framework.inference.chat_completions import ChatCompletionsAdapter
            self.chat = ChatCompletionsAdapter(target, config or {})

    @property
    def supports_streaming(self):
        if self.chat is not None:
            return self.chat.supports_streaming
        if self.contextual:
            return self.target._supports_streaming()
        if not self.responses:
            return False
        native = getattr(type(self.target), 'apredict_stream', None)
        sync = getattr(type(self.target), 'predict_stream', None)
        return ((native is not None and native is not AsyncResponsesAgent.apredict_stream)
                or (sync is not None and sync is not ResponsesAgent.predict_stream))

    def _bind(self, request, context):
        if self.contextual:
            return self.target, request
        custom = dict(request.custom_inputs or {})
        custom['thread_id'] = context.state['thread_id']
        custom.pop('agentops_context', None)
        if context.state.get('user_id'):
            custom.setdefault('user_id', context.state['user_id'])
        if context.state.get('user_memories'):
            custom['agentops_context'] = {'user_memories': context.state['user_memories']}
        return self.target, request.model_copy(update={'custom_inputs': custom})

    async def invoke(self, request, context):
        target, request = self._bind(request, context)
        if self.chat is not None:
            return await self.chat.invoke(request, context)
        if self.contextual:
            result = await target._aprocess_request(request, context)
        elif self.responses:
            result = await call_async_method(target, 'apredict', 'predict', request)
        else:
            result = await call_async_method(target, 'apredict', 'predict',
                context=None, model_input=request.model_dump(exclude_none=True))
        return wrap_target_result(result)

    async def stream(self, request, context):
        target, request = self._bind(request, context)
        if self.chat is not None:
            stream = self.chat.stream(request, context)
        elif self.contextual:
            stream = target._aprocess_request_stream(request, context)
        else:
            native = getattr(target, 'apredict_stream', None)
            stream = (native(request) if isasyncgenfunction(native)
                      else iterate_blocking(target.predict_stream, request))
        async with aclosing(stream):
            async for event in stream:
                yield event


class AgentOPSMiddleware(AgentOPSBase):
    """Shared lifecycle with explicit request context and invocation callbacks.

    No request state is assigned to this instance and no agent is cloned.
    AgentOPSBase supplies the common callback lifecycle and configuration.
    """

    def __init__(self, config, target=None):
        super().__init__(config)
        self.adapter = TargetAgentAdapter(target, config) if target is not None else None
        if self.adapter is not None and self.adapter.contextual:
            self.session_store = target.session_store
            self.pre_llm_guardrails = target.pre_llm_guardrails
            self.post_llm_guardrails = target.post_llm_guardrails
            self.guardrail_middleware = target.guardrail_middleware
            self.guardrails_enabled = target.guardrails_enabled
        self._buffers_stream_output()

    def _session_outputs(self, thread_id, user_id="", *, context=None):
        output = super()._session_outputs(thread_id, user_id, context=context)
        active = mlflow.get_current_active_span()
        output["trace_id"] = active.trace_id if active else mlflow.get_last_active_trace_id()
        output["tags"] = {**self.default_tags, **(context or {}).get("trace_tags", {}),
                          "agentops.thread_id": str(thread_id)}
        return output

    def _buffers_stream_output(self):
        mode = self.config.get('middleware', {}).get('streaming', {}).get('output_mode', 'buffered')
        if mode not in {'buffered', 'passthrough'}:
            raise ValueError('middleware.streaming.output_mode must be buffered or passthrough')
        return mode == 'buffered' or super()._buffers_stream_output()

    async def apredict(self, request):
        if self.adapter is None:
            raise ValueError('Supply an invoke callback to execute() or configure a target')
        return await self.execute(request, invoke=self.adapter.invoke)

    async def apredict_stream(self, request):
        if self.adapter is None:
            raise ValueError('Supply callbacks to execute_stream() or configure a target')
        async with aclosing(self.execute_stream(request, invoke=self.adapter.invoke,
            stream=self.adapter.stream if self.adapter.supports_streaming else None)) as events:
            async for event in events:
                yield event

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        from framework.inference.async_runtime import run_sync
        return run_sync(self.apredict(request))

    def predict_stream(self, request):
        from framework.inference.async_runtime import iterate_async_sync
        yield from iterate_async_sync(lambda: self.apredict_stream(request))
