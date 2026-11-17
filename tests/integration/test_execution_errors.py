"""Failure telemetry must never hide the primary failure or replay the target."""
import asyncio
from unittest.mock import patch
import pytest
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from framework.inference.agent_middleware import AgentOPSMiddleware
from framework.inference.request_context import RequestContext


def request():
    return ResponsesAgentRequest(input=[{'role': 'user', 'content': 'hello'}])


def middleware(config=None):
    return AgentOPSMiddleware(config or {'guardrails': {'enabled': False}})


@pytest.mark.parametrize('kind', ['failure', 'timeout', 'cancel', 'success'])
def test_terminal_outcome_and_no_retries(kind):
    async def scenario():
        context = RequestContext()
        config = {'guardrails': {'enabled': False}, 'middleware': {'timeout_seconds': .02}}
        model = middleware(config)
        calls = []
        original = RuntimeError('private provider content')
        async def invoke(req, ctx):
            calls.append(1)
            if kind == 'failure':
                raise original
            if kind == 'timeout':
                await asyncio.sleep(1)
            if kind == 'cancel':
                raise asyncio.CancelledError()
            return ResponsesAgentResponse(output=[ResponsesAgent.create_text_output_item(text='ok', id='a')])
        with patch('mlflow.update_current_trace') as update:
            if kind == 'failure':
                with pytest.raises(RuntimeError) as caught:
                    await model.execute(request(), invoke=invoke, context=context)
                assert caught.value is original
            elif kind == 'timeout':
                with pytest.raises(TimeoutError):
                    await model.execute(request(), invoke=invoke, context=context)
            elif kind == 'cancel':
                with pytest.raises(asyncio.CancelledError):
                    await model.execute(request(), invoke=invoke, context=context)
            else:
                await model.execute(request(), invoke=invoke, context=context)
            assert 'private provider content' not in str(update.call_args_list)
        assert len(calls) == 1
        assert context.state['execution_finalized']
        assert context.state['execution_outcome'] == {'failure': 'failed', 'timeout': 'failed', 'cancel': 'cancelled', 'success': 'success'}[kind]
        assert context.resources is None
    asyncio.run(scenario())


def test_diagnostics_failure_preserves_primary_exception():
    async def scenario():
        model = middleware()
        async def broken_hook(*args):
            raise ValueError('diagnostic failure')
        model.process_error = broken_hook
        invoked = []
        async def invoke(req, context):
            invoked.append(True)
            raise RuntimeError('original')
        def telemetry(**kwargs):
            if invoked:
                raise ValueError('telemetry offline')
        with patch('mlflow.update_current_trace', side_effect=telemetry):
            with pytest.raises(RuntimeError, match='original'):
                await model.execute(request(), invoke=invoke)
    asyncio.run(scenario())


def test_buffered_stream_failure_releases_no_answer():
    from mlflow.types.responses import ResponsesAgentStreamEvent
    async def scenario():
        context = RequestContext()
        model = middleware()
        async def invoke(req, ctx):
            raise AssertionError('no fallback prediction')
        async def stream(req, ctx):
            yield ResponsesAgentStreamEvent(type='response.output_text.delta', item_id='a', delta='unchecked')
            raise RuntimeError('stream broke')
        delivered = []
        with pytest.raises(RuntimeError, match='stream broke'):
            async for item in model.execute_stream(request(), invoke=invoke, stream=stream, context=context):
                delivered.append(item)
        assert delivered == []
        assert context.state['execution_outcome'] == 'failed'
        assert context.state['execution_stage'] == 'target'
        assert context.state['execution_finalized']
    asyncio.run(scenario())


@pytest.mark.parametrize('stage', ['input_guardrails', 'output_guardrails', 'persistence'])
def test_non_target_failures_are_classified_at_their_stage(stage):
    from types import SimpleNamespace
    async def scenario():
        model = middleware()
        context = RequestContext()
        original = RuntimeError('controlled ' + stage)
        calls = []
        async def fail(*args, **kwargs):
            raise original
        async def identity(value):
            return value
        async def allowed(*args, **kwargs):
            return {'blocked': False}
        async def invoke(req, ctx):
            calls.append(True)
            return ResponsesAgentResponse(output=[ResponsesAgent.create_text_output_item(text='ok', id='a')])
        if stage == 'persistence':
            model.session_store.enabled = True
            model.session_store.asave_full_session = fail
        else:
            model.guardrails_enabled = True
            model.pre_llm_guardrails = SimpleNamespace(acheck=allowed)
            model.post_llm_guardrails = SimpleNamespace(acheck=allowed)
            model.guardrail_middleware = SimpleNamespace(
                process_request=fail if stage == 'input_guardrails' else identity,
                process_response=fail if stage == 'output_guardrails' else identity)
        # Client history avoids a database read; only the intended write is tested.
        req = ResponsesAgentRequest(input=[{'role': 'user', 'content': 'hello'},
            {'role': 'user', 'content': 'follow-up'}])
        with patch('mlflow.update_current_trace'):
            with pytest.raises(RuntimeError) as caught:
                await model.execute(req, invoke=invoke, context=context)
        assert caught.value is original
        assert len(calls) == (0 if stage == 'input_guardrails' else 1)
        assert context.state['execution_stage'] == stage
        assert context.state['execution_outcome'] == 'failed'
        assert context.state['execution_finalized']
    asyncio.run(scenario())


def test_failure_exports_available_usage_once_without_replaying_inference():
    async def scenario():
        model = middleware()
        context = RequestContext()
        async def invoke(req, ctx):
            ctx.state['token_usage_history'] = [{'prompt_tokens': 2, 'completion_tokens': 1,
                'total_tokens': 3, 'requests': 1, 'cost_usd': .01}]
            raise RuntimeError('failure after a metered operation')
        from framework.agent_base import emit_standard_trace_metrics
        with patch('framework.agent_base.emit_standard_trace_metrics', wraps=emit_standard_trace_metrics) as emit:
            with patch('mlflow.update_current_trace'):
                with pytest.raises(RuntimeError):
                    await model.execute(request(), invoke=invoke, context=context)
            assert emit.call_count == 1
        assert context.state['trace_tags']['agentops.tokens.total'] == '3'
        assert context.state['trace_tags']['agentops.llm_requests'] == '1'
    asyncio.run(scenario())
