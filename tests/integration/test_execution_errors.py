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
