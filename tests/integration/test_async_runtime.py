"""Async runtime contracts tested with real MLflow and no workspace calls."""
import asyncio
from contextlib import aclosing
from contextvars import ContextVar
import threading
import time
from unittest.mock import Mock

import pytest
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from framework.agent_base import AgentOPSBase
from framework.inference.async_runtime import AsyncResponsesAgent, iterate_blocking
from framework.observability import usage_tracking


@pytest.fixture(autouse=True)
def local_environment(monkeypatch):
    monkeypatch.setenv('AGENTOPS_ENVIRONMENT', 'test')


def run(coro):
    return asyncio.run(coro)


def test_event_loop_remains_responsive_and_context_is_propagated():
    marker = ContextVar('marker')

    class BlockingAgent(AsyncResponsesAgent):
        def predict(self, request):
            time.sleep(0.05)
            return marker.get()

    async def scenario():
        marker.set('request-trace')
        task = asyncio.create_task(BlockingAgent().apredict(None))
        await asyncio.sleep(0.01)
        assert not task.done()
        assert await task == 'request-trace'

    run(scenario())


def test_stream_order_errors_and_trace_context_stay_on_one_thread():
    marker = ContextVar('stream-marker', default=None)
    threads = []

    def source():
        token = marker.set('trace')
        try:
            for item in range(3):
                threads.append(threading.get_ident())
                yield item
            raise ValueError('upstream failed')
        finally:
            threads.append(threading.get_ident())
            marker.reset(token)

    async def scenario():
        items = []
        with pytest.raises(ValueError, match='upstream failed'):
            async for item in iterate_blocking(source):
                items.append(item)
        assert items == [0, 1, 2]
        assert len(set(threads)) == 1
        assert marker.get() is None

    run(scenario())


def test_stream_close_releases_bounded_worker():
    closed = threading.Event()
    produced = []

    def source():
        try:
            for item in range(1000):
                produced.append(item)
                yield item
        finally:
            closed.set()

    async def scenario():
        async with aclosing(iterate_blocking(source)) as stream:
            assert await anext(stream) == 0
        for _ in range(100):
            if closed.is_set():
                break
            await asyncio.sleep(0.01)
        assert closed.is_set()
        assert len(produced) <= 3

    run(scenario())


@pytest.mark.parametrize("wrapped", [False, True])
def test_base_concurrent_turns_isolate_request_context(monkeypatch, wrapped):
    barrier = threading.Barrier(2)

    class Echo(AgentOPSBase):
        def _process_request(self, request, context):
            context.state['marker'] = request.custom_inputs['marker']
            barrier.wait(timeout=2)
            return ResponsesAgentResponse(output=[self.create_text_output_item(
                text=context.state['marker'], id='answer')])

    monkeypatch.setattr('framework.agent_base.SessionStore', Mock())
    agent = Echo({'guardrails': {'enabled': False}})
    agent.session_store.enabled = False
    agent.session_store.memory_enabled = False
    agent.session_store._store_error = None
    agent.session_store._uc_last_error = None
    if wrapped:
        import importlib
        from unittest.mock import patch
        with patch("mlflow.models.set_model"):
            module = importlib.import_module("framework.inference.cost_tracking_model")
        wrapper = module.CostTrackingResponsesModel()
        from framework.inference.agent_middleware import AgentOPSMiddleware
        wrapper._inner = agent
        wrapper._middleware = AgentOPSMiddleware(agent.config, agent)
        agent = wrapper

    async def scenario():
        requests = [ResponsesAgentRequest(input=[{'role': 'user', 'content': 'hello'}],
                                         custom_inputs={'marker': label}) for label in ['one', 'two']]
        responses = await asyncio.gather(*(agent.apredict(req) for req in requests))
        assert [response.output[0].content[0]['text'] for response in responses] == ['one', 'two']

    run(scenario())


def test_async_runnable_usage_and_errors_not_retried(monkeypatch):
    monkeypatch.setattr(usage_tracking, 'TokenTracker', None)
    monkeypatch.setattr(usage_tracking, 'get_usage_metadata_callback', None)

    class Runnable:
        calls = 0

        async def ainvoke(self, payload, config):
            self.calls += 1
            if payload == 'fail':
                raise TypeError('model failed')
            return {'usage': {'input_tokens': 10, 'output_tokens': 2}}

    async def scenario():
        runnable = Runnable()
        context = {}
        await usage_tracking.ainvoke_with_usage_tracking(runnable, 'ok', context)
        assert context['token_usage_history'][0]['total_tokens'] == 12
        with pytest.raises(TypeError):
            await usage_tracking.ainvoke_with_usage_tracking(runnable, 'fail', context)
        assert runnable.calls == 2

    run(scenario())


def test_async_endpoint_uses_native_transport_and_does_not_retry(monkeypatch):
    import httpx
    from types import SimpleNamespace
    from framework.inference.client import ainvoke_endpoint
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"output": []})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client_type(
        transport=httpx.MockTransport(handler), **kwargs))
    workspace = SimpleNamespace(config=SimpleNamespace(
        host="https://workspace.example", authenticate=lambda: {"Authorization": "test"}))
    result = run(ainvoke_endpoint("agent", {"input": []}, workspace=workspace))
    assert result == {"output": []}
    assert len(requests) == 1
    assert str(requests[0].url) == "https://workspace.example/serving-endpoints/agent/invocations"
    with pytest.raises(ValueError):
        run(ainvoke_endpoint("invalid/name", {}, workspace=workspace))
    assert len(requests) == 1


def test_native_async_guardrails_and_agent_hook_are_awaited(monkeypatch):
    order = []

    class Guard:
        async def acheck(self, *args, **kwargs):
            order.append('guard')
            await asyncio.sleep(0)
            return {'blocked': False}

        def check(self, *args, **kwargs):
            raise AssertionError('Native async guardrail must be awaited')

    class Native(AgentOPSBase):
        async def _aprocess_request(self, request, context):
            order.append('agent')
            await asyncio.sleep(0)
            return ResponsesAgentResponse(output=[self.create_text_output_item(text='answer', id='answer')])

        def _process_request(self, request, context):
            raise AssertionError('Native async agent must be awaited')

    agent = Native({'session_history': {'enabled': False}, 'guardrails': {'enabled': True}})
    agent.pre_llm_guardrails = Guard()
    agent.post_llm_guardrails = Guard()
    request = ResponsesAgentRequest(input=[{'role': 'user', 'content': 'question'}])
    response = run(agent.apredict(request))
    assert response.output[0].content[0]['text'] == 'answer'
    assert order == ['guard', 'agent', 'guard']
    order.clear()
    assert agent.predict(request).output[0].content[0]['text'] == 'answer'
    assert order == ['guard', 'agent', 'guard']


def test_history_recall_and_persistence_run_off_loop_in_order():
    from types import SimpleNamespace
    owner_thread = threading.get_ident()
    order = []

    class Agent(AgentOPSBase):
        def _load_session_history(self, thread_id, messages, *, context):
            assert threading.get_ident() != owner_thread
            order.append('history')
            context["messages"] = messages

        def _recall_long_term_memory(self, user_id, query, *, context):
            assert threading.get_ident() != owner_thread
            order.append('memory')

        async def _aprocess_request(self, request, context):
            order.append('agent')
            return ResponsesAgentResponse(output=[self.create_text_output_item(text='answer', id='answer')])

    def save(**kwargs):
        assert threading.get_ident() != owner_thread
        order.append('save')

    agent = Agent({'guardrails': {'enabled': False}})
    agent.session_store = SimpleNamespace(enabled=True, memory_enabled=True,
        _store_error=None, _uc_last_error=None, save_full_session=save)
    request = ResponsesAgentRequest(input=[{'role': 'user', 'content': 'question'}],
                                   custom_inputs={'user_id': 'owner', 'thread_id': 'session'})
    run(agent.apredict(request))
    assert order == ['history', 'memory', 'agent', 'save']


def middleware_agent(agent_type, config=None):
    from types import SimpleNamespace
    agent = agent_type({'guardrails': {'enabled': False}, **(config or {})})
    agent.session_store = SimpleNamespace(enabled=False, memory_enabled=False,
        _store_error=None, _uc_last_error=None)
    return agent


def test_shared_stream_lifecycle_awaits_guards_and_native_storage():
    from mlflow.types.responses import ResponsesAgentStreamEvent
    from types import SimpleNamespace
    order = []

    class Guard:
        async def acheck(self, *args, **kwargs):
            order.append('guard')
            return {'blocked': False}

    class Native(AgentOPSBase):
        async def _aprocess_request_stream(self, request, context):
            order.append('agent')
            assert context.state['user_memories'] == ['pattern']
            yield ResponsesAgentStreamEvent(type='response.output_item.done',
                item=self.create_text_output_item(text='answer', id='answer'))

    async def history(*args, **kwargs):
        order.append('history')
        return [{'role': 'user', 'content': 'earlier'}]

    async def memory(*args, **kwargs):
        order.append('memory')
        return ['pattern']

    async def save(**kwargs):
        order.append('save')
        assert len(kwargs['messages']) == 2
        assert kwargs['response_text'] == 'answer'

    agent = middleware_agent(Native)
    agent.guardrails_enabled = True
    agent.pre_llm_guardrails = agent.post_llm_guardrails = Guard()
    agent.session_store = SimpleNamespace(enabled=True, memory_enabled=True,
        _store=None, _store_error=None, _uc_last_error=None,
        aget_history=history, arecall_user_memories=memory, asave_full_session=save)
    request = ResponsesAgentRequest(input=[{'role': 'user', 'content': 'question'}],
                                   custom_inputs={'thread_id': 'thread', 'user_id': 'owner'})

    async def scenario():
        events = [event async for event in agent.apredict_stream(request)]
        assert len(events) == 1
        assert events[0].custom_outputs['thread_id'] == 'thread'

    run(scenario())
    assert order == ['history', 'guard', 'memory', 'agent', 'guard', 'save']


@pytest.mark.parametrize('streaming', [False, True])
def test_request_deadline_closes_resources_in_owner_task(streaming):
    from contextlib import asynccontextmanager
    calls = []

    @asynccontextmanager
    async def resource():
        task = asyncio.current_task()
        calls.append('open')
        try:
            yield
        finally:
            assert asyncio.current_task() is task
            calls.append('close')

    class Slow(AgentOPSBase):
        async def _aprocess_request(self, request, context):
            await context.resources.enter_async_context(resource())
            await asyncio.sleep(10)

        async def _aprocess_request_stream(self, request, context):
            await self._aprocess_request(request, context)
            yield  # Establish an async generator without producing an event.

    agent = middleware_agent(Slow, {'middleware': {'timeout_seconds': 0.02}})
    request = ResponsesAgentRequest(input=[{'role': 'user', 'content': 'question'}])

    async def scenario():
        with pytest.raises(TimeoutError):
            if streaming:
                async for _ in agent.apredict_stream(request):
                    pass
            else:
                await agent.apredict(request)

    run(scenario())
    assert calls == ['open', 'close']


@pytest.mark.parametrize('synchronous', [False, True])
def test_stream_close_cleans_native_resources_without_saving_partial_answer(synchronous):
    from contextlib import asynccontextmanager
    from mlflow.types.responses import ResponsesAgentStreamEvent
    closed = threading.Event()
    tasks = []

    @asynccontextmanager
    async def resource():
        tasks.append(asyncio.current_task())
        try:
            yield
        finally:
            assert asyncio.current_task() is tasks[0]
            closed.set()

    class Native(AgentOPSBase):
        async def _aprocess_request_stream(self, request, context):
            await context.resources.enter_async_context(resource())
            yield ResponsesAgentStreamEvent(type='response.output_text.delta',
                item_id='answer', delta='partial')
            await asyncio.sleep(10)

    agent = middleware_agent(Native)
    agent.session_store.enabled = True
    agent.session_store.save_full_session = Mock()
    # Client history avoids a fetch; interrupted output must still not be saved.
    request = ResponsesAgentRequest(input=[
        {'role': 'assistant', 'content': 'previous'},
        {'role': 'user', 'content': 'question'}],
        custom_inputs={'thread_id': 'thread'})

    async def scenario():
        async with aclosing(agent.apredict_stream(request)) as stream:
            assert (await anext(stream)).delta == 'partial'
        assert closed.is_set()

    if synchronous:
        stream = agent.predict_stream(request)
        assert next(stream).delta == 'partial'
        stream.close()
        assert closed.wait(1)
    else:
        run(scenario())
    agent.session_store.save_full_session.assert_not_called()


def test_sync_stream_keeps_resource_task_and_trace_context_until_completion():
    from contextlib import asynccontextmanager
    from mlflow.types.responses import ResponsesAgentStreamEvent
    marker = ContextVar('middleware-trace', default=None)
    tasks = []

    @asynccontextmanager
    async def resource():
        tasks.append(asyncio.current_task())
        try:
            yield
        finally:
            assert asyncio.current_task() is tasks[0]
            assert marker.get() == 'trace'
            tasks.append('closed')

    class Native(AgentOPSBase):
        async def _aprocess_request_stream(self, request, context):
            await context.resources.enter_async_context(resource())
            for value in ['one', 'two']:
                assert asyncio.current_task() is tasks[0]
                assert marker.get() == 'trace'
                yield ResponsesAgentStreamEvent(type='response.output_text.delta',
                    item_id='answer', delta=value)

    agent = middleware_agent(Native)
    marker.set('trace')
    events = list(agent.predict_stream(ResponsesAgentRequest(
        input=[{'role': 'user', 'content': 'question'}])))
    assert [event.delta for event in events] == ['one', 'two']
    assert tasks[-1] == 'closed'


def test_native_streams_keep_concurrent_request_context_isolated():
    from mlflow.types.responses import ResponsesAgentStreamEvent

    class Native(AgentOPSBase):
        async def _aprocess_request_stream(self, request, context):
            context.state['marker'] = request.custom_inputs['marker']
            await asyncio.sleep(0.01)
            yield ResponsesAgentStreamEvent(type='response.output_item.done',
                item=self.create_text_output_item(text=context.state['marker'], id='answer'))

    agent = middleware_agent(Native)

    async def consume(label):
        request = ResponsesAgentRequest(input=[{'role': 'user', 'content': 'question'}],
            custom_inputs={'marker': label, 'thread_id': label})
        events = [event async for event in agent.apredict_stream(request)]
        assert events[0].custom_outputs['thread_id'] == label
        return events[0].item['content'][0]['text']

    async def scenario():
        assert await asyncio.gather(consume('one'), consume('two')) == ['one', 'two']

    run(scenario())


def test_sync_stream_propagates_failure_after_resource_cleanup():
    from contextlib import asynccontextmanager
    closed = threading.Event()

    @asynccontextmanager
    async def resource():
        try:
            yield
        finally:
            closed.set()

    class Failing(AgentOPSBase):
        async def _aprocess_request_stream(self, request, context):
            await context.resources.enter_async_context(resource())
            raise ValueError('tool failed')
            yield

    agent = middleware_agent(Failing)
    with pytest.raises(ValueError, match='tool failed'):
        list(agent.predict_stream(ResponsesAgentRequest(
            input=[{'role': 'user', 'content': 'question'}])))
    assert closed.is_set()


@pytest.mark.parametrize('seconds', [0, -1, float('inf'), float('nan')])
def test_invalid_middleware_deadline_rejected_before_invocation(seconds):
    class Unused(AgentOPSBase):
        async def _aprocess_request(self, request, context):
            raise AssertionError('Invalid deadline must fail before invocation')

    agent = middleware_agent(Unused, {'middleware': {'timeout_seconds': seconds}})
    with pytest.raises(ValueError, match='finite and positive'):
        run(agent.apredict(ResponsesAgentRequest(input=[{'role': 'user', 'content': 'question'}])))
