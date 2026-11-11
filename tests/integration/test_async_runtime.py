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


def test_base_concurrent_turns_isolate_request_context(monkeypatch):
    barrier = threading.Barrier(2)

    class Echo(AgentOPSBase):
        def _process_request(self, request):
            self._request_context['marker'] = request.custom_inputs['marker']
            barrier.wait(timeout=2)
            return ResponsesAgentResponse(output=[self.create_text_output_item(
                text=self._request_context['marker'], id='answer')])

    monkeypatch.setattr('framework.agent_base.SessionStore', Mock())
    agent = Echo({'guardrails': {'enabled': False}})
    agent.session_store.enabled = False
    agent.session_store.memory_enabled = False
    agent.session_store._store_error = None
    agent.session_store._uc_last_error = None

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
        async def _aprocess_request(self, request):
            order.append('agent')
            await asyncio.sleep(0)
            return ResponsesAgentResponse(output=[self.create_text_output_item(text='answer', id='answer')])

        def _process_request(self, request):
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
        def _load_session_history(self, thread_id, messages):
            assert threading.get_ident() != owner_thread
            order.append('history')
            self._augmented_messages = messages

        def _recall_long_term_memory(self, user_id, query):
            assert threading.get_ident() != owner_thread
            order.append('memory')

        async def _aprocess_request(self, request):
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
