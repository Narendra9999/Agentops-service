"""Async team execution, usage isolation and checked streaming contracts."""
import asyncio
from contextlib import aclosing
from types import SimpleNamespace as NS
import sys
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from mlflow.pyfunc import ResponsesAgent

from fe_multi_agent.graph import Execution, build_supervisor_graph
from fe_multi_agent.memory_graph import build_memory_graph
from fe_multi_agent.supervisor_agent import astream_prediction
from fe_multi_agent.usage import capture_request_usage, current_meter


def test_graph_awaits_native_router_and_worker_once():
    calls = []

    class Router:
        def with_structured_output(self, schema):
            self.schema = schema
            return self

        async def ainvoke(self, messages):
            calls.append('supervisor')
            if calls.count('supervisor') == 1:
                return self.schema(next_agent='cost_estimation', task='Estimate costs')
            return self.schema(next_agent='finish', answer='Need workload quantities')

        def invoke(self, *args):
            raise AssertionError('Sync routing must not be used')

    class Worker:
        async def ainvoke(self, state):
            calls.append('worker')
            return {'messages': [AIMessage(content='Need workload quantities')]}

        def invoke(self, *args):
            raise AssertionError('Sync worker must not be used')

    async def scenario():
        graph = build_supervisor_graph(Router(), {'cost_estimation': Worker(), 'platform_docs': Worker()},
                                      'Route workers', Execution('E', 'T', 'A'))
        result = await graph.ainvoke({'messages': [HumanMessage(content='Estimate costs')]})
        assert result['messages'][-1].content == 'Need workload quantities'

    asyncio.run(scenario())
    assert calls == ['supervisor', 'worker', 'supervisor']


def test_memory_graph_awaits_model_and_offloads_blocking_memory():
    import threading
    owner = threading.get_ident()
    calls = []

    class Memory:
        def files(self, query):
            assert threading.get_ident() != owner
            calls.append('memory')
            return []

    class Model:
        def bind_tools(self, *args, **kwargs):
            return self

        async def ainvoke(self, messages):
            name, args = ('list_memory_files', {'query': ''}) if not calls else ('finish_response', {'answer': 'Done'})
            return AIMessage(content='', tool_calls=[{'id': name, 'name': name, 'args': args}])

        def invoke(self, *args):
            raise AssertionError('Model must be awaited')

    async def scenario():
        graph = build_memory_graph(Model(), {}, 'Use memory', Memory(), {})
        result = await graph.ainvoke({'messages': [HumanMessage(content='Review memory')]})
        assert result['messages'][-1].content == 'Done'

    asyncio.run(scenario())
    assert calls == ['memory']


def test_async_usage_capture_is_request_scoped_and_does_not_replay(monkeypatch):
    monkeypatch.setattr('fe_multi_agent.usage.install_native_cost_processor', lambda: None)
    meters = []
    calls = []

    class Agent:
        settings = NS(usage_rates_per_million={})

        @capture_request_usage
        async def apredict(self, label):
            calls.append(label)
            meter = current_meter()
            meters.append(meter)
            await asyncio.sleep(0)
            assert current_meter() is meter
            if label == 'fail':
                raise TypeError('provider failed')
            return ResponsesAgentResponse(output=[])

    async def scenario():
        agent = Agent()
        responses = await asyncio.gather(agent.apredict('one'), agent.apredict('two'))
        assert all(response.custom_outputs['usage_metrics']['llm_requests'] == 0 for response in responses)
        assert meters[0] is not meters[1]
        with pytest.raises(TypeError, match='provider failed'):
            await agent.apredict('fail')
        assert current_meter() is None

    asyncio.run(scenario())
    assert calls == ['one', 'two', 'fail']


def test_activity_stream_waits_for_checked_response_and_cancels_execution():
    from fe_multi_agent.progress import emit

    async def scenario():
        checked = asyncio.Event()
        closed = asyncio.Event()

        async def predict(request):
            emit('agent_started', 'cost_estimation', 'Calling Genie')
            try:
                await checked.wait()
                return ResponsesAgentResponse(output=[ResponsesAgent.create_text_output_item(text='checked', id='answer')])
            finally:
                closed.set()

        request = NS(custom_inputs={'stream_activity': True})
        async with aclosing(astream_prediction(predict, request)) as stream:
            first = await anext(stream)
            assert first.type == 'response.in_progress'
            assert not checked.is_set()
        assert closed.is_set()

    asyncio.run(scenario())


def test_pinned_model_adapter_uses_native_async_gateway_transport(monkeypatch):
    import httpx
    import openai
    from fe_multi_agent.services import chat_model
    calls = []

    class Base:
        def __init__(self, **kwargs):
            self.model = kwargs['model']
            self.client = NS(base_url='https://workspace.example/serving-endpoints/')
            self.timeout = None
            self.use_responses_api = False

        def _prepare_inputs(self, messages, stop, **kwargs):
            return {'model': self.model, 'messages': messages, **kwargs}

        def _convert_response_to_chat_result(self, response):
            return response

    monkeypatch.setitem(sys.modules, 'databricks_langchain', NS(ChatDatabricks=Base))
    client_type = openai.AsyncOpenAI

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={'id': 'chat-test', 'object': 'chat.completion', 'created': 1,
            'model': 'system.ai.claude-sonnet-4-6', 'choices': [{'index': 0,
            'message': {'role': 'assistant', 'content': 'Done'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}})

    monkeypatch.setattr(openai, 'AsyncOpenAI', lambda **kwargs: client_type(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kwargs))
    workspace = NS(config=NS(host='https://workspace.example',
                            authenticate=lambda: {'Authorization': 'Bearer test-only'}))
    model = chat_model(NS(model_transport='ai_gateway'), workspace, 'system.ai.claude-sonnet-4-6')
    response = asyncio.run(model._agenerate([{'role': 'user', 'content': 'question'}]))
    assert response.usage.total_tokens == 12
    assert len(calls) == 1
    assert calls[0].url.path == '/ai-gateway/mlflow/v1/chat/completions'
    assert calls[0].headers['Authorization'] == 'Bearer test-only'
