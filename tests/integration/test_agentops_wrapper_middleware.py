"""AgentOPS must be the caller, independently of target orchestration/framework."""
import asyncio
from contextlib import aclosing, asynccontextmanager
from copy import deepcopy
import importlib
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from mlflow.pyfunc import PythonModel, ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse, ResponsesAgentStreamEvent

from framework.agent_base import AgentOPSBase
from framework.inference import extract_request_text, extract_response_text
from framework.inference.async_runtime import AsyncResponsesAgent


@pytest.fixture(autouse=True)
def test_environment(monkeypatch):
    monkeypatch.setenv('AGENTOPS_ENVIRONMENT', 'test')


class NativeTarget(AsyncResponsesAgent):
    def predict(self, request):
        raise AssertionError('The middleware must await the native async capability')


def response(text='Target answer', **custom):
    return ResponsesAgentResponse(output=[ResponsesAgent.create_text_output_item(text=text, id='answer')],
                                  custom_outputs=custom)


def request(text='Explain Databricks usage', **custom):
    return ResponsesAgentRequest(input=[{'role': 'user', 'content': text}], custom_inputs=custom)


def wrapper(target, config=None):
    with patch('mlflow.models.set_model'):
        module = importlib.import_module('framework.inference.cost_tracking_model')
    config = config or {'guardrails': {'enabled': False}}
    with patch.object(module, '_resolve_inner_model', return_value=target):
        model = module.CostTrackingResponsesModel()
        model.load_context(NS(model_config=config))
    return model


def guards(model, order, blocked_at=None):
    class Guard:
        def __init__(self, stage):
            self.stage = stage

        async def acheck(self, *args, **kwargs):
            order.append(self.stage)
            return {'blocked': self.stage == blocked_at, 'message': 'Filtered by AgentOPS'}

    model._middleware.guardrails_enabled = True
    model._middleware.pre_llm_guardrails = Guard('pre')
    model._middleware.post_llm_guardrails = Guard('post')


@pytest.mark.parametrize('kind', ['sync_responses', 'async_responses', 'python_model'])
@pytest.mark.parametrize('streaming', [False, True])
def test_plain_targets_are_called_inside_one_shared_lifecycle(kind, streaming):
    order = []

    class Sync(ResponsesAgent):
        def predict(self, request):
            order.append('target')
            return response()

    class Native(NativeTarget):
        async def apredict(self, request):
            order.append('target')
            return response()

    class Plain(PythonModel):
        def predict(self, context, model_input):
            order.append('target')
            return 'Target answer'

    model = wrapper({'sync_responses': Sync, 'async_responses': Native, 'python_model': Plain}[kind]())
    guards(model, order)
    if streaming:
        result = list(model.predict_stream(request()))
        assert [event.type for event in result] == ['response.output_item.done']
        assert result[0].item['content'][0]['text'] == 'Target answer'
    else:
        assert extract_response_text(model.predict(request())) == 'Target answer'
    assert order == ['pre', 'target', 'post']


@pytest.mark.parametrize('streaming', [False, True])
def test_rejected_input_never_calls_an_independent_target(streaming):
    calls = []

    class Target(ResponsesAgent):
        def predict(self, request):
            calls.append('target')
            return response()

    model = wrapper(Target())
    guards(model, calls, blocked_at='pre')
    if streaming:
        events = list(model.predict_stream(request()))
        assert events[0].item['content'][0]['text'] == 'Filtered by AgentOPS'
    else:
        assert extract_response_text(model.predict(request())) == 'Filtered by AgentOPS'
    assert calls == ['pre']


@pytest.mark.parametrize('streaming', [False, True])
def test_context_hook_target_public_lifecycle_is_bypassed_without_double_guardrails(streaming):
    order = []

    class Contextual(AgentOPSBase):
        def __copy__(self):
            raise AssertionError('A context-aware target must not be cloned')

        def __deepcopy__(self, memo):
            raise AssertionError('A context-aware target must not be deep-copied')

        def predict(self, request):
            raise AssertionError('Wrapper must call raw target hook, not legacy middleware')

        def predict_stream(self, request):
            raise AssertionError('Wrapper must own the streaming lifecycle')
            yield  # MLflow requires predict_stream to be a generator.

        async def _aprocess_request(self, request, context):
            order.append('target')
            assert context.state['thread_id'] == 'thread'
            return response()

        async def _aprocess_request_stream(self, request, context):
            result = await self._aprocess_request(request, context)
            yield ResponsesAgentStreamEvent(type='response.output_item.done', item=result.output[0])

    model = wrapper(Contextual({'guardrails': {'enabled': False}}))
    guards(model, order)
    list(model.predict_stream(request(thread_id='thread'))) if streaming else model.predict(request(thread_id='thread'))
    assert order == ['pre', 'target', 'post']


@pytest.mark.parametrize('async_client', [False, True])
def test_wrapper_awaits_native_stream_and_releases_only_checked_output(async_client):
    order = []

    class Target(NativeTarget):
        async def apredict_stream(self, req):
            order.append('target')
            yield ResponsesAgentStreamEvent(type='response.output_text.delta', item_id='answer', delta='unsafe')
            yield ResponsesAgentStreamEvent(type='response.output_item.done', item=response('unsafe').output[0])

    model = wrapper(Target())
    guards(model, order, blocked_at='post')

    async def collect():
        return [event async for event in model.apredict_stream(request())]

    events = asyncio.run(collect()) if async_client else list(model.predict_stream(request()))
    assert order == ['pre', 'target', 'post']
    assert len(events) == 1 and events[0].type == 'response.output_item.done'
    assert events[0].item['content'][0]['text'] == 'Filtered by AgentOPS'


def test_history_memory_redaction_and_save_are_owned_by_wrapper():
    calls = []
    seen = []

    class Target(NativeTarget):
        async def apredict(self, req):
            calls.append('target')
            seen.append(req)
            return response('Contact bob@private.example for Databricks usage')

    config = {'guardrails': {'pre_llm': {'enabled_checks': ['pii'], 'pii': {'action': 'redact'}},
                             'post_llm': {'enabled_checks': ['pii_leakage'], 'pii_leakage': {'action': 'redact'}}}}
    model = wrapper(Target(), config)

    async def history(*args, **kwargs):
        calls.append('history')
        return [{'role': 'user', 'content': 'Previous Databricks question'}]

    async def memory(*args, **kwargs):
        calls.append('memory')
        return [{'key': 'pattern', 'content': 'Monthly batch'}]

    async def save(**kwargs):
        calls.append('save')
        assert 'private.example' not in kwargs['response_text']
        assert 'private.example' not in str(kwargs['messages'])

    model._middleware.session_store = NS(enabled=True, memory_enabled=True, _store=None,
        _store_error=None, _uc_last_error=None, aget_history=history,
        arecall_user_memories=memory, asave_full_session=save)
    req = request('Contact alice@private.example about Databricks', user_id='owner', thread_id='thread',
                  agentops_context={'user_memories': ['spoofed']})
    original = deepcopy(req)
    result = model.predict(req)
    assert calls == ['history', 'memory', 'target', 'save']
    assert len(seen[0].input) == 2
    assert '[EMAIL-REDACTED]' in seen[0].input[-1].content
    assert seen[0].custom_inputs['agentops_context']['user_memories'][0]['key'] == 'pattern'
    assert '[EMAIL-REDACTED]' in extract_response_text(result)
    assert req == original


@pytest.mark.parametrize('streaming', [False, True])
def test_structured_artifacts_are_checked_without_supervisor_specific_wrapper(streaming):
    class Target(ResponsesAgent):
        def predict(self, req):
            return response(genie_result={'tables': [{'rows': [['Guaranteed profit']]}]})

    model = wrapper(Target(), {'guardrails': {'pre_llm': {'enabled_checks': []},
                                             'post_llm': {'enabled_checks': ['compliance']}}})
    if streaming:
        events = list(model.predict_stream(request()))
        assert 'genie_result' not in events[0].custom_outputs
        assert 'rephrase' in events[0].item['content'][0]['text']
    else:
        result = model.predict(request())
        assert 'genie_result' not in result.custom_outputs
        assert 'rephrase' in extract_response_text(result)


def test_stream_metadata_usage_and_activity_survive_outer_middleware():
    order = []

    class Target(NativeTarget):
        async def apredict_stream(self, req):
            order.append('target')
            yield ResponsesAgentStreamEvent(type='response.in_progress',
                response={'id': 'status', 'object': 'response', 'status': 'in_progress', 'output': []},
                custom_outputs={'activity': {'kind': 'worker_started', 'agent': 'cost'}})
            yield ResponsesAgentStreamEvent(type='response.output_item.done', item=response().output[0],
                usage={'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12,
                       'input_tokens_details': {'cached_tokens': 0},
                       'output_tokens_details': {'reasoning_tokens': 0}},
                custom_outputs={'trace_id': 'tr-test', 'genie_result': {'tables': []}})

    model = wrapper(Target())
    guards(model, order)
    events = list(model.predict_stream(request(stream_activity=True, thread_id='thread')))
    assert events[0].type == 'response.in_progress'
    assert events[-1].type == 'response.output_item.done'
    assert events[-1].custom_outputs['trace_id'] == 'tr-test'
    assert events[-1].custom_outputs['thread_id'] == 'thread'
    assert events[-1].usage['total_tokens'] == 12
    assert order == ['pre', 'target', 'post']


def test_concurrent_wrapper_calls_do_not_share_explicit_request_context():
    class Contextual(AgentOPSBase):
        async def _aprocess_request(self, req, context):
            context.state['marker'] = req.custom_inputs['marker']
            await asyncio.sleep(0.01)
            return response(context.state['marker'])

    model = wrapper(Contextual({'guardrails': {'enabled': False}}))

    async def scenario():
        results = await asyncio.gather(*(model.apredict(request(marker=value)) for value in ['one', 'two']))
        assert [extract_response_text(result) for result in results] == ['one', 'two']
        assert results[0].custom_outputs['thread_id'] != results[1].custom_outputs['thread_id']

    asyncio.run(scenario())


def test_target_failure_is_not_retried_or_saved_as_success():
    calls = []

    class Target(NativeTarget):
        async def apredict(self, req):
            calls.append('target')
            raise TypeError('provider failed')

    model = wrapper(Target())
    guards(model, calls)
    with pytest.raises(TypeError, match='provider failed'):
        model.predict(request())
    assert calls == ['pre', 'target']


def test_cancelling_wrapper_stream_closes_target_without_saving_partial_output():
    closed = []

    class Target(NativeTarget):
        async def apredict_stream(self, req):
            try:
                yield ResponsesAgentStreamEvent(type='response.in_progress',
                    response={'id': 'status', 'object': 'response', 'status': 'in_progress', 'output': []})
                await asyncio.sleep(10)
            finally:
                closed.append(True)

    model = wrapper(Target())

    async def scenario():
        async with aclosing(model.apredict_stream(request(stream_activity=True))) as stream:
            assert (await anext(stream)).type == 'response.in_progress'
        assert closed == [True]

    asyncio.run(scenario())


def test_invalid_delivery_policy_fails_at_load_before_target_call():
    class Target(ResponsesAgent):
        def predict(self, req):
            raise AssertionError('Never invoke invalid configuration')

    with pytest.raises(ValueError, match='output_mode'):
        wrapper(Target(), {'middleware': {'streaming': {'output_mode': 'invalid'}}})


def test_packaged_responses_loader_uses_outer_middleware_for_both_modes(tmp_path):
    import mlflow
    calls = []

    class Target(ResponsesAgent):
        def predict(self, req):
            calls.append(extract_request_text(req))
            return response('Contact alice@private.example')

    config = {'guardrails': {'pre_llm': {'enabled_checks': ['input_length_min'], 'min_input_length': 5},
                             'post_llm': {'enabled_checks': ['pii_leakage'], 'pii_leakage': {'action': 'redact'}}}}
    model = wrapper(Target(), config)
    module = importlib.import_module('framework.inference.cost_tracking_model')
    with patch.object(module, '_resolve_inner_model', return_value=Target()):
        mlflow.pyfunc.save_model(path=str(tmp_path / 'model'), python_model=model,
                                 model_config=config, pip_requirements=[])
        loaded = mlflow.pyfunc.load_model(str(tmp_path / 'model'))
        result = loaded.predict(request().model_dump(exclude_none=True))
        assert '[EMAIL-REDACTED]' in extract_response_text(result)
        events = list(loaded.predict_stream(request().model_dump(exclude_none=True)))
        assert len(events) == 1
        assert '[EMAIL-REDACTED]' in events[0]['item']['content'][0]['text']
        blocked = loaded.predict(request('?').model_dump(exclude_none=True))
        assert 'too short' in extract_response_text(blocked)


@pytest.mark.parametrize('streaming', [False, True])
def test_actual_supervisor_entrypoint_is_a_target_of_shared_middleware(monkeypatch, streaming):
    import importlib.util
    from pathlib import Path
    import sys
    import yaml
    from databricks import sdk

    team = Path('src/teams/platform-intelligence')
    monkeypatch.syspath_prepend(str(team / 'agent'))
    from fe_multi_agent import supervisor_agent as supervisor
    order = []

    class Graph:
        async def ainvoke(self, state, config):
            order.append('graph')
            assert state['messages'][-1]['content'] == 'Explain Databricks usage'
            return {'messages': [NS(content='Supervisor graph answer')]}

    monkeypatch.setattr(sdk, 'WorkspaceClient', lambda: NS(current_user=NS(me=lambda: NS(user_name='test'))))
    monkeypatch.setattr(supervisor, 'load_prompts', lambda path: {'supervisor': 'Route to specialists'})
    monkeypatch.setattr(supervisor, 'build_graph', lambda *args: Graph())
    monkeypatch.setattr(supervisor.SupervisorAgent, 'load_context', lambda *args: None)
    monkeypatch.setattr('mlflow.langchain.autolog', lambda: None)
    spec = importlib.util.spec_from_file_location('middleware_supervisor_target', team / 'agent/Agent.py')
    module = importlib.util.module_from_spec(spec)
    with patch('mlflow.models.set_model'):
        spec.loader.exec_module(module)
    assert not issubclass(module.ChatModel, AgentOPSBase)
    assert not hasattr(module, '_GuardedTurn')
    config = yaml.safe_load((team / 'config.yaml').read_text())
    config['tracing'] = {}
    model = wrapper(module.ChatModel(), config)
    guards(model, order)
    if streaming:
        events = list(model.predict_stream(request()))
        assert events[-1].item['content'][0]['text'] == 'Supervisor graph answer'
        assert events[-1].custom_outputs['memory']['mode'] == 'stateless'
    else:
        result = model.predict(request())
        assert extract_response_text(result) == 'Supervisor graph answer'
        assert result.custom_outputs['memory']['mode'] == 'stateless'
    assert order == ['pre', 'graph', 'post']


@pytest.mark.parametrize('redact', [False, True])
def test_explicit_passthrough_and_output_redaction_delivery_policy(redact):
    order = []

    class Target(ResponsesAgent):
        def predict(self, req):
            raise AssertionError('Use the streaming capability')

        def predict_stream(self, req):
            order.append('target')
            yield ResponsesAgentStreamEvent(type='response.output_text.delta',
                item_id='answer', delta='Contact alice@private.example')
            yield ResponsesAgentStreamEvent(type='response.output_item.done',
                item=response('Contact alice@private.example').output[0])

    config = {'middleware': {'streaming': {'output_mode': 'passthrough'}},
              'guardrails': {'enabled': redact, 'pre_llm': {'enabled_checks': []},
                  'post_llm': {'enabled_checks': ['pii_leakage'],
                               'pii_leakage': {'action': 'redact'}}}}
    model = wrapper(Target(), config)
    events = list(model.predict_stream(request()))
    if redact:
        assert [e.type for e in events] == ['response.output_item.done']
        assert '[EMAIL-REDACTED]' in events[0].item['content'][0]['text']
        assert 'private.example' not in str([e.model_dump() for e in events])
    else:
        assert [e.type for e in events] == ['response.output_text.delta', 'response.output_item.done']
    assert order == ['target']


def test_stream_failure_does_not_release_buffered_partial_answer_or_persist_it():
    saves = []

    class Target(NativeTarget):
        async def apredict_stream(self, req):
            yield ResponsesAgentStreamEvent(type='response.output_item.done', item=response('Partial').output[0])
            yield ResponsesAgentStreamEvent(type='response.failed',
                response={'id': 'failed', 'object': 'response', 'status': 'failed', 'output': []})

    model = wrapper(Target())

    async def history(*args, **kwargs):
        return []

    async def save(**kwargs):
        saves.append(kwargs)

    model._middleware.session_store = NS(enabled=True, memory_enabled=False,
        _store=None, _store_error=None, _uc_last_error=None,
        aget_history=history, asave_full_session=save)
    events = []
    with pytest.raises(RuntimeError, match='stream failed'):
        for event in model.predict_stream(request()):
            events.append(event)
    assert events == []
    assert saves == []


def test_wrapper_deadline_cancels_native_target_and_closes_request_resources():
    closed = []

    class Contextual(AgentOPSBase):
        async def _aprocess_request(self, req, context):
            @asynccontextmanager
            async def resource():
                try:
                    yield
                finally:
                    closed.append('resource')
            await context.resources.enter_async_context(resource())
            await asyncio.sleep(10)
            raise AssertionError('Deadline should interrupt the target')

    config = {'guardrails': {'enabled': False}, 'middleware': {'timeout_seconds': 0.5}}
    model = wrapper(Contextual(config), config)
    import mlflow
    # Isolate deadline ownership from a missing local tracing backend. Native
    # tracing is verified separately with an actual file-store exporter.
    with mlflow.tracing.context(enabled=False), pytest.raises(TimeoutError):
        model.predict(request())
    assert closed == ['resource']


@pytest.mark.parametrize('streaming', [False, True])
def test_neutral_target_and_middleware_are_never_copied(monkeypatch, streaming):
    seen = []

    class Target(NativeTarget):
        def __copy__(self):
            raise AssertionError('A neutral target must not be cloned')

        def __deepcopy__(self, memo):
            raise AssertionError('A neutral target must not be deep-copied')

        async def apredict(self, req):
            seen.append(self)
            await asyncio.sleep(0)
            return response(req.custom_inputs['thread_id'])

    target = Target()
    model = wrapper(target)
    assert not hasattr(model._middleware, '_async_request_agent')
    before = dict(vars(model._middleware))

    async def turn(value):
        if streaming:
            events = [event async for event in model.apredict_stream(request(thread_id=value))]
            return events[0].item['content'][0]['text']
        return extract_response_text(await model.apredict(request(thread_id=value)))

    async def scenario():
        assert await asyncio.gather(turn('one'), turn('two')) == ['one', 'two']

    asyncio.run(scenario())
    assert seen == [target, target]
    assert vars(model._middleware) == before
    assert not hasattr(model._middleware, '_request_context')
    assert not hasattr(model._middleware, '_request_resources')


@pytest.mark.parametrize('streaming', [False, True])
def test_direct_callback_context_isolation_includes_history_memory_and_resources(streaming):
    from framework.inference.agent_middleware import AgentOPSMiddleware, RequestContext
    middleware = AgentOPSMiddleware({'guardrails': {'enabled': False}})
    contexts = []
    closed = []

    async def history(thread, **kwargs):
        await asyncio.sleep(0)
        return [{'role': 'assistant', 'content': 'History for ' + thread}]

    async def memory(owner, query):
        await asyncio.sleep(0)
        return [{'key': owner, 'content': query}]

    async def save(**kwargs):
        await asyncio.sleep(0)
        assert kwargs['messages'][0]['content'] == 'History for ' + kwargs['session_id']
        assert kwargs['response_text'] == kwargs['session_id']

    middleware.session_store = NS(enabled=True, memory_enabled=True, _store=None,
        _store_error=None, _uc_last_error=None, aget_history=history,
        arecall_user_memories=memory, asave_full_session=save)

    async def invoke(prepared, context):
        contexts.append(context)
        thread = context.state['thread_id']
        assert context.state['user_memories'][0]['key'] == thread
        assert prepared.input[0].content == 'History for ' + thread
        @asynccontextmanager
        async def resource():
            try:
                yield
            finally:
                assert context.resources is not None
                closed.append(thread)
        await context.resources.enter_async_context(resource())
        await asyncio.sleep(0.01)
        context.state['retrieved_docs'] = [thread]
        # Raw callback results normalize before postchecks and persistence.
        return thread

    async def turn(thread):
        context = RequestContext()
        req = request(thread_id=thread, user_id=thread)
        if streaming:
            events = [e async for e in middleware.execute_stream(req, invoke=invoke, context=context)]
            assert events[0].item['content'][0]['text'] == thread
        else:
            assert extract_response_text(await middleware.execute(req, invoke=invoke, context=context)) == thread
        assert context.state['retrieved_docs'] == [thread]
        assert context.resources is None
        return context

    async def scenario():
        result = await asyncio.gather(turn('one'), turn('two'))
        assert result[0] is not result[1]
        assert result[0].state is not result[1].state
        assert sorted(closed) == ['one', 'two']

    asyncio.run(scenario())
    assert len(contexts) == 2
    assert not any(k in vars(middleware) for k in ('_request_context', '_request_resources', '_augmented_messages'))


@pytest.mark.parametrize('ending', ['success', 'error', 'timeout', 'close'])
def test_callback_stream_resources_close_in_owner_task(ending):
    from framework.inference.agent_middleware import AgentOPSMiddleware, RequestContext
    middleware = AgentOPSMiddleware({'guardrails': {'enabled': False},
        'middleware': {'timeout_seconds': 0.1 if ending == 'timeout' else 10}})
    context = RequestContext()
    closed = []

    async def invoke(req, ctx):
        raise AssertionError('Stream callback must replace prediction')

    async def stream(req, ctx):
        owner = asyncio.current_task()
        @asynccontextmanager
        async def resource():
            try:
                yield
            finally:
                assert asyncio.current_task() is owner
                closed.append(True)
        await ctx.resources.enter_async_context(resource())
        yield ResponsesAgentStreamEvent(type='response.in_progress',
            response={'id': 'progress', 'object': 'response', 'status': 'in_progress', 'output': []})
        if ending == 'error':
            raise RuntimeError('callback failed')
        if ending == 'timeout':
            await asyncio.sleep(10)
        yield ResponsesAgentStreamEvent(type='response.output_item.done', item=response().output[0])

    async def scenario():
        async with aclosing(middleware.execute_stream(request(stream_activity=True),
            invoke=invoke, stream=stream, context=context)) as events:
            assert (await anext(events)).type == 'response.in_progress'
            if ending != 'close':
                if ending in {'error', 'timeout'}:
                    with pytest.raises(RuntimeError if ending == 'error' else TimeoutError):
                        await anext(events)
                else:
                    assert (await anext(events)).type == 'response.output_item.done'
                    with pytest.raises(StopAsyncIteration):
                        await anext(events)
        assert context.resources is None
        assert closed == [True]

    asyncio.run(scenario())


def test_request_context_cannot_be_reused_after_completion_or_concurrently():
    from framework.inference.agent_middleware import AgentOPSMiddleware, RequestContext
    middleware = AgentOPSMiddleware({'guardrails': {'enabled': False}})
    context = RequestContext()
    started = asyncio.Event()
    release = asyncio.Event()

    async def invoke(req, ctx):
        started.set()
        await release.wait()
        return response()

    async def scenario():
        task = asyncio.create_task(middleware.execute(request(), invoke=invoke, context=context))
        await started.wait()
        with pytest.raises(ValueError, match='single-use'):
            await middleware.execute(request(), invoke=invoke, context=context)
        assert context.resources is not None
        release.set()
        await task
        with pytest.raises(ValueError, match='single-use'):
            await middleware.execute(request(), invoke=invoke, context=context)

    asyncio.run(scenario())
