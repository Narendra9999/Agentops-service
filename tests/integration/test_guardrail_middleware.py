"""Real ResponsesAgent integration for opt-in transformation guardrails."""
import asyncio
from types import SimpleNamespace as NS

import pytest
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse, ResponsesAgentStreamEvent
from framework.agent_base import AgentOPSBase
from framework.guardrails.middleware import PIIRedactionMiddleware


@pytest.fixture(autouse=True)
def local_environment(monkeypatch):
    monkeypatch.setenv('AGENTOPS_ENVIRONMENT', 'test')


def policy(stage):
    name = 'pii' if stage == 'pre_llm' else 'pii_leakage'
    return {stage: {'enabled_checks': [name], name: {'action': 'redact'}}}


class Echo(AgentOPSBase):
    async def _aprocess_request(self, request):
        return ResponsesAgentResponse(output=[self.create_text_output_item(
            text=request.input[-1].content, id='answer')])


def agent(cls=Echo, config=None):
    result = cls({'guardrails': config or {}})
    result.session_store = NS(enabled=False, memory_enabled=False, _store_error=None, _uc_last_error=None)
    return result


def request(text='Contact alice@private.example for Databricks guidance', **custom):
    return ResponsesAgentRequest(input=[{'role': 'user', 'content': text}], custom_inputs=custom)


def test_default_policy_still_blocks_pii():
    response = agent(config={'pre_llm': {'enabled_checks': ['pii']}}).predict(request())
    assert 'remove sensitive data' in response.output[0].content[0]['text']


def test_pre_redaction_transforms_request_without_mutating_original():
    original = request()
    response = agent(config={**policy('pre_llm'), 'post_llm': {'enabled_checks': []}}).predict(original)
    assert response.output[0].content[0]['text'] == 'Contact [EMAIL-REDACTED] for Databricks guidance'
    assert 'alice@private.example' in original.input[0].content


def test_selected_checks_and_global_enabled_control_redaction():
    for config in [
        {'enabled': False, **policy('pre_llm')},
        {'pre_llm': {'enabled_checks': [], 'pii': {'action': 'redact'}}, 'post_llm': {'enabled_checks': []}},
    ]:
        response = agent(config=config).predict(request())
        assert 'alice@private.example' in response.output[0].content[0]['text']


def test_signed_input_is_not_silently_modified():
    with pytest.raises(ValueError, match='verify and rebind'):
        agent(config=policy('pre_llm')).predict(request(application_context={'signature': 'opaque'}))


def test_post_redacts_nested_genie_content_but_preserves_usage_and_ids():
    middleware = PIIRedactionMiddleware(policy('post_llm'))
    response = ResponsesAgentResponse(output=[Echo.create_text_output_item(text='Email bob@private.example', id='answer')],
        usage={'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12,
               'input_tokens_details': {'cached_tokens': 0}, 'output_tokens_details': {'reasoning_tokens': 0}},
        custom_outputs={'trace_id': 'tr-123', 'genie_result': {'tables': [{'rows': [['bob@private.example']]}]}})
    result = asyncio.run(middleware.process_response(response))
    assert result.output[0].id == 'answer'
    assert result.output[0].content[0]['text'] == 'Email [EMAIL-REDACTED]'
    assert result.custom_outputs['genie_result']['tables'][0]['rows'][0][0] == '[EMAIL-REDACTED]'
    assert result.custom_outputs['trace_id'] == 'tr-123'
    assert result.usage.total_tokens == 12
    assert response.output[0].content[0]['text'] == 'Email bob@private.example'


def test_stream_buffers_cross_chunk_pii_and_saves_sanitized_output():
    saved = []

    class Stream(Echo):
        async def _aprocess_request_stream(self, req):
            for part in ['alice@', 'private.example']:
                yield ResponsesAgentStreamEvent(type='response.output_text.delta', item_id='answer', delta=part)
            yield ResponsesAgentStreamEvent(type='response.output_item.done',
                item=self.create_text_output_item(text='alice@private.example', id='answer'))

    result = agent(Stream, {'pre_llm': {'enabled_checks': []}, **policy('post_llm')})
    result.session_store.enabled = True

    async def save(**kwargs):
        saved.append(kwargs)

    result.session_store.asave_full_session = save
    req = ResponsesAgentRequest(input=[{'role': 'user', 'content': 'previous question'},
        {'role': 'assistant', 'content': 'previous answer'}, {'role': 'user', 'content': 'Please respond'}])
    events = list(result.predict_stream(req))
    assert [event.type for event in events] == ['response.output_item.done']
    assert events[0].item['content'][0]['text'] == '[EMAIL-REDACTED]'
    assert saved[0]['response_text'] == '[EMAIL-REDACTED]'


def test_buffered_stream_cannot_release_other_blocked_content():
    class Stream(Echo):
        async def _aprocess_request_stream(self, req):
            yield ResponsesAgentStreamEvent(type='response.output_item.done', item=self.create_text_output_item(
                text='Guaranteed profit for alice@private.example', id='answer'))

    config = {'pre_llm': {'enabled_checks': []}, 'post_llm': {
        'enabled_checks': ['pii_leakage', 'compliance'], 'pii_leakage': {'action': 'redact'}}}
    events = list(agent(Stream, config).predict_stream(request('Explain Databricks costs')))
    assert len(events) == 1
    assert 'Guaranteed profit' not in events[0].item['content'][0]['text']


@pytest.mark.parametrize('config', [
    {'pre_llm': {'pii': {'action': 'unknown'}}},
    {'post_llm': {'pii_leakage': {'action': 'redact', 'entities': ['unknown']}}},
])
def test_invalid_transform_configuration_fails_at_initialization(config):
    with pytest.raises(ValueError):
        PIIRedactionMiddleware(config)
