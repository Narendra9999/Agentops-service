import hashlib
import json
import time

import pytest

from fe_multi_agent.config import Settings
from fe_multi_agent.memory import resolve_memory_context, sign_context

MESSAGES = [{'role': 'user', 'content': 'Explain MLflow tracing'}]


def test_playground_request_is_stateless_without_mutating_server_settings():
    settings = Settings(memory_enabled=True, unsigned_context_policy='stateless')
    turn, context, mode = resolve_memory_context(settings, {'owner_id': 'untrusted'}, MESSAGES)
    assert mode == 'stateless' and context is None and not turn.memory_enabled
    assert settings.memory_enabled  # A concurrent signed App request keeps its memory.


def test_signed_app_request_retains_verified_memory(monkeypatch):
    monkeypatch.setenv('AGENT_CONTEXT_SIGNING_KEY', 'test-key')
    settings = Settings(memory_enabled=True, unsigned_context_policy='stateless')
    context = {'owner_id': 'verified-user', 'expires_at': int(time.time()) + 300,
               'messages_sha256': hashlib.sha256(json.dumps(MESSAGES, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
    turn, verified, mode = resolve_memory_context(settings, {'application_context': sign_context(context)}, MESSAGES)
    assert turn is settings and mode == 'persistent' and verified == context


@pytest.mark.parametrize('envelope', [{}, None, {'payload': 'tampered', 'signature': 'wrong'}])
def test_invalid_context_never_falls_back_to_stateless(monkeypatch, envelope):
    monkeypatch.setenv('AGENT_CONTEXT_SIGNING_KEY', 'test-key')
    settings = Settings(memory_enabled=True, unsigned_context_policy='stateless')
    with pytest.raises(ValueError, match='signature'):
        resolve_memory_context(settings, {'application_context': envelope}, MESSAGES)


def test_default_policy_requires_signed_context():
    with pytest.raises(ValueError, match='Signed application context'):
        resolve_memory_context(Settings(memory_enabled=True), {}, MESSAGES)


def test_progress_events_have_a_response_envelope():
    from fe_multi_agent.supervisor_agent import stream_prediction
    from fe_multi_agent.progress import emit
    from mlflow.pyfunc import ResponsesAgent
    from mlflow.types.responses import Response, ResponsesAgentResponse

    def predict(request):
        emit('request_started', 'supervisor', 'Preparing the request')
        return ResponsesAgentResponse(output=[ResponsesAgent.create_text_output_item(text='Done', id='answer')])

    from types import SimpleNamespace
    events = list(stream_prediction(predict, SimpleNamespace(custom_inputs={'stream_activity': True})))
    progress = [e for e in events if e.type == 'response.in_progress']
    assert progress and progress[0].custom_outputs['activity']['kind'] == 'request_started'
    for e in progress:
        assert Response.model_validate(e.model_dump()['response']).output == []
    assert events[-1].type == 'response.completed'


def test_default_stream_returns_only_documented_output_events():
    from fe_multi_agent.supervisor_agent import stream_prediction
    from fe_multi_agent.progress import emit
    from mlflow.pyfunc import ResponsesAgent
    from mlflow.types.responses import ResponsesAgentResponse, ResponseOutputItemDoneEvent

    def predict(request):
        emit('request_started', 'supervisor', 'Preparing')
        emit('agent_started', 'cost_estimation', 'Calling Genie')
        return ResponsesAgentResponse(
            output=[ResponsesAgent.create_text_output_item(text='Need runtime', id='answer')],
            custom_outputs={'trace_id': 'tr-test'})

    events = list(stream_prediction(predict, None))
    assert [event.type for event in events] == ['response.output_item.done']
    item = ResponseOutputItemDoneEvent.model_validate(events[0].model_dump())
    assert item.item.id == 'answer'
    assert events[0].custom_outputs['trace_id'] == 'tr-test'
