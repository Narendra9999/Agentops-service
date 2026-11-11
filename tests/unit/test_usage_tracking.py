"""Usage capture must preserve totals and never repeat a model invocation."""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from framework.observability import usage_tracking as tracking


@pytest.fixture(autouse=True)
def raw_tracking(monkeypatch):
    monkeypatch.setattr(tracking, 'TokenTracker', None)


def test_raw_accounting_preserves_details_and_multiple_calls():
    context = {}
    assert tracking.record_usage([
        {'input_tokens': 10, 'output_tokens': 4, 'input_token_details': {'cache_read': 3}},
        {'prompt_tokens': 20, 'completion_tokens': 5, 'reasoning_tokens': 2},
    ], context)
    tags = tracking.emit_standard_trace_metrics(0, context)
    assert tags['agentops.tokens.total'] == '39'
    assert tags['agentops.tokens.cached'] == '3'
    assert tags['agentops.tokens.reasoning'] == '2'
    assert tags['agentops.llm_requests'] == '2'


@pytest.mark.parametrize('source', ['metadata', 'callback', 'response'])
def test_sources_are_alternatives_not_additive(monkeypatch, source):
    usage = {'input_tokens': 10, 'output_tokens': 2}

    @contextmanager
    def callback():
        yield SimpleNamespace(usage_metadata={'model': usage} if source == 'metadata' else {})

    monkeypatch.setattr(tracking, 'get_usage_metadata_callback', callback)
    calls = []

    def invoke(payload, config):
        calls.append(payload)
        if source != 'response':
            config['callbacks'][0].on_llm_end(SimpleNamespace(llm_output={'usage': usage}))
        return {'usage': usage}

    context = {}
    tracking.invoke_with_usage_tracking(SimpleNamespace(invoke=invoke), 'question', context)
    assert len(calls) == 1
    assert len(context['token_usage_history']) == 1
    assert context['token_usage_history'][0]['total_tokens'] == 12


def test_model_errors_are_not_retried(monkeypatch):
    monkeypatch.setattr(tracking, 'get_usage_metadata_callback', None)
    invoke = Mock(side_effect=TypeError('model failed after starting'))
    with pytest.raises(TypeError, match='model failed'):
        tracking.invoke_with_usage_tracking(SimpleNamespace(invoke=invoke), 'question', {})
    invoke.assert_called_once()


def test_tracker_costs_use_same_accounting_path(monkeypatch):
    tracker = Mock()
    tracker.to_dict.return_value = {'prompt_tokens': 10, 'completion_tokens': 2,
                                    'total_tokens': 12, 'requests': 1, 'total_cost_usd': 0.03}
    monkeypatch.setattr(tracking, 'TokenTracker', Mock(return_value=tracker))
    context = {}
    assert tracking.record_response_usage({'usage': {'input_tokens': 10, 'output_tokens': 2}}, context, 'model')
    assert tracking.emit_standard_trace_metrics(0, context)['agentops.cost_usd'] == '0.030000'
    tracker.track.assert_called_once()


def test_no_usage_does_not_record_zero_call():
    context = {}
    assert not tracking.record_response_usage({'output': []}, context)
    assert context == {}
