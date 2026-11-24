"""Native instrumentation belongs to model initialization, not team callbacks."""
from unittest.mock import Mock, patch

import pytest

from framework.observability.tracing import configure_native_instrumentation


@pytest.mark.parametrize('integration', ['langchain', 'openai'])
@pytest.mark.parametrize('enabled', [True, False])
def test_explicit_toggle(integration, enabled):
    module = Mock()
    with patch('importlib.import_module', return_value=module) as load:
        configure_native_instrumentation({'tracing': {'instrumentation': {
            'enabled': enabled, 'integration': integration}}})
    load.assert_called_once_with('mlflow.' + integration)
    module.autolog.assert_called_once_with(disable=not enabled, log_traces=enabled, silent=False)


def test_omitted_configuration_preserves_external_instrumentation():
    with patch('importlib.import_module') as load:
        configure_native_instrumentation({})
    load.assert_not_called()


@pytest.mark.parametrize('options', [True, {'enabled': 'false', 'integration': 'openai'},
    {'enabled': True, 'integration': 'unknown'}, {'enabled': True, 'integration': 'openai', 'typo': True}])
def test_invalid_configuration_fails_before_model_load(options):
    with patch('importlib.import_module') as load, pytest.raises(ValueError):
        configure_native_instrumentation({'tracing': {'instrumentation': options}})
    load.assert_not_called()


def test_tracing_initializes_before_customer_import_and_load():
    from types import SimpleNamespace
    from framework.inference.cost_tracking_model import CostTrackingResponsesModel
    events = []
    target = SimpleNamespace(load_context=lambda context: events.append('load'))
    config = {'tracing': {'instrumentation': {'enabled': True, 'integration': 'openai'}}}
    def resolve(config):
        events.append('import')
        return target
    with patch('framework.observability.tracing.configure_native_instrumentation',
               side_effect=lambda config: events.append('instrument')), \
         patch('framework.inference.cost_tracking_model._resolve_inner_model', side_effect=resolve), \
         patch('framework.inference.cost_tracking_model.AgentOPSMiddleware'):
        CostTrackingResponsesModel().load_context(SimpleNamespace(model_config=config))
    assert events == ['instrument', 'import', 'load']


def test_native_openai_spans_usage_cost_and_handled_errors(tmp_path):
    """Real MLflow/OpenAI integration; only the HTTP transport is simulated."""
    import httpx
    import mlflow
    from openai import OpenAI, BadRequestError
    from types import SimpleNamespace
    previous_uri = mlflow.get_tracking_uri()
    previous_experiment = mlflow.tracking.fluent._active_experiment_id
    mlflow.set_tracking_uri('sqlite:///' + str(tmp_path / 'traces.db'))
    mlflow.set_experiment('native-instrumentation-test')
    config = {'llm': {'endpoint': 'validation-model'}, 'tracing': {'instrumentation': {
        'enabled': True, 'integration': 'openai'}}}
    def reply(request):
        if b'failure' in request.content:
            return httpx.Response(400, json={'error': {'message': 'controlled failure',
                'type': 'invalid_request_error', 'code': 'validation'}})
        return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion',
            'created': 1, 'model': 'validation-model', 'choices': [{'index': 0,
            'message': {'role': 'assistant', 'content': 'hello'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}})
    client = OpenAI(api_key='mock-only', base_url='https://validation.invalid/v1',
        http_client=httpx.Client(transport=httpx.MockTransport(reply)), max_retries=0)
    from mlflow.tracing.config import get_config
    previous_processors = list(get_config().span_processors)
    try:
        with patch('framework.observability.usage_tracking.TokenTracker',
                   return_value=SimpleNamespace(pricing_per_million=lambda: {'input': 1, 'output': 2})):
            configure_native_instrumentation(config)
        for enabled in [True, False]:
            if not enabled:
                config['tracing']['instrumentation']['enabled'] = False
                configure_native_instrumentation(config)
            with mlflow.start_span(name='middleware') as root:
                client.chat.completions.create(model='validation-model', messages=[{'role': 'user', 'content': 'hello'}])
                with pytest.raises(BadRequestError):
                    client.chat.completions.create(model='validation-model', messages=[{'role': 'user', 'content': 'failure'}])
                trace_id = root.trace_id
            trace = mlflow.get_trace(trace_id)
            models = [s for s in trace.data.spans if s.span_type == 'CHAT_MODEL']
            assert len(models) == (2 if enabled else 0)
            assert str(trace.info.state) == 'OK'
            if enabled:
                successful = next(s for s in models if s.status.status_code.value == 'OK')
                assert successful.attributes['mlflow.chat.tokenUsage']['total_tokens'] == 15
                assert successful.attributes['mlflow.llm.cost']['total_cost'] == pytest.approx(0.00002)
                assert sum(s.status.status_code.value == 'ERROR' for s in models) == 1
                assert not root.get_attribute('mlflow.chat.tokenUsage')
    finally:
        import mlflow.openai
        mlflow.openai.autolog(disable=True)
        mlflow.tracing.configure(span_processors=previous_processors)
        mlflow.set_tracking_uri(previous_uri)
        mlflow.tracking.fluent._active_experiment_id = previous_experiment
        client.close()


def test_request_restores_cost_processor_after_runtime_reset():
    """Runtime tracing reconfiguration must not lose framework pricing."""
    import mlflow
    from mlflow.tracing.config import get_config
    from framework.observability.usage_tracking import capture_target_usage
    config = {'llm': {'endpoint': 'validation-model'}, 'tracing': {
        'instrumentation': {'enabled': True, 'integration': 'langchain'}}}
    previous = list(get_config().span_processors)
    external = lambda span: None
    try:
        mlflow.tracing.configure(span_processors=[external])
        with patch('framework.observability.usage_tracking.TokenTracker',
                   return_value=type('Rates', (), {'pricing_per_million': lambda self: {
                       'input': 1, 'output': 2}})()):
            for _ in range(2):
                with capture_target_usage({}, config):
                    pass
        processors = get_config().span_processors
        assert external in processors
        assert sum(p.__name__ == 'agentops.openai.cost.validation-model'
                   for p in processors) == 1
        span = Mock(span_type='CHAT_MODEL')
        attrs = {'mlflow.llm.model': 'validation-model', 'mlflow.chat.tokenUsage': {
            'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}}
        span.get_attribute.side_effect = attrs.get
        span.set_attribute.side_effect = attrs.__setitem__
        for processor in list(processors):
            processor(span)
        assert attrs['mlflow.llm.cost']['total_cost'] == pytest.approx(.00002)
    finally:
        mlflow.tracing.configure(span_processors=previous)
