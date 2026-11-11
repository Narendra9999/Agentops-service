import pytest
from framework.observability.token_usage import set_native_cost_fallback
from fe_multi_agent import usage


class Span:
    def __init__(self, kind='CHAT_MODEL', **attributes):
        self.span_type=kind
        self.attributes={'mlflow.llm.model':'system.ai.claude-sonnet-4-6',
                         'mlflow.chat.tokenUsage':{'input_tokens':100,'output_tokens':50,'cache_read_input_tokens':20},**attributes}
    def get_attribute(self,key):return self.attributes.get(key)
    def set_attribute(self,key,value):self.attributes[key]=value


RATES={'system.ai.claude-sonnet-4-6':{'input':3,'output':15,'cached_input':1.5}}


def test_native_cost_splits_input_output_and_does_not_double_count_cache():
    span=Span();set_native_cost_fallback(span,RATES)
    cost=span.attributes['mlflow.llm.cost']
    assert cost['input_cost']==pytest.approx(.00027)
    assert cost['output_cost']==pytest.approx(.00075)
    assert cost['total_cost']==pytest.approx(.00102)
    assert span.attributes['agentops.cost.source']=='configured_rate_estimate'
    set_native_cost_fallback(span,{'system.ai.claude-sonnet-4-6':{'input':100,'output':100}})
    assert span.attributes['mlflow.llm.cost']==cost


def test_native_cost_preserves_existing_cost_and_skips_parents_missing_usage_rates():
    existing={'input_cost':.1,'output_cost':.2,'total_cost':.3}
    span=Span(**{'mlflow.llm.cost':existing});set_native_cost_fallback(span,RATES)
    assert span.attributes['mlflow.llm.cost'] is existing
    for span,rates in [(Span(kind='AGENT'),RATES),(Span(),{}),
                       (Span(**{'mlflow.chat.tokenUsage':None}),RATES),
                       (Span(),{'system.ai.claude-sonnet-4-6':{'input':-1,'output':15}})]:
        set_native_cost_fallback(span,rates)
        assert 'mlflow.llm.cost' not in span.attributes


def test_processor_is_request_scoped_and_install_preserves_other_processors():
    import mlflow
    from mlflow.tracing.config import get_config
    def redact(span):pass
    with mlflow.tracing.configure(span_processors=[redact]):
        usage.install_native_cost_processor();usage.install_native_cost_processor()
        assert get_config().span_processors==[redact,usage.native_cost_processor]
        span=Span();usage.native_cost_processor(span)
        assert 'mlflow.llm.cost' not in span.attributes
        token=usage._meter.set(usage.UsageMeter(RATES))
        try:usage.native_cost_processor(span)
        finally:usage._meter.reset(token)
        assert span.attributes['mlflow.llm.cost']['total_cost']==pytest.approx(.00102)


def test_install_recovers_after_runtime_configuration_reset():
    import mlflow
    from mlflow.tracing.config import get_config
    from concurrent.futures import ThreadPoolExecutor
    def redact(span):pass
    with mlflow.tracing.configure(span_processors=[redact]):
        usage.install_native_cost_processor()
        mlflow.tracing.configure(span_processors=[redact])
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _:usage.install_native_cost_processor(),range(12)))
        assert get_config().span_processors==[redact,usage.native_cost_processor]


def test_shared_native_export_preserves_autologged_usage_and_never_prices_parent():
    from framework.observability.token_usage import export_native_model_usage
    span = Span()
    original = span.attributes['mlflow.chat.tokenUsage']
    export_native_model_usage(span, usage={'input_tokens': 999, 'output_tokens': 999},
                              model_name='do-not-overwrite', rates_by_model=RATES)
    assert span.attributes['mlflow.chat.tokenUsage'] is original
    assert span.attributes['mlflow.llm.model'] == 'system.ai.claude-sonnet-4-6'
    assert span.attributes['mlflow.llm.cost']['total_cost'] == pytest.approx(.00102)
    parent = Span(kind='AGENT', **{'mlflow.chat.tokenUsage': None})
    export_native_model_usage(parent, usage={'input_tokens': 10}, rates_by_model=RATES)
    assert parent.attributes['mlflow.chat.tokenUsage'] is None
    assert 'mlflow.llm.cost' not in parent.attributes
