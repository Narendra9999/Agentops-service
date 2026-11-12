from types import SimpleNamespace as NS
from uuid import uuid4
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from framework.observability.token_usage import normalize_usage, response_usage, estimate_cost
from fe_multi_agent.usage import UsageMeter, agent_usage_scope, extract_llm_usage


def response(tokens, **metadata):
    return LLMResult(generations=[[ChatGeneration(message=AIMessage(content='answer',usage_metadata=tokens,response_metadata=metadata))]])


def test_responses_usage_including_cached_and_reasoning_is_not_added_to_total():
    value={'input_tokens':100,'output_tokens':50,'total_tokens':150,'input_tokens_details':{'cached_tokens':20},'output_tokens_details':{'reasoning_tokens':30}}
    usage=response_usage({'response':{'usage':value}})
    assert usage=={'input_tokens':100,'output_tokens':50,'total_tokens':150,'cached_tokens':20,'reasoning_tokens':30}
    assert estimate_cost(usage,{'input':3,'output':15,'cached_input':1.5})==pytest.approx(.00102)
    assert estimate_cost(usage,{'input':3,'output':15}) is None
    assert normalize_usage({}) is None


def test_provider_aliases_and_langchain_cache_details():
    usage=normalize_usage({'prompt_tokens':12,'completion_tokens':4,'completion_tokens_details':{'reasoning_tokens':2}})
    assert usage['total_tokens']==16 and usage['reasoning_tokens']==2
    result=response({'input_tokens':12,'output_tokens':4,'total_tokens':16,'input_token_details':{'cache_read':5},'output_token_details':{'reasoning':2}})
    assert extract_llm_usage(result)['cached_tokens']==5
    assert extract_llm_usage(result)['reasoning_tokens']==2


def test_call_deduplication_and_agent_model_breakdown():
    meter=UsageMeter({'model-a':{'input':2,'output':4}})
    one,two=uuid4(),uuid4()
    for agent,run,model in [('supervisor',one,'model-a'),('platform_docs',two,'model-b')]:
        with agent_usage_scope(agent):
            meter.on_chat_model_start({},[],run_id=run,invocation_params={'model':model})
        result=response({'input_tokens':10,'output_tokens':5,'total_tokens':15},token_usage={'prompt_tokens':10,'completion_tokens':5})
        meter.on_llm_end(result,run_id=run)
        meter.on_llm_end(result,run_id=run)
    summary=meter.summary()
    assert summary['llm_requests']==2 and summary['total_tokens']==30
    assert all(g['total_tokens']==15 for g in summary['by_agent_model'])
    assert [g['agent'] for g in summary['by_agent_model']]==['supervisor','platform_docs']
    assert summary['unpriced_calls']==1 and summary['cost_status']=='partial_estimate'
    assert summary['estimated_cost_usd']==pytest.approx(.00004)


def test_missing_usage_and_genie_are_explicitly_partial():
    meter=UsageMeter()
    meter.on_llm_end(NS(llm_output={},generations=[]),run_id=uuid4())
    meter.unmetered.add('genie')
    summary=meter.summary()
    assert summary['usage_missing_calls']==1
    assert summary['usage_status']=='partial'
    assert summary['unmetered_services']==['genie']


def test_request_without_model_calls_is_zero_usage():
    summary=UsageMeter().summary()
    assert summary['total_tokens']==0 and summary['llm_requests']==0
    assert summary['estimated_cost_usd']==0 and summary['usage_status']=='complete'


def test_compact_tags_match_response_and_preserve_detailed_breakdown(monkeypatch):
    import mlflow
    from fe_multi_agent.usage import capture_request_usage, current_meter
    tags={}
    monkeypatch.setattr(mlflow,'update_current_trace',lambda **kwargs:tags.update(kwargs['tags']))
    class Response:
        def __init__(self,**data):self.data=data
        def model_dump(self,**kwargs):return self.data
    @capture_request_usage
    def predict(self):
        for agent in ['supervisor','cost_estimation','platform_docs']:
            run=uuid4()
            with agent_usage_scope(agent):
                current_meter().on_chat_model_start({},[],run_id=run,invocation_params={'model':'system.ai.claude-sonnet-4-6'})
            current_meter().on_llm_end(response({'input_tokens':100,'output_tokens':20,'total_tokens':120}),run_id=run)
        return Response(output=[],custom_outputs={'tags':{'agentops.system_id':'test-system'}})
    result=predict(NS(settings=NS(usage_rates_per_million={})))
    assert all(len(value.encode())<=255 for value in tags.values())
    assert tags['agentops.tokens.total']=='360'
    assert not any(k.startswith('agentops.usage.agent_model.') for k in tags)
    assert len(tags)<20
    assert result.data['custom_outputs']['tags']=={'agentops.system_id':'test-system',**tags}
    assert result.data['usage']['total_tokens']==360
    assert [g['agent'] for g in result.data['custom_outputs']['usage_metrics']['by_agent_model']]==['supervisor','cost_estimation','platform_docs']
    assert len(result.data['custom_outputs']['usage_metrics']['by_agent_model'])==3


def test_failed_request_still_exports_compact_usage_and_resets_meter(monkeypatch):
    import mlflow
    from fe_multi_agent.usage import capture_request_usage, current_meter
    tags={}
    monkeypatch.setattr(mlflow,'update_current_trace',lambda **kwargs:tags.update(kwargs['tags']))
    @capture_request_usage
    def predict(self):
        current_meter().unmetered.add('genie')
        raise RuntimeError('worker failed')
    with pytest.raises(RuntimeError,match='worker failed'):
        predict(NS(settings=NS(usage_rates_per_million={})))
    assert tags['agentops.usage.status']=='partial'
    assert tags['agentops.cost.status']=='unavailable'
    assert 'agentops.cost_usd' not in tags
    assert current_meter() is None


def test_remote_rollup_deduplicates_invocations_and_does_not_sum_nested_breakdown():
    meter=UsageMeter({'model-a':{'input':2,'output':4}})
    run=uuid4();meter.on_chat_model_start({},[],run_id=run,invocation_params={'model':'model-a'})
    meter.on_llm_end(response({'input_tokens':10,'output_tokens':5,'total_tokens':15}),run_id=run)
    remote={'usage':{'input_tokens':100,'output_tokens':20},'custom_outputs':{
        'trace_id':'trace-worker','execution_id':'worker-execution',
        'usage_metrics':{'llm_requests':2,'usage_status':'complete','cost_status':'estimated',
                        'estimated_cost_usd':0.01,'model_call_ids':['remote-1','remote-2'],
                        'by_agent_model':[{'total_tokens':120}]}}}
    meter.observe_remote('worker','endpoint',remote,invocation_id='transport-1')
    meter.observe_remote('worker','endpoint',remote,invocation_id='transport-1')
    summary=meter.summary()
    assert summary['total_tokens']==135 and summary['llm_requests']==3
    assert summary['estimated_cost_usd']==pytest.approx(.01004)
    assert summary['usage_status']=='complete' and summary['cost_status']=='estimated'
    assert len(summary['remote_workers'])==1
    assert summary['scope']=='observed_model_calls_and_remote_workers'
    assert summary['remote_workers'][0]['accounting_status']=='included'
    assert meter.summary()['total_tokens']==135  # summary is pure, not accumulated twice


def test_remote_missing_usage_or_cost_remains_partial_not_zero():
    meter=UsageMeter()
    meter.observe_remote('one','endpoint',{'output':[]},invocation_id='one')
    meter.observe_remote('two','endpoint',{'usage':{'input_tokens':10,'output_tokens':2}},invocation_id='two')
    summary=meter.summary()
    assert summary['total_tokens']==12 and summary['usage_status']=='partial'
    assert summary['cost_status']=='unavailable' and summary['estimated_cost_usd'] is None
    assert 'remote_worker:one' in summary['unmetered_services']


def test_remote_aggregate_overlapping_local_calls_is_not_added():
    meter=UsageMeter();run=uuid4()
    meter.on_llm_end(response({'input_tokens':10,'output_tokens':5,'total_tokens':15}),run_id=run)
    meter.observe_remote('worker','endpoint',{'usage':{'input_tokens':10,'output_tokens':5},
        'custom_outputs':{'usage_metrics':{'model_call_ids':[str(run)]}}},invocation_id='transport')
    summary=meter.summary()
    assert summary['total_tokens']==15 and summary['usage_status']=='partial'
    assert summary['remote_workers'][0]['accounting_status']=='overlap_excluded'


def test_distinct_invocations_sharing_trace_are_both_counted():
    meter=UsageMeter()
    remote={'usage':{'input_tokens':10,'output_tokens':2},'custom_outputs':{'trace_id':'joined-trace'}}
    meter.observe_remote('worker','endpoint',remote,invocation_id='one')
    meter.observe_remote('worker','endpoint',remote,invocation_id='two')
    assert meter.summary()['total_tokens']==24


def test_remote_agentops_tags_supply_cost_and_call_count():
    meter=UsageMeter()
    meter.observe_remote('worker','endpoint',{'usage':{'input_tokens':10,'output_tokens':2},
        'custom_outputs':{'tags':{'agentops.cost_usd':'.02','agentops.llm_requests':'2',
                                 'agentops.usage.status':'complete','agentops.cost.status':'estimated'}}},invocation_id='one')
    summary=meter.summary()
    assert summary['estimated_cost_usd']==.02 and summary['llm_requests']==2
    assert summary['cost_status']=='estimated'
