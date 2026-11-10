import pytest
from fe_multi_agent.evaluation import response_text
from fe_multi_agent.evaluation import routing

PAYLOAD={'output':[{'id':'answer1','type':'message','role':'assistant',
    'content':[{'type':'output_text','text':'Case is investigating, version 2','annotations':[]}]}]}


def test_responses_dictionary_content_is_scored_as_text():
    assert response_text(PAYLOAD)=='Case is investigating, version 2'


def test_mlflow_responses_object_is_scored_as_text():
    pytest.importorskip('numpy')
    from mlflow.types.responses import ResponsesAgentResponse
    assert response_text(ResponsesAgentResponse(**PAYLOAD))=='Case is investigating, version 2'


@pytest.mark.parametrize('actual,expected,passes',[
    (['supervisor'],['cost_estimation'],False),
    (['supervisor','cost_estimation','platform_docs'],['cost_estimation'],False),
    (['supervisor','cost_estimation'],['cost_estimation'],True),
    (['cost_estimation','platform_docs'],['cost_estimation','platform_docs'],True),
])
def test_routing_requires_actual_expected_workers_and_rejects_unneeded_workers(actual,expected,passes):
    from types import SimpleNamespace as NS
    trace=NS(search_spans=lambda:[NS(attributes={'agentops.agent_id':agent}) for agent in actual])
    assert routing(trace,{'agents':expected}) is passes


def test_serving_routing_uses_returned_execution_trace(monkeypatch):
    import sys
    from types import SimpleNamespace as NS
    caller=NS(info=NS(trace_id='caller'),search_spans=lambda:[])
    remote=NS(search_spans=lambda:[NS(attributes={'agentops.agent_id':'cost_estimation'})])
    seen=[]
    def get_trace(reference):seen.append(reference);return remote
    monkeypatch.setitem(sys.modules,'mlflow',NS(get_trace=get_trace))
    assert routing(caller,{'agents':['cost_estimation']},{'execution_trace_id':'serving-trace'})
    assert seen==['serving-trace']


def test_unavailable_serving_trace_fails_routing_closed(monkeypatch):
    import sys,time
    from types import SimpleNamespace as NS
    monkeypatch.setitem(sys.modules,'mlflow',NS(get_trace=lambda _:None))
    monkeypatch.setattr(time,'sleep',lambda _:None)
    with pytest.raises(RuntimeError,match='Serving execution trace is unavailable'):
        routing(NS(info=NS(trace_id='caller')),{'agents':['cost_estimation']},
                {'execution_trace_id':'unavailable'})
