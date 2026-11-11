"""Exercise the serving wrapper without importing the heavy model-loading runtime."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Generator
import pytest
import mlflow


def wrapper(monkeypatch):
    source=Path(__file__).resolve().parents[3]/'src/framework/inference/cost_tracking_model.py'
    tree=ast.parse(source.read_text())
    nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in {'_record_usage','CostTrackingResponsesModel'}]
    namespace={'ResponsesAgent':type('ResponsesAgent',(),{'responses_agent_output_reducer':staticmethod(lambda events:events)}),
               'mlflow':NS(trace=lambda **kwargs:lambda fn:fn),'ResponsesAgentRequest':object,
               'ResponsesAgentResponse':object,'ResponsesAgentStreamEvent':object,'Generator':Generator,
               'logger':NS(warning=lambda *a,**k:None)}
    namespace['AsyncResponsesAgent'] = namespace['ResponsesAgent']
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    result=namespace['CostTrackingResponsesModel']()
    result._inner_is_responses_agent=True
    tags=[]
    monkeypatch.setattr(mlflow,'update_current_trace',lambda **kwargs:tags.append(kwargs['tags']))
    return result,tags


def test_streaming_usage_snapshots_are_not_summed_and_events_unchanged(monkeypatch):
    model,tags=wrapper(monkeypatch)
    events=[{'type':'response.in_progress'},
            {'type':'response.output_item.done','usage':{'input_tokens':100,'output_tokens':10}},
            {'type':'response.completed','response':{'usage':{'input_tokens':100,'output_tokens':20}}}]
    model._inner=NS(predict_stream=lambda request:iter(events))
    assert list(model.predict_stream(object()))==events
    assert len(tags)==1 and tags[0]['agentops.tokens.total']=='120'
    assert tags[0]['agentops.usage.scope']=='response_aggregate'


def test_stream_error_is_preserved_and_observed_usage_tagged(monkeypatch):
    model,tags=wrapper(monkeypatch)
    def stream(request):
        yield {'type':'response.output_item.done','usage':{'input_tokens':7,'output_tokens':2}}
        raise RuntimeError('original provider failure')
    model._inner=NS(predict_stream=stream)
    with pytest.raises(RuntimeError,match='original provider failure'):
        list(model.predict_stream(object()))
    assert tags[0]['agentops.tokens.total']=='9'
