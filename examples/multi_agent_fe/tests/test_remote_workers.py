from types import SimpleNamespace as NS
from uuid import uuid4
import pytest
from langchain_core.messages import HumanMessage
from fe_multi_agent.config import Settings
from fe_multi_agent.services import ExistingResponsesWorker, ExistingGenieWorker
from fe_multi_agent.usage import UsageMeter, _meter


def test_existing_responses_worker_contract_and_separate_usage(monkeypatch):
    import requests
    calls=[]
    class Response:
        def raise_for_status(self):pass
        def json(self):return {'output':[{'content':[{'type':'output_text','text':'Existing worker answer'}]}],
            'usage':{'input_tokens':100,'output_tokens':20},'custom_outputs':{'trace_id':'worker-trace'}}
    def post(url,**kwargs):calls.append((url,kwargs));return Response()
    monkeypatch.setattr(requests,'post',post)
    w=NS(config=NS(host='https://workspace.invalid',authenticate=lambda:{'Authorization':'test'}))
    meter=UsageMeter();token=_meter.set(meter)
    try:
        result=ExistingResponsesWorker(w,'knowledge','platform_docs').invoke({'messages':[HumanMessage(content='Question'),HumanMessage(content='Delegated task')]})
    finally:_meter.reset(token)
    assert result['messages'][0].content=='Existing worker answer'
    assert len(calls)==1 and calls[0][1]['timeout']==(20,240)
    assert calls[0][1]['json']=={'input':[{'role':'user','content':'Question'},{'role':'user','content':'Delegated task'}]}
    summary=meter.summary()
    assert summary['total_tokens']==0 and summary['remote_workers'][0]['usage']['total_tokens']==120
    assert summary['cost_status']=='unavailable'
    assert 'remote_worker:platform_docs' in summary['unmetered_services']


def test_remote_failure_is_not_retried(monkeypatch):
    import requests
    calls=[]
    def fail(*args,**kwargs):calls.append(1);raise requests.ReadTimeout('timeout')
    monkeypatch.setattr(requests,'post',fail)
    w=NS(config=NS(host='https://workspace.invalid',authenticate=lambda:{}))
    with pytest.raises(requests.ReadTimeout):
        ExistingResponsesWorker(w,'knowledge','platform_docs').invoke({'messages':[HumanMessage(content='Question')]})
    assert len(calls)==1


def test_existing_genie_receives_question_and_supervisor_task(monkeypatch):
    worker=ExistingGenieWorker(None,'existing-space');seen=[]
    def ask(question):seen.append(question);return {'answer':'Estimate','presentation':{'tables':[]}}
    monkeypatch.setattr(worker.service,'ask',ask)
    result=worker.invoke({'messages':[HumanMessage(content='Monthly batch'),HumanMessage(content='Ask for missing inputs')]})
    assert seen==['Monthly batch\n\nAsk for missing inputs']
    assert result['genie_result']=={'tables':[]}


def test_supervisor_needs_worker_binding_but_not_lakebase_or_vector_search(tmp_path):
    path=tmp_path/'prompts.json';path.write_text('{}')
    Settings(platform_endpoint='existing-knowledge',genie_space_id='existing-space',prompt_manifest=str(path)).validate()
    with pytest.raises(ValueError,match='endpoint names'):
        Settings(platform_endpoint='https://untrusted.invalid',genie_space_id='s',prompt_manifest=str(path)).validate()
