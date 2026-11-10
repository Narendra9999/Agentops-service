import hashlib
import json
import time
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from fe_multi_agent.memory import sign_context, verify_context, MemoryStore
from fe_multi_agent.memory_graph import build_memory_graph


def test_signed_context_binds_owner_expiry_and_exact_messages():
    messages=[{'role':'user','content':'Estimate my workload'}]
    data={'owner_id':'alice','expires_at':int(time.time())+300,
          'messages_sha256':hashlib.sha256(json.dumps(messages,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
    envelope=sign_context(data,'test-key')
    assert verify_context(envelope,messages,'test-key')['owner_id']=='alice'
    with pytest.raises(ValueError):verify_context(envelope,messages,'wrong-key')
    with pytest.raises(ValueError):verify_context(envelope,[{'role':'user','content':'Changed'}],'test-key')
    data['expires_at']=0
    with pytest.raises(ValueError):verify_context(sign_context(data,'test-key'),messages,'test-key')


class Memory:
    def __init__(self):self.calls=[]
    def search(self,query):self.calls.append(('search',query));return [{'id':'m','content':'Use eu-west-1','status':'active'}]
    def store(self,*args):self.calls.append(('store',args));return {'stored':True,'status':'pending'}
    def change(self,*args):self.calls.append(('change',args));return {'changed':False}

class Model:
    def __init__(self,calls):self.calls=iter(calls);self.prompts=[];self.bindings=[]
    def bind_tools(self,tools,**kwargs):self.bindings.append(([t.name for t in tools],kwargs));return self
    def invoke(self,messages):self.prompts.append(messages);name,args=next(self.calls);return AIMessage(content='',tool_calls=[{'id':'call','name':name,'args':args}])

class Worker:
    def __init__(self):self.calls=[]
    def invoke(self,state):self.calls.append(state);return {'messages':[AIMessage(content='Use workload quantities and verified rates.')]}


def test_model_tools_enforce_search_before_delegation_and_preserve_context():
    memory=Memory();worker=Worker()
    model=Model([('search_memory',{'query':'batch costs'}),('delegate_to_worker',{'agent':'cost_estimation','task':'Use remembered region eu-west-1; verify rates'}),('store_memory',{'kind':'cost_pattern','content':'Use workload quantities and verified rates.','evidence':'Use workload quantities and verified rates.'}),('finish_response',{'answer':'Please provide batch size.'})])
    graph=build_memory_graph(model,{'cost_estimation':worker,'platform_docs':worker},'Use tools',memory,{'summary':'Earlier batch context'})
    result=graph.invoke({'messages':[HumanMessage(content='Estimate batch costs')]})
    assert result['messages'][-1].content=='Please provide batch size.'
    assert [c[0] for c in memory.calls]==['search','store']
    assert 'eu-west-1' in worker.calls[0]['messages'][-1].content
    assert model.bindings[0][0]==['search_memory']
    assert model.bindings[0][1]['tool_choice']=='search_memory'
    assert 'Earlier batch context' in str(model.prompts[0])


def test_model_cannot_skip_initial_memory_search():
    worker=Worker()
    graph=build_memory_graph(Model([('delegate_to_worker',{'agent':'cost_estimation','task':'skip'})]),{'cost_estimation':worker,'platform_docs':worker},'',Memory(),{})
    with pytest.raises(ValueError,match='first supervisor tool'):graph.invoke({'messages':[HumanMessage(content='Cost')]})
    assert not worker.calls


def test_pending_patterns_require_exact_user_approval_and_no_db_for_invalid_evidence():
    def no_db():raise AssertionError('Database should not be called')
    store=MemoryStore(no_db,'alice','conversation')
    assert not store.store('user_fact','invented fact','evidence','Use eu-west-1','')['stored']
    assert not store.store('cost_pattern','invented pattern','evidence','Estimate','')['stored']
    assert not store.change('00000000-0000-0000-0000-000000000001','approve','approve all patterns')['changed']


def test_memory_unavailable_is_disclosed_and_no_fabricated_recall():
    memory=Memory()
    memory.search=lambda query:(_ for _ in ()).throw(RuntimeError('database down'))
    model=Model([('search_memory',{'query':'cost'}),('finish_response',{'answer':'Please provide the region.'})])
    worker=Worker()
    result=build_memory_graph(model,{'cost_estimation':worker,'platform_docs':worker},'',memory,{}).invoke({'messages':[HumanMessage(content='Cost')]})
    assert 'Memory was unavailable' in result['messages'][-1].content
