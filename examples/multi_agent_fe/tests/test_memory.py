import hashlib
import json
import time
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from fe_multi_agent.memory import sign_context, verify_context, MemoryStore, MemoryWorkspace
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
    def files(self,query):self.calls.append(('list',query));return [{'path':'/memories/profile.md','revision':1,'preview':'Use eu-west-1'}]
    def write_file(self,*args):self.calls.append(('write',args));return {'written':True,'revision':1}
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
    model=Model([('list_memory_files',{'query':''}),('delegate_to_worker',{'agent':'cost_estimation','task':'Use remembered region eu-west-1; verify rates'}),('write_memory_file',{'path':'/memories/cost-patterns/batch.md','content':'# PROPOSED\nUse workload quantities and verified rates.','expected_revision':0}),('finish_response',{'answer':'Please provide batch size.'})])
    graph=build_memory_graph(model,{'cost_estimation':worker,'platform_docs':worker},'Use tools',memory,{'summary':'Earlier batch context'})
    result=graph.invoke({'messages':[HumanMessage(content='Estimate batch costs')]})
    assert result['messages'][-1].content=='Please provide batch size.'
    assert [c[0] for c in memory.calls]==['list','write']
    assert 'eu-west-1' in worker.calls[0]['messages'][-1].content
    assert model.bindings[0][0]==['list_memory_files']
    assert model.bindings[0][1]['tool_choice']=='list_memory_files'
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
    memory.files=lambda query:(_ for _ in ()).throw(RuntimeError('database down'))
    model=Model([('list_memory_files',{'query':''}),('finish_response',{'answer':'Please provide the region.'})])
    worker=Worker()
    result=build_memory_graph(model,{'cost_estimation':worker,'platform_docs':worker},'',memory,{}).invoke({'messages':[HumanMessage(content='Cost')]})
    assert 'Memory was unavailable' in result['messages'][-1].content


def test_workspace_rejects_unsafe_paths_and_sensitive_content_before_database_access():
    def no_db():raise AssertionError('Invalid writes must not reach the database')
    workspace=MemoryWorkspace(no_db,'alice','conversation')
    for path in ['/etc/profile.md','/memories/../bob.md','/memories/a/b.txt']:
        with pytest.raises(ValueError):workspace.read_file(path)
    assert not workspace.write_file('/memories/profile.md','password = example-credential',0,'','')['written']


@pytest.mark.parametrize('tool_name,worker_id', [('cost_estimation','cost_estimation'),('platform_knowledge','platform_docs')])
def test_named_worker_tool_routes_and_preserves_results(tool_name,worker_id):
    memory=Memory();worker=Worker()
    class ResultWorker(Worker):
        def invoke(self,state):
            result=super().invoke(state)
            result.update(genie_result={'conversation_id':'genie-test'},source_urls=['https://docs.databricks.com/'])
            return result
    worker=ResultWorker()
    model=Model([('list_memory_files',{}),(tool_name,{'task':'Use current workload details'}),
        ('write_memory_file',{'path':'/memories/workloads/batch.md','content':'# Workload facts','expected_revision':0}),
        ('finish_response',{'answer':'Done'})])
    result=build_memory_graph(model,{worker_id:worker},'',memory,{}).invoke({'messages':[HumanMessage(content='Workload question')]})
    assert len(worker.calls)==1
    assert result['completed']==[worker_id]
    assert {'cost_estimation','platform_knowledge'}<=set(model.bindings[1][0])
    assert 'delegate_to_worker' not in model.bindings[1][0]
    if worker_id=='cost_estimation':
        assert result['genie_result']['conversation_id']=='genie-test'
        assert memory.calls[-1][1][-1]=='Use workload quantities and verified rates.'
    else:
        assert 'https://docs.databricks.com/' in result['messages'][-1].content


@pytest.mark.parametrize('bad_call', [('cost_estimation',{}),('cost_estimation',{'task':' '}),('invented_tool',{})])
def test_invalid_tool_call_can_be_corrected_without_invoking_worker(bad_call):
    worker=Worker()
    model=Model([('list_memory_files',{}),bad_call,('cost_estimation',{'task':'Estimate workload costs'}),
        ('cost_estimation',{'task':'Duplicate request'}),('finish_response',{'answer':'Done'})])
    result=build_memory_graph(model,{'cost_estimation':worker},'',Memory(),{}).invoke({'messages':[HumanMessage(content='Cost')]})
    assert result['messages'][-1].content=='Done'
    assert len(worker.calls)==1
    errors=[json.loads(m.content)['error'] for m in result['messages'] if m.type=='tool' and 'error' in json.loads(m.content)]
    assert len(errors)==2


def test_parallel_worker_batch_is_corrected_without_executing_either_call():
    cost=Worker();platform=Worker()
    class ParallelModel(Model):
        def invoke(self,messages):
            if len(self.prompts)==1:
                self.prompts.append(messages)
                return AIMessage(content='',tool_calls=[
                    {'id':'cost-call','name':'cost_estimation','args':{'task':'Cost inputs'}},
                    {'id':'docs-call','name':'platform_knowledge','args':{'task':'Serverless jobs'}}])
            return super().invoke(messages)
    model=ParallelModel([('list_memory_files',{}),('cost_estimation',{'task':'Cost inputs'}),
        ('platform_knowledge',{'task':'Serverless jobs'}),('finish_response',{'answer':'Done'})])
    result=build_memory_graph(model,{'cost_estimation':cost,'platform_docs':platform},'',Memory(),{}).invoke({'messages':[HumanMessage(content='Cost and docs')]})
    assert len(cost.calls)==len(platform.calls)==1
    errors=[m for m in result['messages'] if m.type=='tool' and m.tool_call_id in {'cost-call','docs-call'}]
    assert len(errors)==2 and all('No calls in this batch were executed' in m.content for m in errors)
