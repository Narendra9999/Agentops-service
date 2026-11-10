# Databricks notebook source
# Compatibility validation only: memory is an in-process stub. No database writes,
# permission grants, secret creation, model registration or endpoint update.
import json
import subprocess
import os
import mlflow
from databricks.sdk import WorkspaceClient
from langchain_core.messages import HumanMessage
from fe_multi_agent.config import Settings
from fe_multi_agent.services import chat_model
from fe_multi_agent.memory_graph import build_memory_graph
from fe_multi_agent.platform_worker import PlatformKnowledgeWorker
from fe_multi_agent.prompts import TEMPLATES
from fe_multi_agent.progress import progress_sink

mlflow.set_tracking_uri('databricks')
mlflow.set_experiment(experiment_id='4384417075366092')
mlflow.langchain.autolog()
workspace=WorkspaceClient()
settings=Settings(genie_space_id='01f1d6efcecd1620b04cf487817675b7',platform_mode='in_process',docs_index='fevm_catalog_naren.agentops.databricks_docs_index',
                  prompt_manifest=str(root/'team/configs/prompts.json'),max_steps=60)
settings.validate()
class StubMemory:
 def __init__(self):self.rows={};self.calls=[]
 def files(self,query=''):
  self.calls.append('list_memory_files')
  return [{'path':p,'revision':v['revision'],'preview':v['content'][:240]} for p,v in self.rows.items()]
 def read_file(self,path):
  self.calls.append('read_memory_file');return self.rows.get(path,{'not_found':True})
 def write_file(self,path,content,expected_revision,*args):
  self.calls.append('write_memory_file')
  revision=self.rows.get(path,{}).get('revision',0)
  if revision!=expected_revision:return {'written':False,'reason':'Revision conflict'}
  self.rows[path]={'content':content,'revision':revision+1}
  return {'written':True,'path':path,'revision':revision+1}
 def delete_file(self,*args):raise AssertionError('No deletion requested')
 def history(self,*args):return []
class NoCost:
 def invoke(self,state):raise AssertionError('Cost worker not needed for memory-only and platform tests')

memory=StubMemory()
model=chat_model(settings,workspace,settings.supervisor_model)
def turn(question,platform):
 events=[]
 with progress_sink(events.append):
  graph=build_memory_graph(model,{'cost_estimation':NoCost(),'platform_docs':platform},TEMPLATES['supervisor'],memory,{'summary':''})
  result=graph.invoke({'messages':[HumanMessage(content=question)]},config={'recursion_limit':60})
 return result,events

with mlflow.start_span(name='validation.memory_embedded_platform') as span:
 platform=PlatformKnowledgeWorker(settings,workspace)
 result,events=turn('Remember this workload fact exactly: Our scheduled batch runs in eu-west-1. Only save the fact; do not estimate costs or invoke a specialist.',platform)
 assert memory.calls[0]=='list_memory_files' and 'write_memory_file' in memory.calls,memory.calls
 assert memory.rows and any('eu-west-1' in v['content'] for v in memory.rows.values()),memory.rows
 result,events=turn('Which region does our scheduled batch use? Answer from saved memory only; do not invoke a specialist.',platform)
 assert 'eu-west-1' in result['messages'][-1].content,result
 result,events=turn('Explain MLflow tracing for LangGraph using retrieved Databricks documentation. Do not estimate costs.',platform)
 assert result.get('source_urls'),result
 assert any(e.get('tool')=='retrieve_platform_docs' for e in events),events
 report={'model_memory_tools':True,'new_graph_recall_with_stub_memory':True,'embedded_platform_retrieval':True,
         'worker_endpoints_deployed':0,'persistent_memory_writes':0,'secret_changes':0,'trace_id':span.trace_id}
mlflow.flush_trace_async_logging()
dbutils.notebook.exit(json.dumps(report))
