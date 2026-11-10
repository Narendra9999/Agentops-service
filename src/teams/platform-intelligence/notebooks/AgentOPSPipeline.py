# Databricks notebook source
import json
import os
import sys
from pathlib import Path
import mlflow
from databricks.sdk import WorkspaceClient

for key in ['project_root','stage']:
    dbutils.widgets.text(key,'')
root=Path(dbutils.widgets.get('project_root'))
stage=dbutils.widgets.get('stage')
sys.path.insert(0,str(root/'src'))
sys.path.insert(0,str(root/'src/teams/platform-intelligence/agent'))
from fe_multi_agent import pipeline

os.environ['AGENT_CONTEXT_SIGNING_KEY']=dbutils.secrets.get('fe-agentops-context','signing-key')
os.environ['AGENTOPS_ENVIRONMENT']='fevm'
mlflow.set_tracking_uri('databricks')
mlflow.set_registry_uri('databricks-uc')
mlflow.set_experiment(experiment_id=pipeline.EXPERIMENT)
w=WorkspaceClient()
if stage=='preflight':
    from fe_multi_agent.config import Settings
    _,config=pipeline.configuration(root)
    Settings(**config['multi_agent']['settings']).validate()
    endpoint=config['multi_agent']['settings']['memory_endpoint']
    assert w.postgres.get_endpoint(name=endpoint).status.hosts.host
    result={'lakebase_endpoint':endpoint,'team':'platform-intelligence','workers_deployed':0}
elif stage=='register':
    result=pipeline.register(root)
    for key in ['model_uri','version']:dbutils.jobs.taskValues.set(key=key,value=result[key])
elif stage in ['pre','post']:
    uri=dbutils.jobs.taskValues.get(taskKey='register_model',key='model_uri')
    result=pipeline.evaluate(root,uri,stage)
elif stage=='deploy':
    version=dbutils.jobs.taskValues.get(taskKey='register_model',key='version')
    result=pipeline.deploy(version)
elif stage=='smoke':result=pipeline.smoke()
else:raise ValueError('Unknown pipeline stage')
mlflow.flush_trace_async_logging()
dbutils.notebook.exit(json.dumps(result,default=str))
