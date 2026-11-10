# Databricks notebook source
# APPROVAL GATE: run after memory schema, signing secret and grants are approved.
# Registers a candidate only; does not change endpoint traffic or deploy the App.
import json
import os
import yaml
from fe_multi_agent.config import Settings
from fe_multi_agent.deployment import register_model

os.environ['AGENT_CONTEXT_SIGNING_KEY']=dbutils.secrets.get('fe-agentops-context','signing-key')
config=yaml.safe_load((root/'team/config.yaml').read_text())
values=config['multi_agent']['settings']
values.update(experiment_id='4384417075366092',prompt_manifest=str(root/'team/configs/prompts.json'))
settings=Settings(**values)
report=register_model(settings,'fevm_catalog_naren.agentops_multi.platform_team',root,root/'runtime.json',root/'requirements.lock')
dbutils.notebook.exit(json.dumps(report))
