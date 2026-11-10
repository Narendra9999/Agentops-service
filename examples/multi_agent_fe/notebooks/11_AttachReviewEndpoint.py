# Databricks notebook source
import mlflow
from mlflow.genai import get_review_app

mlflow.set_tracking_uri('databricks')
mlflow.set_experiment(experiment_id='4384417075366092')
review = get_review_app(experiment_id='4384417075366092').add_agent(
    agent_name='platform_engineering',
    model_serving_endpoint='fe-platform-agent-team', overwrite=True)
dbutils.notebook.exit(json.dumps({'endpoint_attached': True,
                                'endpoint': 'fe-platform-agent-team'}))
