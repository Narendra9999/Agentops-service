# Databricks notebook source
import mlflow
from mlflow.genai import get_review_app,create_labeling_session,label_schemas
mlflow.set_tracking_uri('databricks')
mlflow.set_experiment(experiment_id='4384417075366092')
label_schemas.create_label_schema(name='team_quality',type='feedback',title='Agent team quality',
    input=label_schemas.InputNumeric(min_value=1,max_value=5),enable_comment=True,
    instruction='Assess routing, source-supported answers and truthful audited mutation outcomes.',overwrite=True)
review=get_review_app(experiment_id='4384417075366092')
session=create_labeling_session(name='platform-team-initial-review',label_schemas=['team_quality'])
traces=mlflow.search_traces(experiment_ids=['4384417075366092'],return_type='list',
    filter_string="tags.`fe.eval_phase` = 'after'",max_results=3)
if traces:session=session.add_traces(traces=traces)
dbutils.notebook.exit(json.dumps({'labeling_url':session.url,'traces_added':len(traces),
                                'endpoint_attached':False}))
