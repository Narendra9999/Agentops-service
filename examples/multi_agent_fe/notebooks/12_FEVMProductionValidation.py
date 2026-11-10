# Databricks notebook source
import json
import mlflow
from mlflow.genai import get_labeling_sessions
from fe_multi_agent.operations import daily_monitor

mlflow.set_tracking_uri('databricks')
mlflow.set_experiment(experiment_id='4384417075366092')
traces = mlflow.search_traces(experiment_ids=['4384417075366092'], return_type='list',
    filter_string="tags.`agentops.system_id` = 'fe-platform-engineering'", max_results=100)
production = [trace for trace in traces if not trace.info.tags.get('fe.eval_phase')
              and not trace.info.tags.get('mlflow.eval.requestId')]
if not production:
    raise RuntimeError('No persisted serving traces; do not activate the monitor')
sessions = [session for session in get_labeling_sessions()
            if session.name == 'platform-team-initial-review']
if len(sessions) != 1:
    raise RuntimeError('Expected one existing initial review session; no duplicate will be created')
marker = 'fe.review.initial_session'
pending = [trace for trace in production if trace.info.tags.get(marker) != 'added']
if pending:
    sessions[0].add_traces(traces=pending)
    for trace in pending:
        mlflow.set_trace_tag(trace_id=trace.info.trace_id, key=marker, value='added')
report = daily_monitor('4384417075366092', 'databricks-gpt-oss-120b')
dbutils.notebook.exit(json.dumps({'production_traces': len(production),
    'review_traces_added': len(pending), 'review_url': sessions[0].url,
    'monitor': report}, default=str))
