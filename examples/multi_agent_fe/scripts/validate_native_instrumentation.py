# Databricks notebook source
"""FEVM native-tracing toggle and handled-failure checks without team span hooks.

Mock transports keep failure injection isolated from deployed customer endpoints.
The actual endpoint checks are a separate notebook run after pipeline rollout.
"""
import json
import sys
import time
from pathlib import Path

import httpx
import mlflow
from openai import OpenAI
from langchain_core.runnables import RunnableLambda

for key, default in [('source_root', ''), ('experiment_name', '/Shared/agentops-native-instrumentation-validation')]:
    dbutils.widgets.text(key, default)
sys.path.insert(0, str(Path(dbutils.widgets.get('source_root')) / 'src'))
from framework.observability.tracing import configure_native_instrumentation

mlflow.set_tracking_uri('databricks')
experiment = mlflow.set_experiment(dbutils.widgets.get('experiment_name'))
from framework.observability.tracing import use_experiment_destination
use_experiment_destination(experiment.experiment_id)
report = {'experiment_id': experiment.experiment_id, 'scenarios': []}


def reply(request):
    if b'controlled-failure' in request.content:
        return httpx.Response(400, json={'error': {'message': 'controlled validation failure',
            'type': 'invalid_request_error', 'code': 'validation'}})
    return httpx.Response(200, json={'id': 'validation', 'object': 'chat.completion',
        'created': 1, 'model': 'databricks-gpt-oss-120b', 'choices': [{'index': 0,
        'message': {'role': 'assistant', 'content': 'validation response'}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}})


client = OpenAI(api_key='mock-validation-only', base_url='https://validation.invalid/v1',
    http_client=httpx.Client(transport=httpx.MockTransport(reply)), max_retries=0)


def operation(value):
    if value == 'controlled-failure':
        raise RuntimeError('controlled validation failure')
    return 'validation response'


for integration in ['openai', 'langchain']:
    for enabled in [True, False]:
        configure_native_instrumentation({'tracing': {'instrumentation': {
            'enabled': enabled, 'integration': integration}}})
        for fail in [False, True]:
            question = 'controlled-failure' if fail else 'success'
            # Only the outer middleware-style span is explicit. The target calls
            # below have no decorators, callbacks, or observer.agent() wrappers.
            with mlflow.start_span(name='agentops.validation.request', span_type='AGENT') as root:
                try:
                    if integration == 'openai':
                        client.chat.completions.create(model='databricks-gpt-oss-120b',
                            messages=[{'role': 'user', 'content': question}])
                    else:
                        RunnableLambda(operation).invoke(question)
                    assert not fail
                except (RuntimeError, __import__('openai').BadRequestError):
                    assert fail
                trace_id = root.trace_id
            mlflow.flush_trace_async_logging()
            trace = mlflow.get_trace(trace_id)
            assert trace is not None
            children = [s for s in trace.data.spans if s.parent_id is not None]
            assert bool(children) == enabled, (integration, enabled, children)
            errors = [s for s in children if s.status.status_code.value == 'ERROR']
            assert bool(errors) == (enabled and fail)
            # The caller handled the failure; the overall request remains OK.
            assert str(trace.info.state) == 'OK'
            if enabled and integration == 'openai' and not fail:
                assert trace.info.token_usage['total_tokens'] == 15
            item = {'integration': integration, 'enabled': enabled, 'failure_injected': fail,
                    'trace_id': trace_id, 'child_spans': len(children), 'error_spans': len(errors),
                    'status': 'passed'}
            report['scenarios'].append(item)
            print(json.dumps(item))
        configure_native_instrumentation({'tracing': {'instrumentation': {
            'enabled': False, 'integration': integration}}})
print(json.dumps(report, indent=2))
dbutils.notebook.exit(json.dumps(report))
