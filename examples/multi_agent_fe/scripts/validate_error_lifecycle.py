# Databricks notebook source
"""Controlled fault validation on serverless compute; no external side effects."""
import asyncio
import json
import os
import sys
from pathlib import Path

# The pipeline syncs the repository at this workspace location.
dbutils.widgets.text('source_root', '')
source_root = dbutils.widgets.get('source_root')
assert source_root
sys.path.insert(0, str(Path(source_root) / 'src'))
os.environ['AGENTOPS_ENVIRONMENT'] = 'test'
import mlflow
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse, ResponsesAgentStreamEvent
from framework.inference.agent_middleware import AgentOPSMiddleware
from framework.inference.request_context import RequestContext

mlflow.set_tracking_uri('databricks')
mlflow.set_experiment('/Shared/agentops-error-lifecycle-validation')
results = []

async def validate(kind):
    context = RequestContext()
    middleware = AgentOPSMiddleware({'guardrails': {'enabled': False},
        'middleware': {'timeout_seconds': .1}})
    calls = []
    async def invoke(request, context):
        calls.append(True)
        if kind == 'timeout':
            await asyncio.sleep(2)
        if kind == 'cancelled':
            raise asyncio.CancelledError()
        if kind == 'failure':
            raise RuntimeError('controlled validation failure')
        return ResponsesAgentResponse(output=[middleware.create_text_output_item(text='ok', id='answer')])
    async def stream(request, context):
        calls.append(True)
        yield ResponsesAgentStreamEvent(type='response.output_text.delta', item_id='answer', delta='unchecked')
        raise RuntimeError('controlled stream failure')
    request = ResponsesAgentRequest(input=[{'role': 'user', 'content': 'Validation request'}])
    with mlflow.start_span(name='validation.' + kind) as span:
        delivered = []
        try:
            if kind == 'stream_failure':
                async for event in middleware.execute_stream(request, invoke=invoke, stream=stream, context=context):
                    delivered.append(event)
            else:
                await middleware.execute(request, invoke=invoke, context=context)
        except (RuntimeError, TimeoutError, asyncio.CancelledError):
            assert kind != 'success'
            span.set_status('ERROR')
        else:
            assert kind == 'success'
        expected = 'success' if kind == 'success' else 'cancelled' if kind == 'cancelled' else 'failed'
        assert context.state['execution_outcome'] == expected
        assert context.state['execution_finalized']
        assert context.resources is None
        assert len(calls) == 1
        assert not delivered
        results.append({'scenario': kind, 'outcome': expected, 'trace_id': span.trace_id})

for kind in ['success', 'failure', 'timeout', 'cancelled', 'stream_failure']:
    await validate(kind)
mlflow.flush_trace_async_logging()
for row in results:
    trace = mlflow.get_trace(row['trace_id'])
    assert trace.info.tags['agentops.status'] == row['outcome'], trace.info.tags
    if row['outcome'] == 'failed':
        assert trace.info.tags.get('agentops.error.code')
print(json.dumps(results, indent=2))
dbutils.notebook.exit(json.dumps(results))
