# Databricks notebook source
"""Live contract/routing/accounting and signed cross-session memory validation.

Run on FEVM serverless compute with the original team dependency environment.
Only isolated evaluation-owned memory files are written and soft-deleted.
"""
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import mlflow
import requests
from databricks.sdk import WorkspaceClient
from mlflow import MlflowClient
from mlflow.types.responses import ResponseOutputItemDoneEvent, ResponsesAgentResponse

for key, default in [('source_root', ''), ('endpoint', 'fe-platform-intelligence'), ('version', '')]:
    dbutils.widgets.text(key, default)
source_root = Path(dbutils.widgets.get('source_root'))
endpoint_name = dbutils.widgets.get('endpoint')
version = dbutils.widgets.get('version')
assert version and str(source_root) != '.'
sys.path.insert(0, str(source_root / 'src'))
sys.path.insert(0, str(source_root / 'src/teams/platform-intelligence/agent'))
from fe_multi_agent.memory import sign_context

workspace = WorkspaceClient()
client = MlflowClient(tracking_uri='databricks', registry_uri='databricks-uc')
endpoint = workspace.serving_endpoints.get(endpoint_name)
versions = {entity.name: str(entity.entity_version) for entity in endpoint.config.served_entities}
live = {versions[getattr(route, 'served_entity_name', None) or route.served_model_name]
        for route in endpoint.config.traffic_config.routes if route.traffic_percentage > 0}
assert endpoint.state.ready.value == 'READY'
assert endpoint.state.config_update.value == 'NOT_UPDATING'
assert live == {version}, live
report = {'endpoint': endpoint_name, 'version': version, 'scenarios': {}}
url = workspace.config.host.rstrip('/') + '/serving-endpoints/' + endpoint_name + '/invocations'


def get_trace(trace_id):
    for attempt in range(10):
        trace = client.get_trace(trace_id)
        if trace is not None:
            return trace
        time.sleep(2)
    raise AssertionError('Trace not exported: ' + trace_id)


def invoke(name, question, expected_workers=(), context=None, streaming=True):
    custom = {'thread_id': 'reusable-validation-' + str(uuid.uuid4())}
    if context is not None:
        messages = [{'role': 'user', 'content': question}]
        context = {**context, 'conversation_id': str(uuid.uuid4()),
                   'expires_at': int(time.time()) + 300,
                   'messages_sha256': hashlib.sha256(json.dumps(
                       messages, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
        custom['application_context'] = sign_context(context, signing_key)
        custom['stream_activity'] = True
    payload = {'input': [{'role': 'user', 'content': question}], 'custom_inputs': custom, 'stream': streaming}
    items, metadata, activity, event_types = [], {}, [], []
    started = time.monotonic()
    with requests.post(url, headers=workspace.config.authenticate(), json=payload,
                       stream=streaming, timeout=(20, 240)) as response:
        response.raise_for_status()
        if not streaming:
            body = response.json()
            ResponsesAgentResponse.model_validate(body)
            items = body['output']
            metadata = body.get('custom_outputs') or {}
        else:
            for line in response.iter_lines():
                if not line or not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    break
                event = json.loads(data)
                kind = event.get('type')
                assert kind not in {'error', 'response.failed'} and not event.get('error'), event
                event_types.append(kind)
                extra = event.get('custom_outputs') or {}
                if extra.get('activity'):
                    activity.append(extra['activity'])
                if kind == 'response.output_item.done':
                    ResponseOutputItemDoneEvent.model_validate(event)
                    items.append(event['item'])
                    metadata.update(extra)
                if kind == 'response.completed':
                    metadata.update((event.get('response') or {}).get('custom_outputs') or {})
            if context is None:
                assert event_types and set(event_types) == {'response.output_item.done'}, event_types
            else:
                assert 'response.completed' in event_types
    text = '\n'.join(part.get('text', '') for item in items for part in item.get('content', []))
    assert text.strip(), name
    assert metadata['thread_id'] == custom['thread_id']
    assert metadata['memory']['mode'] == ('persistent' if context is not None else 'stateless')
    trace_id = metadata['trace_id']
    trace = get_trace(trace_id)
    assert str(trace.info.state) == 'OK', (name, trace.info.state)
    spans = trace.data.spans
    names = {span.name for span in spans}
    workers = (names & {'cost_estimation', 'platform_docs'}) | {
        a['agent'] for a in activity if a.get('kind') == 'agent_started'}
    assert workers == set(expected_workers), (name, workers)
    guard_counts = {key: sum(span.name == key for span in spans)
                    for key in ('pre_llm_check', 'post_llm_check')}
    assert guard_counts == {'pre_llm_check': 1, 'post_llm_check': 1}, guard_counts
    tags = trace.info.tags
    assert int(tags['agentops.tokens.total']) > 0
    assert trace.info.token_usage['total_tokens'] == int(tags['agentops.tokens.total'])
    native_cost = json.loads(trace.info.request_metadata['mlflow.trace.cost'])
    assert abs(native_cost['total_cost'] - float(tags['agentops.cost_usd'])) < 1e-6
    assert all(span.span_type in {'LLM', 'CHAT_MODEL'} for span in spans
               if span.attributes.get('mlflow.chat.tokenUsage'))
    if 'platform_docs' in expected_workers:
        assert 'docs.databricks.com' in text, text
    if context is not None:
        tools = [a.get('tool') for a in activity if a.get('tool')]
        assert tools and tools[0] == 'list_memory_files', tools
        assert 'Memory was unavailable' not in text
    report['scenarios'][name] = {'trace_id': trace_id, 'workers': sorted(workers),
        'memory_mode': metadata['memory']['mode'], 'guard_counts': guard_counts,
        'tokens': trace.info.token_usage['total_tokens'], 'cost_usd': native_cost['total_cost'],
        'event_types': sorted(set(event_types)), 'span_names': sorted(n for n in names if n),
        'tools': [a.get('tool') for a in activity if a.get('tool')],
        'seconds': round(time.monotonic() - started, 1)}
    print(json.dumps({'scenario': name, **report['scenarios'][name]}), flush=True)
    return text, activity


invoke('platform_stream', 'Explain MLflow tracing for LangGraph using retrieved Databricks documentation. Cite the documentation URLs. Do not estimate costs.', ['platform_docs'])
invoke('cost_stream', 'Ask Cost intelligence which inputs are missing to estimate a batch workload running 30 times monthly. Runtime and volume are unknown. Do not invent rates.', ['cost_estimation'])
invoke('combined_prediction', 'Ask Cost intelligence for the missing inputs to estimate a monthly batch workload, and retrieve Databricks documentation about serverless jobs. Cite retrieved documentation and do not invent a quote.', ['cost_estimation', 'platform_docs'], streaming=False)

signing_key = dbutils.secrets.get('fe-agentops-context', 'signing-key')
owner = 'evaluation:reusable-supervisor:' + str(uuid.uuid4())
context = {'owner_id': owner, 'summary': '', 'preferences': {}}
text, activity = invoke('memory_write', 'Remember this workload fact in /memories/workloads/reusable-validation.md: Our validation batch runs in eu-west-1. Save it for future sessions. Do not call a worker or estimate costs.', context=context)
assert any(a.get('tool') == 'write_memory_file' for a in activity)
text, activity = invoke('memory_fresh_session_recall', 'Which region does our validation batch use? Read the saved workload memory and answer. Do not call a worker.', context=context)
assert 'eu-west-1' in text and any(a.get('tool') == 'read_memory_file' for a in activity)
text, activity = invoke('memory_cleanup', 'Read /memories/workloads/reusable-validation.md and delete that test memory file using its current revision. Do not call a worker.', context=context)
assert any(a.get('tool') == 'delete_memory_file' for a in activity)
report['status'] = 'passed'
report['evaluation_owner'] = owner
print(json.dumps(report, indent=2))
dbutils.notebook.exit(json.dumps(report))
