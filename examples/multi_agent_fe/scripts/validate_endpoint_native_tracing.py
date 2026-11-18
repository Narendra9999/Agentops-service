# Databricks notebook source
"""Check real serving contracts and native accounting for both agent patterns."""
import json
import time
import uuid
import requests
import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.types.responses import ResponsesAgentResponse, ResponseOutputItemDoneEvent
from mlflow.exceptions import RestException

for key, default in [('endpoints', '{}'), ('require_native_cost', 'true')]:
    dbutils.widgets.text(key, default)
endpoints = json.loads(dbutils.widgets.get('endpoints'))
require_cost = dbutils.widgets.get('require_native_cost') == 'true'
workspace = WorkspaceClient()
client = mlflow.MlflowClient(tracking_uri='databricks')
report = {'scenarios': []}
for endpoint_name, version in endpoints.items():
    for readiness_attempt in range(30):
        endpoint = workspace.serving_endpoints.get(endpoint_name)
        entities = {entity.name: str(entity.entity_version) for entity in endpoint.config.served_entities}
        live = {entities[getattr(route, 'served_entity_name', None) or route.served_model_name]
                for route in endpoint.config.traffic_config.routes if route.traffic_percentage > 0}
        if (endpoint.state.ready.value == 'READY'
                and endpoint.state.config_update.value == 'NOT_UPDATING'
                and live == {str(version)}):
            break
        time.sleep(10)
    else:
        raise AssertionError(('Expected version not stable', endpoint_name, version,
                              endpoint.state.as_dict(), sorted(live)))
    scenarios = [('platform_prediction', False,
        'Explain MLflow tracing for LangGraph using a retrieved Databricks documentation citation.'),
        ('platform_stream', True,
        'Explain Unity Catalog access controls using a retrieved Databricks documentation citation.')]
    if endpoint_name == 'fe-platform-intelligence':
        scenarios.append(('cost_stream', True,
            'Ask Cost intelligence for the missing inputs to estimate a scheduled batch '
            'that runs 30 times a month. Runtime is unknown. Do not invent rates.'))
    for name, streaming, question in scenarios:
        custom = {'thread_id': 'native-toggle-validation-' + str(uuid.uuid4())}
        payload = {'input': [{'role': 'user', 'content': question}],
                   'custom_inputs': custom, 'stream': streaming}
        items, metadata, event_types = [], {}, []
        url = workspace.config.host.rstrip('/') + '/serving-endpoints/' + endpoint_name + '/invocations'
        with requests.post(url, headers=workspace.config.authenticate(), json=payload,
                           stream=streaming, timeout=(20, 300)) as response:
            response.raise_for_status()
            if not streaming:
                body = response.json()
                ResponsesAgentResponse.model_validate(body)
                items = body['output']; metadata = body.get('custom_outputs') or {}
            else:
                for raw in response.iter_lines():
                    if not raw or not raw.startswith(b'data:'):
                        continue
                    raw = raw[5:].strip()
                    if raw == b'[DONE]':
                        break
                    event = json.loads(raw)
                    kind = event.get('type'); event_types.append(kind)
                    assert kind not in {'error', 'response.failed'} and not event.get('error'), event
                    if kind == 'response.output_item.done':
                        ResponseOutputItemDoneEvent.model_validate(event)
                        items.append(event['item'])
                        metadata.update(event.get('custom_outputs') or {})
                assert event_types and set(event_types) == {'response.output_item.done'}, event_types
        text = '\n'.join(p.get('text', '') for item in items for p in item.get('content', []))
        assert text.strip()
        if name.startswith('platform'):
            assert 'docs.databricks.com' in text, text
        assert metadata['thread_id'] == custom['thread_id']
        trace_id = metadata['trace_id']
        trace = None
        for attempt in range(20):
            try:
                trace = client.get_trace(trace_id)
            except RestException as exc:
                if exc.error_code not in {'NOT_FOUND', 'RESOURCE_DOES_NOT_EXIST'}:
                    raise
            if trace is not None:
                break
            time.sleep(3)
        assert trace and str(trace.info.state) == 'OK'
        spans = trace.data.spans; tags = trace.info.tags
        guards = {key: sum(s.name == key for s in spans) for key in ['pre_llm_check', 'post_llm_check']}
        assert guards == {'pre_llm_check': 1, 'post_llm_check': 1}, guards
        assert not any('agentops.invocation_id' in s.attributes for s in spans)
        assert not any(s.name == 'workflow.execution' for s in spans)
        models = [s for s in spans if s.span_type in {'LLM', 'CHAT_MODEL'}]
        usage = [s for s in spans if s.attributes.get('mlflow.chat.tokenUsage')]
        assert models and usage
        assert all(s.span_type in {'LLM', 'CHAT_MODEL'} for s in usage)
        assert trace.info.token_usage['total_tokens'] == int(tags['agentops.tokens.total'])
        native_cost = (trace.info.trace_metadata or {}).get('mlflow.trace.cost')
        native_cost = json.loads(native_cost) if isinstance(native_cost, str) else native_cost
        if require_cost:
            assert native_cost is not None, ('missing native cost', endpoint_name, trace_id)
            model_cost = sum((s.attributes.get('mlflow.llm.cost') or {}).get('total_cost', 0)
                             for s in models)
            assert abs(native_cost['total_cost'] - model_cost) < 1e-9
            assert float(tags['agentops.cost_usd']) >= 0
        if endpoint_name == 'fe-platform-single-validation':
            assert len(models) == 1, [(s.name, s.span_type) for s in models]
            assert models[0].name != 'chat_completions.model'
        else:
            expected = 'cost_estimation' if name.startswith('cost') else 'platform_docs'
            assert expected in {s.name for s in spans}
        result = {'endpoint': endpoint_name, 'version': str(version), 'scenario': name,
                  'trace_id': trace_id, 'native_model_spans': len(models), 'guardrail_spans': guards,
                  'tokens': trace.info.token_usage['total_tokens'],
                  'agentops_cost_usd': tags.get('agentops.cost_usd'), 'native_cost': native_cost,
                  'cost_estimates_match': (abs(native_cost['total_cost'] - float(tags['agentops.cost_usd'])) < 1e-6)
                      if native_cost is not None else None,
                  'status': 'passed'}
        report['scenarios'].append(result)
        print(json.dumps(result))
print(json.dumps(report, indent=2))
dbutils.notebook.exit(json.dumps(report))
