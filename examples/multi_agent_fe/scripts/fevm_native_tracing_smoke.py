"""Verify native supervisor tracing on a deployed observer-disabled model."""
import argparse
import json
import os
from pathlib import Path
import time

import requests
from databricks.sdk import WorkspaceClient
from mlflow import MlflowClient
from mlflow.types.responses import ResponseOutputItemDoneEvent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--output', default='native-tracing-smoke.json')
    args = parser.parse_args()
    os.environ['DATABRICKS_CONFIG_PROFILE'] = args.profile
    workspace = WorkspaceClient(profile=args.profile)
    endpoint = workspace.serving_endpoints.get(args.endpoint)
    entities = {item.name: str(item.entity_version) for item in endpoint.config.served_entities}
    live = {entities[getattr(route, 'served_entity_name', None) or route.served_model_name]
            for route in endpoint.config.traffic_config.routes if route.traffic_percentage > 0}
    if (endpoint.state.ready.value != 'READY'
            or endpoint.state.config_update.value != 'NOT_UPDATING'
            or live != {args.version}):
        raise RuntimeError('Requested model version is not fully ready')
    client = MlflowClient(tracking_uri='databricks://' + args.profile,
                          registry_uri='databricks-uc://' + args.profile)
    report = {'endpoint': args.endpoint, 'version': args.version, 'scenarios': {}}
    scenarios = {
        'cost': ('Ask Cost intelligence for the missing inputs to estimate a scheduled '
                 'batch that runs 30 times a month. Runtime is unknown. Do not invent rates.',
                 {'cost_estimation'}),
        'platform': ('Explain MLflow tracing for LangGraph with a retrieved Databricks '
                     'documentation citation. Do not estimate costs.', {'platform_docs'}),
        'combined': ('Use Cost intelligence to identify the missing workload inputs for a '
                     'monthly batch estimate, then retrieve Databricks documentation about '
                     'serverless jobs. Cite documentation and do not invent a quote.',
                     {'cost_estimation', 'platform_docs'}),
    }
    for name, (question, expected) in scenarios.items():
        items, custom, event_types = [], {}, []
        with requests.post(
            workspace.config.host.rstrip('/') + '/serving-endpoints/' + args.endpoint + '/invocations',
            headers=workspace.config.authenticate(),
            json={'input': [{'role': 'user', 'content': question}], 'stream': True},
            stream=True, timeout=(20, 240),
        ) as response:
            response.raise_for_status()
            for raw in response.iter_lines():
                if not raw or not raw.startswith(b'data:'):
                    continue
                raw = raw[5:].strip()
                if raw == b'[DONE]':
                    break
                event = json.loads(raw)
                if event.get('type') != 'response.output_item.done':
                    raise AssertionError('Unexpected default stream event: ' + str(event.get('type')))
                ResponseOutputItemDoneEvent.model_validate(event)
                items.append(event['item'])
                custom.update(event.get('custom_outputs') or {})
                event_types.append(event['type'])
        text = '\n'.join(part.get('text', '') for item in items for part in item.get('content', []))
        assert text and custom['memory']['mode'] == 'stateless'
        assert custom['tags']['agentops.observer_enabled'] == 'false'
        if 'platform_docs' in expected:
            assert 'docs.databricks.com' in text
        trace_id = custom['trace_id']
        trace = None
        for attempt in range(6):
            trace = client.get_trace(trace_id)
            if trace is not None:
                break
            time.sleep(2)
        assert trace is not None and str(trace.info.state) == 'OK', trace_id
        tags = trace.info.tags
        assert int(tags['agentops.tokens.total']) > 0
        assert float(tags['agentops.cost_usd']) >= 0
        spans = trace.data.spans
        names = {span.name for span in spans if span.name}
        guard_counts = {key: sum(span.name == key for span in spans)
                        for key in ('pre_llm_check', 'post_llm_check')}
        assert guard_counts == {'pre_llm_check': 1, 'post_llm_check': 1}, (name, guard_counts)
        assert all(span.span_type in {'LLM', 'CHAT_MODEL'} for span in spans
                   if span.attributes.get('mlflow.chat.tokenUsage')), 'Parent spans must not repeat model usage'
        workers = names & {'cost_estimation', 'platform_docs'}
        assert workers == expected, (name, workers)
        assert 'supervisor' in names
        assert not names & {'workflow.execution', 'agent.cost_estimation', 'agent.platform_docs'}
        assert not any('agentops.invocation_id' in span.attributes for span in spans)
        usage = [span.attributes['mlflow.chat.tokenUsage'] for span in spans
                 if span.attributes.get('mlflow.chat.tokenUsage')]
        assert usage, 'Native model token usage missing'
        native_total = trace.info.token_usage['total_tokens']
        assert native_total == int(tags['agentops.tokens.total']), (
            'Native usage differs from observed model usage', native_total, tags['agentops.tokens.total'])
        native_cost = json.loads(trace.info.request_metadata['mlflow.trace.cost'])
        assert abs(native_cost['total_cost'] - float(tags['agentops.cost_usd'])) < 1e-6, (
            'Native cost differs from AgentOPS estimate', native_cost, tags['agentops.cost_usd'])
        report['scenarios'][name] = {
            'trace_id': trace_id, 'state': str(trace.info.state), 'workers': sorted(workers),
            'observer_enabled': False, 'middleware_guard_counts': guard_counts, 'span_count': len(spans), 'span_names': sorted(names),
            'unnamed_span_count': sum(not span.name for span in spans),
            'token_usage': usage, 'native_cost': native_cost, 'event_types': event_types,
            'usage_tags': {key: tags[key] for key in (
                'agentops.tokens.total', 'agentops.cost_usd', 'agentops.cost.status')},
            'native_worker_parent_ids': {span.name: span.parent_id for span in spans
                                         if span.name in expected},
        }
        output = Path(__file__).resolve().parents[1] / 'validation-results' / args.output
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps({'scenario': name, 'trace_id': trace_id, 'workers': sorted(workers),
                          'status': 'passed'}), flush=True)
    print('NATIVE TRACING WITHOUT AGENTOBSERVER: PASSED', flush=True)


if __name__ == '__main__':
    main()
