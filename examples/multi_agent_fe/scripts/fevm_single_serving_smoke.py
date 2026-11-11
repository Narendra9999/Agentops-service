"""Check single-agent boundary contracts on an explicitly selected serving version."""
import argparse
import json
import time
import uuid
from pathlib import Path

import requests
from databricks.sdk import WorkspaceClient
from mlflow import MlflowClient
from mlflow.types.responses import ResponsesAgentResponse, ResponseOutputItemDoneEvent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--experiment-name', default='/Shared/agentops-platform-single-validation')
    args = parser.parse_args()
    workspace = WorkspaceClient(profile=args.profile)
    endpoint = workspace.serving_endpoints.get(args.endpoint)
    versions = {item.name: str(item.entity_version) for item in endpoint.config.served_entities}
    live = {versions[getattr(route, 'served_entity_name', None) or route.served_model_name]
            for route in endpoint.config.traffic_config.routes if route.traffic_percentage > 0}
    assert endpoint.state.ready.value == 'READY' and endpoint.state.config_update.value == 'NOT_UPDATING'
    assert live == {args.version}, 'Requested version is not receiving all traffic'
    client = MlflowClient(tracking_uri='databricks://' + args.profile)
    experiment = client.get_experiment_by_name(args.experiment_name)
    assert experiment is not None
    report = {'endpoint': args.endpoint, 'version': args.version, 'scenarios': {}}
    scenarios = {
        'prediction': ('Explain MLflow tracing for LangGraph and cite retrieved Databricks documentation URLs.', False),
        'streaming': ('Explain Auto Loader incremental file ingestion and cite retrieved Databricks documentation URLs.', True),
        'input_block': ('?', False),
    }
    for name, (question, streaming) in scenarios.items():
        marker = str(uuid.uuid4())
        payload = {'input': [{'role': 'user', 'content': question}], 'stream': streaming,
                   'custom_inputs': {'thread_id': marker, 'tags': {'agentops.validation_id': marker}}}
        with requests.post(workspace.config.host.rstrip('/') + '/serving-endpoints/' + args.endpoint + '/invocations',
                           headers=workspace.config.authenticate(), json=payload,
                           stream=streaming, timeout=(20, 240)) as response:
            response.raise_for_status()
            if streaming:
                items = []
                for raw in response.iter_lines():
                    if not raw or not raw.startswith(b'data:'):
                        continue
                    raw = raw[5:].strip()
                    if raw == b'[DONE]':
                        break
                    event = json.loads(raw)
                    assert not event.get('error') and event.get('type') not in {'error', 'response.failed'}, event
                    assert event.get('type') in {
                        'response.output_text.delta', 'response.output_text.annotation.added',
                        'response.output_item.done',
                    }, ('Playground-incompatible event', event.get('type'))
                    if event.get('type') == 'response.output_item.done':
                        ResponseOutputItemDoneEvent.model_validate(event)
                        items.append(event['item'])
                        assert event['custom_outputs']['thread_id'] == marker
                result = {'output': items}
            else:
                result = response.json()
                ResponsesAgentResponse.model_validate(result)
                assert result['custom_outputs']['thread_id'] == marker
        text = '\n'.join(part.get('text', '') for item in result['output'] for part in item.get('content', []))
        assert text, name
        assert (('too short' in text.lower()) if name == 'input_block'
                else ('https://docs.databricks.com/' in text)), (name, 'Expected block message or retrieved citation')
        trace = None
        for _ in range(12):
            candidates = client.search_traces(locations=[experiment.experiment_id], max_results=100, include_spans=False)
            trace = next((item for item in candidates if item.info.tags.get('agentops.validation_id') == marker), None)
            if trace is not None:
                break
            time.sleep(2)
        assert trace is not None and str(trace.info.state) == 'OK', name
        tags = trace.info.tags
        native_usage = json.loads(trace.info.request_metadata.get('mlflow.trace.tokenUsage', '{}'))
        native_cost = json.loads(trace.info.request_metadata.get('mlflow.trace.cost', '{}'))
        if name == 'input_block':
            assert tags['agentops.guardrail.pre_llm.blocked'] == 'true'
            assert int(tags.get('agentops.tokens.total', 0)) == 0
            assert native_usage.get('total_tokens', 0) == 0
        else:
            assert int(tags['agentops.tokens.total']) > 0
            assert float(tags['agentops.cost_usd']) >= 0
            assert int(tags['agentops.llm_requests']) == 1
            assert native_usage['total_tokens'] == int(tags['agentops.tokens.total'])
            assert abs(native_cost['total_cost'] - float(tags['agentops.cost_usd'])) <= 0.000001
        report['scenarios'][name] = {'trace_id': trace.info.trace_id, 'state': str(trace.info.state),
            'tokens': tags.get('agentops.tokens.total'), 'cost_usd': tags.get('agentops.cost_usd'), 'status': 'passed',
            'native_token_usage': native_usage, 'native_cost': native_cost}
        print(json.dumps({'scenario': name, **report['scenarios'][name]}), flush=True)
    out = Path(__file__).resolve().parents[1] / 'validation-results' / 'async-single-serving.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
