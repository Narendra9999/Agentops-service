"""Exercise the supervisor against existing knowledge and cost workers."""
import argparse
import json
import time
from pathlib import Path


def main():
    from databricks.sdk import WorkspaceClient
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--wait-seconds', type=int, default=0)
    args = parser.parse_args()
    w = WorkspaceClient(profile=args.profile)
    endpoint = w.serving_endpoints.get('fe-platform-agent-team')
    deadline = time.monotonic() + args.wait_seconds
    while (endpoint.state.ready.value != 'READY' or endpoint.state.config_update.value != 'NOT_UPDATING') and time.monotonic() < deadline:
        print('Waiting for serving configuration completion', flush=True)
        time.sleep(15)
        endpoint = w.serving_endpoints.get('fe-platform-agent-team')
    if endpoint.state.ready.value != 'READY' or endpoint.state.config_update.value != 'NOT_UPDATING':
        raise RuntimeError('Wait for serving update completion before testing')
    if endpoint.config.served_entities[0].entity_version != args.version:
        raise RuntimeError('Requested validated model version must be serving')
    evidence = Path(__file__).resolve().parents[1] / 'validation-results'
    report = {}

    def call(name, question, custom=None):
        body = {'input': [{'role': 'user', 'content': question}]}
        if custom:
            body['custom_inputs'] = custom
        response = w.api_client.do('POST', '/serving-endpoints/fe-platform-agent-team/invocations', body=body)
        report[name] = response
        (evidence / 'serving-smoke.json').write_text(json.dumps(report, indent=2))
        print(name, json.dumps(response)[:1500], flush=True)
        assert response.get('output'), 'Response must include an answer'
        assert response.get('custom_outputs', {}).get('trace_id'), 'A serving trace ID is required'
        return response

    call('docs', 'How does MLflow tracing help debug LangGraph workers? Cite a Databricks documentation URL.')
    call('cost', 'Estimate the monthly Databricks DCP cost for a new ingestion workload. I have not supplied volume or run frequency. Ask for the missing inputs; do not invent a numeric quote.')


if __name__ == '__main__':
    main()
