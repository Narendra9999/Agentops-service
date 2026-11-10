"""Verify persisted serving traces and the worker routing."""
import argparse
import json
from pathlib import Path


def main():
    from databricks.sdk import WorkspaceClient
    from mlflow import MlflowClient
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / 'validation-results'
    responses = json.loads((root / 'serving-smoke.json').read_text())
    client = MlflowClient(tracking_uri='databricks://' + args.profile,
                          registry_uri='databricks-uc://' + args.profile)
    expected = {'docs':'platform_docs','cost':'cost_estimation'}
    supplemental = root / 'serving-genie-smoke.json'
    if supplemental.exists():
        responses['explicit_genie'] = json.loads(supplemental.read_text())
        expected['explicit_genie'] = 'cost_estimation'
    report = {'traces': {}, 'routing_gaps': []}
    for name, worker in expected.items():
        trace_id = responses[name]['custom_outputs']['trace_id']
        trace = client.get_trace(trace_id)
        assert str(trace.info.experiment_id) == '4384417075366092'
        assert str(trace.info.state) == 'OK', (name, trace.info.state)
        spans = trace.data.spans
        routed = any(span.name == 'agent.' + worker for span in spans)
        if not routed:
            report['routing_gaps'].append({'scenario':name,'expected_worker':worker})
        tokens = [span.attributes.get('mlflow.chat.tokenUsage') for span in spans
                  if span.attributes.get('mlflow.chat.tokenUsage')]
        lineage = [span for span in spans if span.attributes.get('agentops.agent_id') == worker]
        if routed:
            assert lineage and lineage[0].attributes.get('agentops.caller_invocation_id')
        report['traces'][name] = {'trace_id': trace_id, 'state': str(trace.info.state),
            'worker': worker, 'span_count': len(spans), 'token_usage': tokens,
            'expected_worker_observed': routed,
            'context_visible': any(span.inputs for span in spans if span.span_type == 'CHAT_MODEL')}
    (root / 'deployment-verification.json').write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, default=str))


if __name__ == '__main__':
    main()
