"""Validate concurrent nonstreaming requests on an explicitly selected FEVM version."""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path
import uuid

import httpx
from databricks.sdk import WorkspaceClient
from mlflow import MlflowClient


async def exercise(workspace, endpoint):
    questions = {
        'platform': 'Explain MLflow tracing for LangGraph. Retrieve Databricks documentation and cite its URLs. Do not estimate costs.',
        'cost': 'Ask Cost intelligence which inputs are missing to estimate a scheduled batch workload. Volume, frequency and duration are unknown. Do not invent a numerical quote.',
    }
    headers = await asyncio.to_thread(workspace.config.authenticate)
    async with httpx.AsyncClient(timeout=httpx.Timeout(240, connect=20)) as client:
        async def invoke(name, question):
            thread = 'async-validation-' + str(uuid.uuid4())
            response = await client.post(
                workspace.config.host.rstrip('/') + '/serving-endpoints/' + endpoint + '/invocations',
                headers=headers, json={'input': [{'role': 'user', 'content': question}],
                                       'custom_inputs': {'thread_id': thread}})
            response.raise_for_status()
            payload = response.json()
            assert isinstance(payload.get('output'), list) and payload['output'], name
            custom = payload.get('custom_outputs') or {}
            assert custom.get('thread_id') == thread, 'Concurrent request identity mismatch'
            assert custom.get('trace_id'), name
            usage = payload.get('usage') or {}
            assert usage.get('total_tokens', 0) > 0, 'Response token usage missing'
            assert custom.get('memory', {}).get('mode') == 'stateless', name
            return name, {'thread_id': thread, 'trace_id': custom['trace_id'], 'usage': usage}
        return dict(await asyncio.gather(*(invoke(name, question) for name, question in questions.items())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--output', default='async-supervisor-concurrency.json')
    args = parser.parse_args()
    os.environ['DATABRICKS_CONFIG_PROFILE'] = args.profile
    workspace = WorkspaceClient(profile=args.profile)
    endpoint = workspace.serving_endpoints.get(args.endpoint)
    versions = {entity.name: str(entity.entity_version) for entity in endpoint.config.served_entities}
    live = {versions[getattr(route, 'served_entity_name', None) or route.served_model_name]
            for route in endpoint.config.traffic_config.routes if route.traffic_percentage > 0}
    assert endpoint.state.ready.value == 'READY' and endpoint.state.config_update.value == 'NOT_UPDATING'
    assert live == {args.version}, 'Expected version is not receiving all traffic'
    scenarios = asyncio.run(exercise(workspace, args.endpoint))
    assert len({result['trace_id'] for result in scenarios.values()}) == len(scenarios)
    client = MlflowClient(tracking_uri='databricks://' + args.profile,
                          registry_uri='databricks-uc://' + args.profile)
    for name, result in scenarios.items():
        trace = None
        for _ in range(6):
            trace = client.get_trace(result['trace_id'])
            if trace is not None:
                break
            time.sleep(2)
        assert trace is not None and str(trace.info.state) == 'OK', name
        assert int(trace.info.tags.get('agentops.tokens.total', 0)) > 0, name
        result['trace_state'] = str(trace.info.state)
    report = {'endpoint': args.endpoint, 'version': args.version, 'concurrent_requests': len(scenarios),
              'scenarios': scenarios, 'status': 'passed'}
    output = Path(__file__).resolve().parents[1] / 'validation-results' / args.output
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
