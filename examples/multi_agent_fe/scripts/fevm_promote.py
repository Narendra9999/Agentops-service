"""Promote a validated demo version, preserving identity and enabling trace capture."""
import argparse
import json
from pathlib import Path


def main():
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ServedEntityInput
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    workspace = WorkspaceClient(profile=args.profile)
    endpoint = workspace.serving_endpoints.get('fe-platform-agent-team')
    if endpoint.state.config_update.value != 'NOT_UPDATING':
        raise RuntimeError('Wait for the current configuration update to finish')
    current = endpoint.config.served_entities[0]
    if current.entity_name != 'fevm_catalog_naren.agentops_multi.platform_team':
        raise RuntimeError('Unexpected endpoint model; refusing to change it')
    environment = {**current.environment_vars,
                   'ENABLE_MLFLOW_TRACING': 'true',
                   'MLFLOW_ENABLE_TRACE_IN_SERVING': 'true',
                   'MLFLOW_EXPERIMENT_ID': '4384417075366092'}
    workspace.serving_endpoints.update_config(
        name=endpoint.name, served_entities=[ServedEntityInput(
            entity_name=current.entity_name, entity_version=args.version,
            workload_size='Small', scale_to_zero_enabled=True, environment_vars=environment)])
    result = {'endpoint': endpoint.name, 'requested_version': args.version,
              'tracing_enabled': True, 'experiment_id': '4384417075366092'}
    output = Path(__file__).resolve().parents[1] / 'validation-results/promotion.json'
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
