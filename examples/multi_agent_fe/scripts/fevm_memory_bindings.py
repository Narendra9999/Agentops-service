"""Approval-gated signing bindings. Preserves existing App resources and model version."""
import argparse
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import App,AppResource
from databricks.sdk.service.serving import ServedEntityInput


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',required=True)
    parser.add_argument('--version',required=True,help='Validated memory/embedded-platform candidate')
    args=parser.parse_args()
    w=WorkspaceClient(profile=args.profile)
    # Patch only resources; preserve all existing endpoint/identity bindings.
    app=w.apps.get('fe-platform-team-fevm')
    resources=[r.as_dict() for r in app.resources or [] if r.name!='context-signing-key']
    resources.append({'name':'context-signing-key','secret':{'scope':'fe-agentops-context','key':'signing-key','permission':'READ'}})
    w.apps.create_update(app_name=app.name,update_mask='resources',app=App(resources=[AppResource.from_dict(r) for r in resources])).result()
    endpoint=w.serving_endpoints.get('fe-platform-agent-team')
    if endpoint.state.config_update.value!='NOT_UPDATING':raise RuntimeError('Wait for serving update completion')
    if len(endpoint.config.served_entities)!=1:raise RuntimeError('Unexpected serving topology')
    current=endpoint.config.served_entities[0]
    if current.entity_name!='fevm_catalog_naren.agentops_multi.platform_team':raise RuntimeError('Unexpected model')
    env={**current.environment_vars,'AGENT_CONTEXT_SIGNING_KEY':'{{secrets/fe-agentops-context/signing-key}}'}
    w.serving_endpoints.update_config(name=endpoint.name,served_entities=[ServedEntityInput(entity_name=current.entity_name,entity_version=args.version,workload_size='Small',scale_to_zero_enabled=True,environment_vars=env)])
    print('Signing resource attached and candidate serving update requested. Wait for readiness before deploying App source.')


if __name__=='__main__':main()
