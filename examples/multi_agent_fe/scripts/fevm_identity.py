"""Provision demo serving credentials only after explicit identity approval.

Secrets go directly to Databricks Secrets and are never printed or written locally.
Resource permission grants and Lakebase role setup are separate deployment steps.
"""
import argparse
import json
from pathlib import Path


def main():
    from databricks.sdk import WorkspaceClient
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    workspace = WorkspaceClient(profile=args.profile)
    matches = list(workspace.service_principals.list(
        filter='displayName eq "fe-agentops-multi-runtime"'))
    if len(matches) > 1:
        raise ValueError("Multiple demo identities exist; choose one explicitly")
    principal = matches[0] if matches else workspace.service_principals.create(
        display_name="fe-agentops-multi-runtime", active=True)
    scope = "fe-agentops-multi"
    if scope not in {s.name for s in workspace.secrets.list_scopes()}:
        workspace.secrets.create_scope(scope=scope)
    keys = {key.key for key in workspace.secrets.list_secrets(scope=scope)}
    if "oauth-client-secret" not in keys:
        credential = workspace.service_principal_secrets_proxy.create(
            service_principal_id=principal.id, lifetime="2592000s")
        workspace.secrets.put_secret(scope=scope, key="oauth-client-secret",
                                    string_value=credential.secret)
    manifest = {
        "principal_id": principal.id,
        "application_id": principal.application_id,
        "secret_scope": scope,
        "secret_reference": "{{secrets/fe-agentops-multi/oauth-client-secret}}",
    }
    output = Path(__file__).resolve().parents[1] / "validation-results"
    output.mkdir(exist_ok=True)
    (output / "identity.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
