# AgentOPS serverless team pipeline

This FEVM deployment is a team-scoped AgentOPS pipeline. It uses the standard
`agent/Agent.py:ChatModel` contract, `framework/inference/cost_tracking_model.py`
entry point, `AgentOPSBase` guardrails, shared model-resource declarations, team
configuration, MLflow evaluation datasets/scorers, and existing serving endpoint.
It does not execute the root customer bundle, which requires rendered Jenkins
placeholders, an offline wheel volume and customer cluster policies.

From this directory, with an explicitly selected FEVM profile:

```sh
databricks bundle validate --strict -t fevm --profile fevm-naren
databricks bundle deploy -t fevm --profile fevm-naren
databricks bundle run agentops_pipeline -t fevm --profile fevm-naren --no-wait
```

Stages: Lakebase preflight → model registration → pre-deployment MLflow evaluation
→ serving deployment → framework smoke checks → post-deployment MLflow evaluation.
Deployment runs only after pre-evaluation passes; registration does not promote
aliases. Deployment preserves the existing endpoint's runtime identity, secret
references and Gateway configuration, waits for the requested model version to
be fully ready, then records `champion` and `previous` aliases. A failed post gate
marks the job failed; it does not automatically roll the endpoint back. Review
the failed assessments and use the recorded previous version when necessary.

Evaluation covers cost-only, platform-only and combined requests using the team's
durable MLflow evaluation dataset. Existing registered scorers are reused; missing
configured builtin/domain scorers are registered. Missing, nonfinite or low quality
scores and scorer errors fail the gate. Detailed functional validation additionally
uses the live App and isolated Lakebase memory owners; it never truncates history
or memory tables. No worker endpoint, Genie space or retrieval index is deployed.

## Lakebase storage boundary

Conversation history, preferences and curated Markdown memory use the existing
Lakebase Autoscaling project endpoint. The App uses `HISTORY_LAKEBASE_ENDPOINT`;
agent memory uses `multi_agent.settings.memory_endpoint`. Both reject hostnames and
external connection strings. Connections resolve the Lakebase host through the
Databricks SDK, obtain short-lived OAuth credentials and verify TLS.

`WorkspaceClient.postgres`, `databricks.sdk.service.postgres`, `postgres_role` and
`databricks_postgres` are Databricks API identifiers and the existing Lakebase
database name. `pg8000` is the protocol driver for Lakebase. These necessary names
do not represent a separate PostgreSQL service. The old disposable local database
test suite has been retired; live durability checks run against FEVM Lakebase.
