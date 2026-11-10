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

Pre-deployment evaluation clears the notebook's application import cache before
loading the registered artifact and verifies that runtime imports resolve inside
the downloaded model. It also imports the lazy Genie transport before running
scenarios. Existing Unity Catalog datasets are reloaded by name for mutation;
changed registered guideline scorers use the Databricks update API while preserving
their sampling settings. Prediction/scorer errors in individual evaluation rows
fail the gate even when aggregate scores appear to pass.

An additional native MLflow routing scorer requires exactly the workers specified
by each scenario. A correct-looking answer without the required worker invocation
fails. Post-deployment predictions include the ResponsesAgent execution trace ID;
the routing scorer loads that serving trace and fails if it is unavailable, rather
than judging the caller's HTTP span. The six gates are correctness, relevance,
safety, cost grounding, platform evidence and routing, each at the configured
0.8 threshold. Cost-input clarification still requires Genie delegation and omits
numeric currency rates or monetary totals; estimates require query-backed rates.

## Documentation index recovery

The original FEVM index `fevm_catalog_naren.agentops.databricks_docs_index`
cannot resume incremental sync: Databricks reports
`VECTOR_SEARCH_SOURCE_HISTORY_OUT_OF_RETENTION`. Its existing 81 indexed chunks
remain available. A team-owned replacement,
`fevm_catalog_naren.agentops_multi.platform_docs_index`, was created on the same
Vector Search endpoint from the existing Delta source table, with 84 chunks.
It includes three chunks from the official serverless jobs documentation and
grants the existing serving identity SELECT. Team config now binds the replacement.
Neither the old index nor its source table was deleted. This was a one-time
FEVM resource repair; the AgentOPS deployment pipeline still reuses its configured
index instead of provisioning indexes or workers.

To refresh the added official jobs page and request index sync:

```sh
python examples/multi_agent_fe/scripts/fevm_refresh_jobs_docs.py \
  --profile fevm-naren --warehouse-id 2b169603bdabc084
```

Index synchronization is asynchronous. Verify readiness and that queries return
the new document before treating the refresh as effective.

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

## AgentOPS pipeline rollout — version 19 (2026-11-10)

The team-scoped serverless AgentOPS pipeline completed successfully:
[run 891259459764591](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/701419594686647/runs/891259459764591?o=7474644082064164).
All six tasks passed: preflight, registration, pre-evaluation, deployment, framework
smoke, and post-evaluation. Endpoint `fe-platform-agent-team` is READY /
NOT_UPDATING on `fevm_catalog_naren.agentops_multi.platform_team` version **19**;
`champion=19`, `previous=18`.

| Evaluation gate | Before | After |
|---|---|---|
| Correctness | 1.0 | 1.0 |
| Relevance | 1.0 | 1.0 |
| Safety | 1.0 | 1.0 |
| Cost grounding | 1.0 | 1.0 |
| Platform evidence | 1.0 | 1.0 |
| Actual specialist routing | 1.0 | 1.0 |

Each phase evaluated cost-only, platform-only and combined requests. Pre-evaluation
run: `a6c22bd1478447f39ca4406f5de7299f`; post-evaluation run:
`6448b18787f648149178264d3b3b589d`. These scenarios validate routing and grounded
input clarification; they do not establish accuracy for every numerical quote.

Additional live checks passed:

- Framework response tags and input guardrail: `tr-25970fa2a6a0cd09f1628a823d59e731`.
- Explicit memory write persisted in Lakebase and fresh-session read succeeded.
- Embedded Platform retrieval and Genie-only cost delegation selected the expected workers.
- Authenticated App history, conversation rename, memory write, fresh-conversation
  recall and test-note soft deletion succeeded. All seven App/serving traces are OK.
- Platform trace `tr-336734114ae97f505be741de0dc5f7d1` contains seven model spans
  with native token usage and configured-rate cost attributes. Genie internal usage
  remains unmetered; these are estimates for observed calls, not a total Genie bill.

Local validation: 95 example Python tests passed, one optional test skipped;
10 framework/deployment unit tests and nine Node chart/link tests passed.
Strict bundle validation and Git whitespace checks passed. The full legacy root
suite was not validated: its customer resource template remains unavailable.

Persistence uses the approved Lakebase project endpoint only. There is no external
or local PostgreSQL service/DSN. Required Lakebase SDK/protocol identifiers such as
`WorkspaceClient.postgres`, `postgres_role`, `databricks_postgres` and `pg8000`
remain because Lakebase uses the PostgreSQL protocol.

The team-owned documentation index
`fevm_catalog_naren.agentops_multi.platform_docs_index` is READY with 84 chunks,
including three official serverless Jobs documentation chunks. The original
81-chunk index was retained; its failed incremental-sync pipeline was stopped.
The rollout reuses the existing Genie agent and deploys no separate worker endpoint.

This is the team-scoped pipeline using the shared AgentOPS framework. The legacy
root customer/Jenkins bundle was not run. Machine-readable rollout, smoke and trace
checks are retained locally under ignored `examples/multi_agent_fe/validation-results/`.
