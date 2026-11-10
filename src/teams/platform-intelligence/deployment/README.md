> Fresh agent `fe-platform-intelligence`, model version **1**, passed the original
> pipeline and live validation. See [fresh deployment results](FRESH_DEPLOYMENT.md).
> App deployment remains deferred; earlier endpoint results below are historical.

# Original AgentOPS pipeline in FEVM

This bundle supplies FEVM compute and team parameters to the **original AgentOPS
notebooks**. It contains no separate registration, evaluation, deployment or smoke
implementation. Its task paths and dependencies are checked against
`resources/pipeline-resource.yml` by `tests/unit/test_team_runtime.py`.

| Stage | Original shared notebook |
|---|---|
| Vector Search setup | `src/data_preparation/vector_search/notebooks/VectorSearch.py` |
| Register model | `src/agent_development/agent/notebooks/RegisterModel.py` |
| Pre-deployment evaluation | `src/agent_development/agent_evaluation/notebooks/PreDeploymentEval.py` |
| Deploy agent | `src/agent_development/agent/notebooks/DeployAgent.py` |
| Smoke test | `src/agent_development/agent_evaluation/notebooks/SmokeTest.py` |
| Post-deployment evaluation | `src/agent_development/agent_evaluation/notebooks/PostDeploymentEval.py` |

Run from this directory with the explicitly selected FEVM profile:

```sh
databricks bundle validate --strict -t fevm --profile fevm-naren
databricks bundle deploy -t fevm --profile fevm-naren
databricks bundle run agentops_pipeline -t fevm --profile fevm-naren --no-wait
```

The deployment notebook uses `databricks.agents.deploy()`. Registration uses the
existing framework wheel and team `agent/Agent.py` through the shared
`CostTrackingResponsesModel`. Evaluation uses the shared `run_evaluation()`, audit
logging, team fixtures and scorer loader. No Genie agent or worker endpoint is
created. Vector Search creation is disabled because the configured index exists.

FEVM uses serverless environments with the team's pinned dependencies, selected
by `dependency_source: environment`. Original customer jobs retain the default
`volume` mode and offline dependency installation. FEVM disables environment
packing in team config; the shared registration path still supports its default
offline `env_pack` behavior.

`framework.team_runtime.TeamRuntime` loads an optional trusted
`teams/<team>/request_adapter.py` to construct signed evaluation requests. This
team's adapter assigns isolated evaluation owners and signs bounded context using
a secret reference; it does not substitute lifecycle steps. The shared scorer
loader loads `scorers/routing.py` from this team. Canonical golden fixtures retain
`request`, `expected_response` and typed `expected_output` fields. Adversarial
fixtures cover this team's enabled length, PII and injection checks.

This team opts out of registration-time promotion. The original notebooks pass
this run's candidate version to evaluation and deployment and advance `champion`
after serving readiness. Existing serving environment/secret references are
preserved. Post-deployment failure does not automatically roll back deployment.

The earlier version 19 rollout used a separate team lifecycle and is historical
functional evidence only. Results for the original notebooks are recorded below.

## Original AgentOPS pipeline rollout — version 20 (2026-11-10)

The original AgentOPS notebook stages completed successfully in FEVM:
[run 414613831508506](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/701419594686647/runs/414613831508506?o=7474644082064164).
The run was repaired for shared registration/bootstrap integration errors before
its successful quality gates. Registration produced model version **20**, and
`DeployAgent.py` deployed it through `databricks.agents.deploy()`.

| Quality gate | Before | After |
|---|---|---|
| correctness | 1.0 | 1.0 |
| relevance_to_query | 1.0 | 1.0 |
| safety | 1.0 | 1.0 |
| cost_grounding | 1.0 | 1.0 |
| platform_evidence | 1.0 | 1.0 |
| routing | 1.0 | 1.0 |

Both phases evaluated the three team scenarios through the shared
`run_evaluation()` and enforced all six thresholds. Guardrail accuracy was 1.0
on 17 team fixtures covering the configured length, PII and injection checks.
The Vector Search task reused the configured index with creation disabled.
Original smoke and post-deployment stages passed. No Genie agent or worker
endpoint was deployed.

Live App/serving checks passed for persistent memory writes, fresh-session
recall, Platform retrieval, Genie cost routing, conversation history and rename,
and test-note soft deletion. This extends the original pipeline checks with
functional validation; these scripts do not replace pipeline stages.

Local validation: 70 focused shared-framework tests, 96 example Python tests,
nine Node chart/link tests, strict bundle validation and an isolated framework
wheel build passed. The full repository suite remains affected by pre-existing
chunking stub, dashboard-monitor tests and a missing monitoring resource file.
The chunking and dashboard source files were verified unchanged from the
previous rollout commit.

The team bundle supplies FEVM serverless compute and parameters to the original
notebooks. The customer Jenkins-rendered root bundle was not executed directly.
Default customer offline dependency behavior remains available; FEVM explicitly
uses its pinned job environment and disables environment packing and Prompt
Registry in team config. Hash-verified prompt artifacts are packaged instead.

Lakebase remains the only persistence service. Required Lakebase SDK names and
PostgreSQL protocol identifiers remain; no standalone PostgreSQL host or DSN is
used. Request usage reports provider tokens and configured-rate cost estimates;
Genie internal model usage remains unmetered.

Validation evidence is retained locally under the ignored
`examples/multi_agent_fe/validation-results/` directory.
