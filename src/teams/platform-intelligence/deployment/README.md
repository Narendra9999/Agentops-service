> Current endpoint: `fe-platform-intelligence`, version **3**.
> Original AgentOPS pipeline, Playground API compatibility and signed memory checks passed.
> See [validation and remaining issues](PLAYGROUND_FIX.md).

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
by the job-level parameter `dependency_source: environment`. Both the job
environment and model registration reference `../configs/requirements.lock`. Original customer jobs retain the default
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


## Pipeline refactor validation

Common notebook startup, inference invocation and metric gates now use shared
helpers. Registration builds one framework wheel and reuses it for upload.
Resources and sanity questions are explicitly configured by the team. Common
parameters are job-level parameters pushed to the original six notebook tasks.
See [the contract](../../../../docs/development/team-pipeline-contract.md).
FEVM run `470860625546934` completed all six original notebook stages and
promoted model version 5. Pre- and post-deployment evaluation each scored 1.0
for correctness, relevance, safety, routing, platform evidence and cost grounding;
all 17 guardrail fixtures passed. Smoke checks passed retrieval, normal queries,
injection blocking, off-topic handling and grounding. Registration initially
exposed a removed-helper reference; that regression was fixed and covered by a
startup test before this successful rerun.

The separate single-agent fixture registered successfully but its quality gate
blocked deployment because retrieval/answer evidence did not meet 0.8 thresholds.
See the [single-agent validation status](../../platform-engineering-test/README.md).
No threshold was lowered and the original single-agent endpoint was untouched.

## AgentObserver disabled: FEVM version 6

Run [736913769886698](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/701419594686647/runs/736913769886698?o=7474644082064164)
completed all six original AgentOPS stages and promoted version 6 on
`fe-platform-intelligence`. The registered artifact was checked directly:
`observer_enabled: false`, `memory_enabled: true`.

All six quality metrics remained 1.0 before and after deployment, matching the
observer-enabled version 5 baseline. All 17 guardrail fixtures and the original
serving smoke checks passed. The existing Vector Search index was reused.

Three direct streaming scenarios verified cost-only, platform-only and combined
routing using native LangGraph worker nodes. Each trace is OK, has model token
usage and persisted token/cost tags, and has no AgentObserver wrapper or invocation
attributes. Genie-related requests retain `partial_estimate` cost coverage.
Four signed-context scenarios verified Lakebase memory writes, fresh-session
recall, platform retrieval and Genie routing. Their traces are OK with the
observer flag false; memory tools and progress events remain available.

- Native cost trace: `tr-0ce4bca760a4830cc8410d1165cc87c2`.
- Native platform trace: `tr-aad6472aa8b53ad589720e499c367aad`.
- Native combined trace: `tr-958554bd604ff975a173eaf5f90e1e2c`.
- Cross-session memory recall: `tr-2cfd051d13c33419cd7da9d39729a799`.

Local checks passed 114 multi-agent tests and 16 team/startup tests, plus a real
local MLflow autologging check of the native graph without observer wrappers.
The updated bundle passed strict validation. App deployment remains deferred.
Machine-readable sanitized results are in
[validation-observer-disabled.json](validation-observer-disabled.json).

## Async middleware and supervisor: FEVM version 9

[Run 142625413609326](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/701419594686647/runs/142625413609326?o=7474644082064164)
completed all six original AgentOPS stages. Version 9 serves 100% of
`fe-platform-intelligence` traffic. Pre- and post-deployment quality metrics
were all 1.0; all 17 guardrail fixtures passed.

The shared framework uses one async middleware lifecycle; the supervisor awaits
its LangGraph nodes and model transport. Synchronous MLflow serving entry points
remain compatible. Genie and Lakebase SDK operations remain thread-adapted.
AgentObserver remains disabled. Customer orchestration and operation policies
remain team-owned; AgentOPS applies policies at the outer agent boundary.

Three direct streaming scenarios passed cost-only, platform-only and combined
routing with the Playground response contract. Native MLflow token totals match
AgentOPS totals for every scenario. LangChain is the sole model usage span: the
async HTTP transport avoids duplicate OpenAI instrumentation. This corrects
telemetry double counting, not duplicate provider requests. Genie-related cost
coverage remains `partial_estimate`.

Four signed-context scenarios passed explicit memory writes, persisted Lakebase
files, recall from a fresh session, documentation retrieval and Genie routing.
Two simultaneous nonstreaming requests returned separate thread IDs and OK traces
with response usage. The 250 local framework, middleware and supervisor tests
also passed.

Optional PII transformation middleware is packaged but the deployed team keeps
its existing blocking policy. Enabling input redaction for a signed application
context requires a team adapter; telemetry redaction is a separate policy.
See [middleware configuration](../../../../docs/development/guardrail-middleware.md)
and [async runtime contracts](../../../../docs/development/async-runtime.md).
App deployment remains deferred.

Sanitized results: [validation-async-runtime.json](validation-async-runtime.json).
