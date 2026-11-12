# Wrapper-owned middleware validation

Validated locally on 2026-11-12. Commit: `54c3f90` on
`feat/platform-intelligence-supervisor`.

## Architecture

The shared serving wrapper constructs and calls AgentOPS middleware:

`CostTrackingResponsesModel → AgentOPSMiddleware → TargetAgentAdapter → target`

The middleware reuses `AgentOPSBase._run_lifecycle` for prediction and streaming.
The supervisor target no longer creates `_GuardedTurn`. Legacy targets are called
through raw hooks, avoiding a second middleware lifecycle. Model/tool orchestration
remains owned by each target.

## Local evidence

233 tests passed in separate pytest processes:

- 84 inference integration tests, including 27 wrapper-owned middleware cases.
- 122 supervisor example tests.
- 27 shared usage/runtime unit tests.

Unit tests that install module stubs run separately from integrations requiring
real MLflow packages. Collecting both in one process causes stub contamination.
Deadline cleanup is tested independently of the local default tracing backend;
native metric tests use a real file-store exporter.

Coverage includes synchronous ResponsesAgents, native async agents, PythonModels,
legacy hooks, actual supervisor entrypoint, MLflow save/load, prediction and
streaming, exact precheck/target/postcheck ordering, input blocks preventing calls,
output blocks before delivery, input/output PII redaction, history and memory
recall, session persistence, concurrent state isolation, structured Genie artifact
checks, usage preservation, streaming failures, cooperative cancellation and
whole-request deadline cleanup. Native model metrics tests verify one provider
request and one usage-bearing model span, cached-token pricing and cost parity.

Streaming final output is buffered by default. Explicit passthrough releases output
before checks finish, so later checks cannot retract delivered tokens. Output PII
redaction overrides passthrough. Optional public activity is separate from output
and requires the client opt-in; it does not expose private model reasoning.

## FEVM rollout

Profile: `fevm-naren`. Identity: previous authenticated-deployer method. Original
six-stage AgentOPS bundles retained, with unchanged quality gates. App deployment
is outside this rollout. Both pipelines completed all six stages with SUCCESS.

| Agent | Endpoint / version | Pipeline result | Pipeline run |
| --- | --- | --- | --- |
| Single agent | `fe-platform-single-validation` / 10 | SUCCESS | [354884206504902](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/180098491824601/runs/354884206504902?o=7474644082064164) |
| Supervisor | `fe-platform-intelligence` / 14 | SUCCESS | [568632648935660](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/701419594686647/runs/568632648935660?o=7474644082064164) |

## Direct supervisor checks

Version 14 was READY / NOT_UPDATING before testing. All three scenarios passed
with AgentObserver disabled and default Playground-compatible completed-item
streaming. Each trace had exactly one `pre_llm_check` and one `post_llm_check`.
Only LLM/CHAT_MODEL spans carried token usage, and native totals matched AgentOPS
summary tags. No extra model invocation was introduced for middleware.

| Scenario | Workers | Trace | Tokens | Estimated USD |
| --- | --- | --- | --- | --- |
| cost | cost_estimation | `tr-91d865a0967ea3d30e0cca29fc38d0b0` | 5064 | 0.02236800 |
| platform | platform_docs | `tr-499b048392bcb990ccd3a14cf6595728` | 9540 | 0.03371900 |
| combined | cost_estimation, platform_docs | `tr-bb8bb82ce381e4efcba1cca9375a0ece` | 38455 | 0.09163500 |

Costs remain configured-rate estimates, not billing assertions. Genie internal
usage is unmetered, so cost and combined scenarios retain `partial_estimate`.

## Direct single-agent checks

Version 10 was READY / NOT_UPDATING before testing. Prediction and streaming
returned valid Responses output with retrieved citations and exactly one model
request each. Each accepted request had one precheck and one postcheck. Blocked
input had one precheck, no postcheck, no model usage and no fabricated cost.

| Scenario | Trace | Tokens | Estimated USD |
| --- | --- | --- | --- |
| prediction | `tr-3d6fbaa128ea729f7f3187e22cf8cf55` | 3337 | 0.004037 |
| streaming | `tr-b3a7ea213cb33b9048a1c37ebe3fdc67` | 4743 | 0.005498 |
| input_block | `tr-a00dc1e49ae11e4489c4fe682997ea24` | No model call | No model call |
