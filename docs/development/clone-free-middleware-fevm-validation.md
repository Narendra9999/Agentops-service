# Clone-free callback middleware validation

Framework commit: `516ccae`, branch `feat/platform-intelligence-supervisor`.
Validation date: 2026-11-12. FEVM profile: `fevm-naren`.

## Implementation

AgentOPS invokes targets through callbacks with explicit, single-use
`RequestContext`. No middleware or target agent is shallow-copied or deep-copied.
Context-aware target hooks receive state and resources explicitly. Independent
ResponsesAgents retain their public interface. Both paths use one shared
middleware lifecycle. Request resources close on completion, failure, timeout or
cooperative cancellation. No agent instance owns mutable per-request context.

Remaining defensive deep copies protect configuration resolution and nested
payload redaction. They do not copy agents, clients, or resource stacks.

## Local validation

267 tests passed in separate processes: 215 integration/supervisor example tests
and 52 usage, runtime and input-guardrail unit tests. Unit tests install module
stubs, so they must run separately from integrations using actual MLflow.

The 36 wrapper tests passed again after adding copy/deepcopy sentinels to plain
and context-aware targets. This is a targeted repeat, not 36 additional tests.
Coverage includes concurrent request isolation, raw hook dispatch without a
second lifecycle, streaming fallback, guardrail ordering/redaction, resource
cleanup, deadlines, failed streams, single-use context rejection, model save/load,
and native token/cost parity without duplicate model calls.

Both team bundles passed strict validation. All five lifecycle SVGs and the HTML
viewer were regenerated; function dependency documentation matches current source.

## FEVM pipelines

Original six-stage AgentOPS pipelines and quality thresholds are retained.
The App is outside this rollout. Both pipelines completed all six stages with
SUCCESS. Direct checks passed on stable versions 11 and 15, both READY / NOT_UPDATING.

| Target | Validated version | Pipeline run |
| --- | --- | --- |
| Single agent | 11 | [299101421022807](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/180098491824601/runs/299101421022807?o=7474644082064164) |
| Supervisor | 15 | [377155193313747](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/701419594686647/runs/377155193313747?o=7474644082064164) |

The supervisor's first predeployment evaluation failed platform evidence
(0.667 versus 0.8); all other six metrics scored 1.0 and guardrail accuracy was
100%. The automatically scheduled retry passed the unchanged gate. The failed
assessment on trace `tr-9a967fdd5102e74c3988ddfe034054bc` identified Serverless Jobs
guidance citing agent/model-serving documentation. This is a citation-quality
limitation; successful retry does not erase that evidence. Evaluation audit ID:
`edf5921c-d2b7-4874-9b0e-197aff30654b`.

Earlier provisional callback runs `1120451891175442` and `312221790795013` were
cancelled when the design changed to remove all legacy cloning compatibility.
They are not counted as validation of this refactor.

## Direct single-agent checks

Stable serving version 11 was READY / NOT_UPDATING before testing. All scenarios
passed. Native token/cost metrics matched AgentOPS tags. Successful requests had
exactly one precheck, one postcheck and one provider call. Input rejection had
one precheck, no postcheck and no provider call.

| Scenario | Trace | Tokens | Estimated USD |
| --- | --- | --- | --- |
| prediction | `tr-d97eb3f4d02c86fc54ea01d4ad27959b` | 3379 | 0.004281 |
| streaming | `tr-ec97208d8aa3362d836a4d6e18188790` | 4724 | 0.005460 |
| input_block | `tr-e107134e6ae6562649b8f61ff5fdee8d` | N/A | N/A |

## Usage coverage boundary

Observed local model calls are deduplicated by run ID. Response aggregates are
exported without adding them again to child totals. The remote-worker adapter
captures returned usage/trace IDs and sends trace context, but does not fold remote
worker aggregates into observed-model totals. Genie internal usage remains
unmetered where unavailable. Neither this refactor nor successful evaluation
claims complete remote-service billing coverage.

## Direct supervisor checks

All three routing scenarios passed with AgentObserver disabled, valid completed
output events, exactly one precheck/postcheck, and matching native/AgentOPS
model-call usage totals. Genie coverage remains explicitly partial.

| Scenario | Trace | Workers |
| --- | --- | --- |
| cost | `tr-86d274da899af67f32a9b8d88abaa24e` | cost_estimation |
| platform | `tr-581101ab2873bc7e4a1f53b5ec59a123` | platform_docs |
| combined | `tr-d29aa76e1003bf470df983297d819f94` | cost_estimation, platform_docs |
