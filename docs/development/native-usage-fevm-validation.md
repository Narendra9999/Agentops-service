# Shared native usage: FEVM validation

Validated on 2026-11-11 using the explicitly selected `fevm-naren` profile.
Both teams ran the original six-stage AgentOPS pipeline with the previous
authenticated-deployer identity; App deployment was outside this rollout.

| Agent | Endpoint | Model version | Pipeline run | Result |
| --- | --- | --- | --- | --- |
| Single agent | `fe-platform-single-validation` | 9 | [452001645922443](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/180098491824601/runs/452001645922443?o=7474644082064164) | SUCCESS |
| Supervisor | `fe-platform-intelligence` | 13 | [672021889787809](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/701419594686647/runs/672021889787809?o=7474644082064164) | SUCCESS |

## Implementation and checks

Both paths use `framework.observability.token_usage.export_native_model_usage`.
Direct Responses requests create one LLM span per model call. The supervisor
retains LangChain autologging and exports on its existing model spans.
Parent spans receive no token usage or cost, preventing double counting.
AgentObserver remains disabled for the supervisor.

The configured-cost finalizer preserves an explicitly selected AgentOPS rate
estimate after MLflow's built-in end-of-span pricing. Existing costs are
preserved when the shared exporter first encounters them. Prices are estimates
in USD, not billing assertions. Genie internal usage remains unmetered and the
supervisor cost summary retains its partial-coverage status.

Local validation passed 206 tests: 57 integration, 27 unit, and 122 supervisor
checks. Regression coverage reproduces MLflow's end-of-span cost overwrite,
checks prediction and streaming, preserves existing costs/processors, and
verifies cached-token pricing and one usage-bearing span per provider call.

## Pipeline quality gates

All final pre/postdeployment quality scores and guardrail accuracy were 1.0.
The single-agent first postdeployment attempt scored 0.667 on correctness
(threshold 0.8); its configured automatic retry passed. Thresholds and evaluation
data were not changed. The overall pipeline result is SUCCESS after that retry.

| Agent | Predeployment evaluation | Final postdeployment evaluation |
| --- | --- | --- |
| Single agent | `62633bd2-0951-4d2d-a77e-cdb01f312763` | `1c32ffa7-a268-4569-b378-5ef3000afc15` |
| Supervisor | `913db074-06d8-4d3e-9f39-8cf43b7ac777` | `8e5f2121-ceca-48e6-a47a-44620e00afdb` |

## Direct single-agent checks

The endpoint was READY / NOT_UPDATING on version 9 before testing.
All three scenarios passed. Native trace-header tokens and cost matched the
AgentOPS summary; streaming emitted supported output events and retained
citations. Blocked input generated no model usage or fabricated cost.

| Scenario | Trace | Tokens | Estimated USD |
| --- | --- | --- | --- |
| Prediction | `tr-b32944cc4ecf9b751506216a08848be2` | 3417 | 0.004389 |
| Streaming | `tr-81b85e473e3f8ba71cd62bbc21ad32a9` | 4965 | 0.005958 |
| Input block | `tr-c7b20f06af245f4741526dfc12e7b8a6` | No model call | No model call |

## Direct supervisor checks

The endpoint was READY / NOT_UPDATING on version 13 before testing.
All scenarios passed with AgentObserver disabled. Native tokens and costs
matched AgentOPS summaries; worker selection and supported streaming events
were checked, along with documentation citations for platform responses.

| Scenario | Workers | Trace | Tokens | Estimated USD |
| --- | --- | --- | --- | --- |
| cost | cost_estimation | `tr-0319386221fb3aaa3b8f31547f0fb443` | 6465 | 0.02682300 |
| platform | platform_docs | `tr-dec83acb8f34920a64c25d6f828c4414` | 8882 | 0.03032300 |
| combined | cost_estimation, platform_docs | `tr-7629f98d3c2d1356185c325107f32292` | 66071 | 0.11031500 |
