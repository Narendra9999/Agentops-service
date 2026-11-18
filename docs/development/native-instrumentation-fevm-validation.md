# Native instrumentation FEVM validation

Implementation commit: `a84cacf` on `feat/platform-intelligence-supervisor`.

## Shared behavior

Both agent patterns use the original AgentOPS registration/deployment pipeline and the shared serving wrapper. `tracing.instrumentation` selects one native MLflow integration during model initialization, before importing the team target. The supervisor selects `langchain`; the plain Chat Completions target selects `openai`. No customer observer wrappers are required for supported operations. The separate custom observer remains disabled.

The supervisor team entry point no longer enables autologging itself. The single-agent Chat Completions test target no longer creates a manual model span or exports native model usage. Missing native OpenAI cost uses the same rate card as shared response accounting. Existing native costs take precedence on model spans; existing response cost tags still use the configured rate card, as documented below.

## Isolated FEVM serverless checks

[Validation run](https://fevm-classic-stable-naren.cloud.databricks.com/?o=7474644082064164#job/275582506282706/run/1099168886023276). Experiment: `3116421550757936`. All eight cases passed. HTTP failures were injected through a mock transport; no deployed worker endpoint was changed. LangChain failures used a controlled Runnable operation.

| Integration | Enabled | Injected failure | Internal spans | Error spans | Trace |
|---|---|---|---|---|---|
| openai | True | False | 1 | 0 | `tr-ad4d1ea18b845e7a5534fd1f1fa75744` |
| openai | True | True | 1 | 1 | `tr-6d697f8e422a09401ff16437224b064e` |
| openai | False | False | 0 | 0 | `tr-26892dcb6205c28ca0743e82493ebd3c` |
| openai | False | True | 0 | 0 | `tr-ca7c3832146fb2e66d495940be5712dd` |
| langchain | True | False | 1 | 0 | `tr-e3e881a7bcaccf9eb791f7b53825876e` |
| langchain | True | True | 1 | 1 | `tr-dcf5fe0e1a66d2ed5eabed4a67a272b6` |
| langchain | False | False | 0 | 0 | `tr-15476b31420ebe065938e7ee9764154e` |
| langchain | False | True | 0 | 0 | `tr-5b3daad5e077f715043064f2ac31ea97` |

A handled internal failure remained visible on the child span without incorrectly failing the outer request. Disabling the integration omitted internal native spans while retaining the explicit outer middleware span.

## Local checks

- 140 integration tests passed, excluding the separate live endpoint test module.
- Seven team/resource tests passed in a separate process.
- Native OpenAI integration exercised through a simulated HTTP transport: one success and one handled failure, enabled and disabled; exact model token usage and configured cost checked without parent usage duplication.

## Rollout and live endpoint checks

Both original six-task pipelines completed successfully. Every configured pre- and post-deployment quality score and guardrail accuracy was 1.0.

| Pattern | Endpoint | Version | Successful pipeline run |
|---|---|---|---|
| Single / Chat Completions | `fe-platform-single-validation` | 16 | [776806651287589](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/180098491824601/runs/776806651287589?o=7474644082064164) |
| Supervisor / LangGraph | `fe-platform-intelligence` | 20 | [528791539956287](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/701419594686647/runs/528791539956287?o=7474644082064164) |

| Pattern | Pre-deployment evaluation ID | Post-deployment evaluation ID |
|---|---|---|
| Single | `b2ab7739-8d6d-4da7-a76c-60ef387167dd` | `7088c7d3-bf14-43a1-86ba-6d94a316cd94` |
| Supervisor | `45737556-99bc-4a45-811a-18922a9d3691` | `e9131508-0d5d-4fae-b317-a7d8e523318a` |

The stable supervisor endpoint passed all three standalone scenarios in [run 788666634529684](https://fevm-classic-stable-naren.cloud.databricks.com/?o=7474644082064164#job/731888306596166/run/788666634529684). Each scenario had one pre-guardrail and one post-guardrail span, no observer wrappers, matching native/AgentOPS token totals, and native cost equal to the sum of its model spans.

| Supervisor scenario | Trace | Model spans | Tokens | Native estimate USD |
|---|---|---|---|---|
| platform_prediction | `tr-4867e6e769c64fe2c4765b29525887c8` | 7 | 28778 | 0.06046500 |
| platform_stream | `tr-6c01757c460666690e9ceda7b1081f76` | 9 | 33023 | 0.05892600 |
| cost_stream | `tr-3dea37da7cb6c5ee17f32f95fedfc592` | 2 | 5029 | 0.02160300 |

The stable single-agent endpoint passed both standalone scenarios in [run 694369424656823](https://fevm-classic-stable-naren.cloud.databricks.com/?o=7474644082064164#job/1011181759735213/run/694369424656823). Each had exactly one native model span, one pre-guardrail and one post-guardrail span, no observer wrappers, matching native/AgentOPS token totals, and correct native cost aggregation. Both estimates are reported because their rate cards differ.

| Single-agent scenario | Trace | Model spans | Tokens | Native estimate USD | AgentOPS configured estimate USD |
|---|---|---|---|---|---|
| platform_prediction | `tr-3ef83d39467cd702f71557f16a18da0a` | 1 | 3077 | 0.00076215 | 0.003553 |
| platform_stream | `tr-5da9ae64fbc5a2267071072a0166d744` | 1 | 2137 | 0.00056805 | 0.002495 |

All five real endpoint scenarios and all eight isolated toggle/failure scenarios passed. Both final endpoints were READY and NOT_UPDATING with 100% traffic on the tested version before inference. Machine-readable evidence: [native-instrumentation-fevm-results.json](native-instrumentation-fevm-results.json).

## Findings preserved

- Supervisor candidate 19 was blocked by the platform-evidence gate on both configured attempts (0.667 against 0.8). Other gated quality scores and guardrail accuracy were 1.0. Combined answers used line-number placeholders; the platform worker prompt now requires inline links to exact retrieved URLs. Candidate 20 passed all pre-deployment scores at 1.0. No scorer or threshold was changed.
- The first isolated test snapshot compared an enum to an incorrect string. The corrected fresh run passed all eight cases.
- An accidental supervisor launch omitted the prompt edit after a local path error. It was cancelled before registration; the corrected run uses candidate 20.
- The first real version-16 trace fetch hit asynchronous export latency. The validator now retries only NOT_FOUND/RESOURCE_DOES_NOT_EXIST errors with a bounded wait; unrelated API errors still fail the test.
- The first version-20 standalone test exceeded its bounded readiness wait during old-entity cleanup without invoking the agent.
- A version-15 direct diagnostic stopped at the endpoint readiness assertion while configuration was still updating. It did not invoke the agent. Final checks require READY, NOT_UPDATING, and the expected version receiving all traffic.
- Single-agent version 15 completed all original pipeline tasks successfully. Version 16 additionally contains shared native OpenAI cost enrichment.

## Boundaries

Native tracing covers supported integration operations, not arbitrary code. Disabled native instrumentation does not remove customer-authored explicit spans. Remote worker internals require tracing at the worker and supported propagation. Token equality and native cost aggregation checks cover observed model calls; Genie internal usage remains explicitly unavailable. Costs derived from configured rates and MLflow catalog rates are estimates, not billing records. A real version-16 trace (`tr-d83ab79ce27a8324e09074048a7c0a61`) showed matching native/AgentOPS token totals (3,296) but different cost estimates: MLflow $0.00089354748 versus configured-rate AgentOPS $0.003991. Native cost is preserved; the existing response tags retain their configured rate card. Standardizing the two cost sources remains a separate gap. Validation requires native aggregate cost to equal the sum of model-span costs and explicitly reports whether the two estimates match; it does not force one estimate onto the other.
