> Current verified endpoint: `fe-platform-intelligence`, version **3**.
> Original pipeline and all live compatibility/memory checks passed.
> The version 2 results below are historical; see the follow-up section for version 3.

# Playground compatibility fix — 2026-11-10

Endpoint `fe-platform-intelligence` serves model version **2**, with 100% traffic
and registry alias `champion`. Endpoint state verified READY / NOT_UPDATING.

[Original AgentOPS pipeline run](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/701419594686647/runs/211536938707897?o=7474644082064164)

## Root cause and behavior

Ordinary Playground requests do not carry the App's signed
`custom_inputs.application_context`. Previously the memory-enabled supervisor
verified an empty envelope, raising `Invalid application context signature`.
The stream error then appeared as a Playground response-schema error.

The team explicitly configures `unsigned_context_policy: stateless`. Requests
without application context can use the supervisor and workers without stored
memory or App history. Supplied conversation messages remain available. Signed
App requests retain scoped Lakebase memory tools. A supplied invalid, empty,
expired or mismatched context remains rejected; it never falls back to stateless.
Unsigned owner IDs cannot select a memory namespace. Settings are copied per
turn so stateless calls cannot disable memory for concurrent signed calls.

`custom_outputs.memory` and trace tag `agentops.memory.mode` identify the mode.
Progress events include a Responses envelope; completed output events include
output indices and the response completion identifies its ID and status.
The Settings default remains `reject`; the team opts into stateless behavior.

## Live verification

Passed three unsigned API scenarios: ordinary response, Platform retrieval stream
with documentation citations, and Genie cost-input clarification. MLflow Responses
schema validation passed. A forged signed context was rejected. Successful
unsigned requests reported `{"mode":"stateless","persistent":false}`.

Passed four signed scenarios: Lakebase write and database readback, recall in a
fresh conversation, Platform retrieval, and Genie cost routing. All seven
successful traces completed OK in experiment **240926829857302**.
These checks used Playground-style API requests; the browser Playground UI was
not operated.

The signed Platform request took **290.8 seconds**. Latency remains unresolved.

## Original pipeline result: FAILED quality gate

Registration, pre-deployment evaluation, deployment, and smoke test passed.
The endpoint remains live; the full pipeline must not be described as successful.

- Pre-deployment: all six configured quality scores **1.0**, guardrails **100%**.
- First post-deployment attempt: Platform evidence **0.667**, below **0.8**,
  on the combined Genie + serverless Jobs scenario; other configured gates 1.0.
- Automatic retry: routing **0.333**, below **0.8**. The audit rows for the
  standalone Platform and combined requests contain response `None`, so this
  score does not establish incorrect worker selection. The failure to obtain
  complete evaluation responses needs investigation. Remaining reported means
  were 1.0, but missing responses limit that result.
- Guardrails: **100%** on both post-deployment attempts.

No thresholds were lowered. Evaluation IDs:
`48e6139d-eca4-417a-a567-be8c3781d0b4` (first attempt),
`d18c2c3b-ec1f-41e4-ae6d-cab4a215329a` (retry).

Local verification passed 103 example tests, 74 focused shared-framework tests,
strict bundle validation, and seven final context/stream tests. The smoke helper
now verifies traffic routing instead of assuming the first listed entity is live.

App deployment remains deferred; its existing endpoint remains unchanged and
healthy. Advanced AI Gateway limitations documented in FRESH_DEPLOYMENT.md still
apply. Sanitized evidence is retained locally under ignored validation-results/.

## Follow-up: expired Playground example and routing schema

A subsequent Playground request supplied `application_context` from the saved
model example. The original registration notebook used `TeamRuntime.request()`
to create that example, persisting a signed evaluation envelope that expired
five minutes later. Fresh unsigned API checks did not exercise this prefilled
Playground path, so the initial validation missed it.

Registration now uses `TeamRuntime.input_example()` separately from authenticated
evaluation requests. The platform-intelligence adapter publishes an unsigned
example when stateless requests are enabled, or omits the example for signed-only
configuration. The version **3** UC artifact was downloaded and verified to
contain only `input`, with no signed context. Existing Playground tabs may retain
the old payload: remove `custom_inputs.application_context` or load the new
example. Invalid or expired signed contexts continue to be rejected.

A separate live failure occurred when a delegation omitted the unused `answer`
field. Routing now defaults unused `task` and `answer` fields to empty strings,
while existing validation still requires a non-empty task for delegation and a
non-empty answer for finish. Regression tests cover both missing-unused-field
success and missing-required-for-action rejection.

Implementation commit: `fa796f2`. Validation: 106 agent tests, 15 shared-runtime
tests, strict FEVM bundle validation; the 23 focused graph/context tests passed.

[Version 3 original pipeline run](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/701419594686647/runs/777025154117943?o=7474644082064164).
Version 3 deployment and the original smoke test passed. Live validation passed
all three unsigned API scenarios and four signed memory/worker scenarios. The
actual published input example was invoked after its former expiry window, and
the exact user-reported 30-runs/month question completed successfully
(`tr-fafe396916bebf7fc2e4e61e9437885a`). The model holds `champion` and receives
100% traffic. The original pipeline completed **SUCCESS**. All six configured quality scores
were **1.0** before and after deployment, and guardrail accuracy was **100%**
in both phases. The endpoint settled READY / NOT_UPDATING. Earlier version 2
failures above remain historical evidence; this result applies to version 3.

Version 3 evaluation IDs: `29df3035-2244-44b8-927d-a588ccf5cb4f` (before),
`e6424d9b-326e-4cfc-8a6f-655258e00f48` (after). Published-example trace:
`tr-8461962ac7eafb5d01c6f7ea21af5e4c`. Successful signed regression traces were
verified in experiment 240926829857302. The browser Playground UI was not operated.

Version 2's retry failures were subsequently traced to the evaluation client's
240-second HTTP read timeouts on Platform and combined requests, explaining the
`None` audit responses. Version 3's signed Platform check took 67.3 seconds, but
a single faster check does not establish a general latency improvement.
