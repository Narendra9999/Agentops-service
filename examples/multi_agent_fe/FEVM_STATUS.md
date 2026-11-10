> The current integration executes the original AgentOPS lifecycle notebooks.
> See [deployment details](../../src/teams/platform-intelligence/deployment/README.md). Version 19 results below used the superseded
> team lifecycle and do not validate the original notebooks. A new FEVM run is
> in progress; its outcome will be recorded separately.

# Current FEVM status

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

---

# Historical rollout records

> Current source layout: the supervisor-only implementation is under
> `src/teams/platform-intelligence`. It references existing Platform knowledge and
> Cost intelligence workers; the case worker is removed. The records below retain
> historical deployment evidence and do not imply deployment of the new team config.
> See the team README for current architecture, bindings and remaining gaps.

# FEVM deployment checkpoint

Workspace: https://fevm-classic-stable-naren.cloud.databricks.com
Profile: `fevm-naren`

| Component | Verified result |
|---|---|
| Native LangGraph supervisor | Version 6 deployed; workflow spans distinguished from agent calls; live plans and tool events streamed |
| AI Gateway model calls | `system.ai.claude-sonnet-4-6` and `system.ai.gpt-oss-120b` through `/ai-gateway/mlflow/v1/` |
| Genie worker | Explicit cost-input lookup invoked Genie successfully; existing DCP Calculator unchanged |
| Docs worker | Existing 81-row Vector Search index reused; retrieved documentation URLs returned |
| Lakebase worker | Live read and approved update passed; case version 1 → 2 with exactly one audit row |
| Exact mutation retry | Original committed receipt returned without another write or model call |
| MLflow tracing | Version 6 Genie table trace OK and controlled error trace ERROR; version 5 all-worker checks and prior audit validation retained |
| Token usage | Model spans contain reported input/output/total tokens; Genie internal model usage is not exposed |
| ResponsesAgent | UC model `fevm_catalog_naren.agentops_multi.platform_team`, version 6 |
| AgentOPS | Optional observation records existing execution; customer application retains orchestration and state ownership |
| Chat App | RUNNING as Databricks Specialists; live version 6 Genie result table and status events verified |
| Inference tables | Nine request/response rows observed in `fevm_catalog_naren.agentops_multi.fe_multi_payload` |
| Review App | Seven serving/App traces added to the existing labeling session; endpoint attached |
| Captured-trace monitor | Seven traces scored in two batches; safety and relevance means both 1.0; no application replay |
| Runtime identity | Dedicated service principal; approved resource permissions and workspace access enabled |
| Prompt management | Managed Prompt Registry disabled for this organization; immutable hash-verified artifacts used |

## Remaining gaps

The supervisor answered the initial missing-input cost question itself rather than
invoking Genie. This routing gap is retained in `deployment-verification.json`;
a separate explicitly requested Genie lookup passed. It does not establish
reliable routing for every cost question or validate a numerical cost quote.

The docs smoke made several retrieval/model calls, totaling nine instrumented model
spans. Cost and latency optimization remain to evaluate. Model context visibility
covers captured application/model inputs and outputs, not hidden reasoning or
Genie's internal model prompts. Receipt retries correctly have no model context.
No conversational checkpoint store or long-term agent memory is configured.

The outer ResponsesAgent endpoint rejects AI Gateway PII, rate-limit and usage
tracking settings as unsupported for this endpoint type in this workspace.
Inference logging alone is enabled. Dedicated Gateway model-service policies
have not been configured or enforcement-tested; shared `system.ai` policies were
not changed. Gateway usage-table ingestion remains unverified: the runtime-filtered
query returned no rows, although API responses and model spans report token usage.
Safety judge scores do not validate PII enforcement.

## Paired preproduction evaluation

Three scenarios per phase; this is a smoke comparison, not statistical evidence.

| Metric | Before | After |
|---|---:|---:|
| Routing | 3/3 | 3/3 |
| Safety judge | 3/3 | 3/3 |
| Answer correctness | 3/3 | 2/3 |
| Relevance | 3/3 | 2/3 |
| Audited mutation (one applicable row) | 1/1 | 1/1 |

The after-phase Genie clarification omitted workload facts such as run frequency.
No prompt-quality improvement is demonstrated. Production monitor safety/relevance
scores are separate from this paired evaluation and do not measure routing or
transaction correctness.

## Current streaming revision

Version 5 URI: `models:/m-d1b710e9ca7a4d1abf86230a2e8fbe75`.
Registration/packaged validation job `567845930464598` succeeded. Serving is READY /
NOT_UPDATING with version 5. Live case, docs and Genie requests streamed actual
supervisor delegated tasks, worker start/end and named tool start/end events before
the final answer. All three traces were retrieved with OK state.

The deployed App forwarded the first execution event at 1.54s, the selected plan
at 2.80s, read_dispute at 3.77s and the final answer at 6.22s. The composer clears
immediately and preserves a newly typed draft when the previous answer finishes.
Stop aborts the browser stream; it does not undo a committed write. Propagation of
disconnect/cancellation is best effort, and in-flight model/tool calls can complete.
The attached operation ID is retained for an exact retry. Browser interaction checks
verified Stop without retry and draft preservation using controlled response fixtures.
No new database mutation was needed for this UI/streaming validation.

## Prior transaction and operations validation (version 4)

Version 4 URI: `models:/m-d0d96917c8b74533a59d05b4778859b8`.
Registration job `1013439741686122` succeeded, including packaged Gateway docs
and committed receipt checks. Serving required both READY and NOT_UPDATING before
smoke requests. Six retrievable traces cover docs, initial cost clarification,
case read, mutation, exact retry and an explicit Genie lookup.

Synthetic `DEMO-SERVING-004` is investigating/version 2. Exactly one matching
audit row records runtime principal `e3bcf75e-7b34-4026-8860-d5e4423ea2b7`.
Live transaction checks also verified concurrent duplicates, optimistic conflicts,
audit rollback and payload-digest conflicts. All demo users share the synthetic
`fe-demo` tenant; this is not per-user production authorization.

Serving uses pg8000 1.31.5 / scramp 1.4.5 because its image lacks libpq.
TLS certificate validation and platform FIPS settings are retained. MLflow traces
export directly to the experiment using its public destination API, avoiding an
incompatible platform-injected inference span processor. Inference request/response
logging is configured independently. The candidate dependency closure was tested
on FEVM serverless; managed serving injects additional packages, so its environment
is not claimed to exactly match the lockfile.

Local suite: 38 passed, 6 skipped. Production validation job `621411458754902`
succeeded and scored five captured serving traces, with no prediction function.
Supplemental validation job `410350055142480` succeeded, adding and scoring the
Genie and App traces. Daily monitor job `629083761269610` is confirmed UNPAUSED
at 07:00 UTC. Version 4 is confirmed READY / NOT_UPDATING. Monitor batches cap at 1,000 traces in 24 hours;
a crash after scoring but before tagging can duplicate assessments on retry.

## Workspace links

- [Chat App](https://fe-platform-team-fevm-7474644082064164.aws.databricksapps.com)
- [Serving endpoint](https://fevm-classic-stable-naren.cloud.databricks.com/ml/endpoints/fe-platform-agent-team)
- [MLflow traces and evaluations](https://fevm-classic-stable-naren.cloud.databricks.com/ml/experiments/4384417075366092)
- [Review App](https://fevm-classic-stable-naren.cloud.databricks.com/ml/review-v2/6c9dc69370944bab8e8b8e450749e246/tasks/labeling/47902299-5e82-4186-b273-43bc6cd0983e?o=7474644082064164)
- [Daily monitor job](https://fevm-classic-stable-naren.cloud.databricks.com/?o=7474644082064164#job/629083761269610)

`VALIDATION.md` retains the intermediate deployment failures and corrections.

## App timeout correction (2026-11-09)

The logged 20:49:10 UTC failure was the legacy `/api/chat` SDK call reaching
its default 60-second read timeout. An earlier cost-estimation trace
`tr-9b6bd6ecb6b24186247c79c2ad8e7091` completed OK in 60.924 seconds
(worker: 52.72 seconds); this illustrates why the outer timeout is insufficient,
but is not asserted to be the exact failed browser request.

Both App routes now consume the same authenticated serving stream, with a
20-second connection timeout and 240-second read inactivity timeout. Heartbeats
keep the transport active; 240 seconds is not an overall execution deadline.
The legacy JSON route collects the final result and preserves trace metadata.
Transport timeouts return HTTP 504; invocations are never automatically retried.
Existing open clients can continue to use the legacy route, although refreshing
loads the UI with live activity and Stop support.

A read-only Genie request through the deployed App completed in 21.57 seconds,
with actual supervisor delegation and `genie_query` completion events; persisted
trace `tr-6045c52e51c778d79decad7d5e1b57f4` is OK. This smoke check does not
validate a live request longer than 60 seconds or numerical estimate accuracy.

Post-fix deployment `01f1dc8366581167bc482c788271ef81` is RUNNING / SUCCEEDED.
Both API paths were checked after activation:

| Check | End-to-end seconds | Persisted OK trace |
|---|---:|---|
| Legacy JSON route, Genie | 24.70 | `tr-ce692432ca8e3bcae88276b4156fc56b` |
| Legacy JSON route, read-only case | 10.78 | `tr-a6aa28693399e9f01c3c55fcb50e9976` |
| Streaming UI route, Genie | 27.13 | `tr-75487738a65c7c4ee1f351c485a44bc4` |

The streaming check observed supervisor delegation at 3 seconds, Genie start at
5 seconds, completion at 18 seconds and synthesis at 24 seconds. Evidence:
`validation-results/app-legacy-timeout-fix.json` and
`validation-results/app-genie-timeout-check.json` (local ignored diagnostic files).

## Workflow labels and Genie failures (2026-11-09)

The display name is **Databricks Specialists**. New model traces use
`databricks_specialists.turn` → `workflow.execution` (CHAIN) →
`workflow.databricks_specialists` (CHAIN), with sibling `supervisor` and
`cost_estimation` nodes. The cost worker contains an explicit `genie.query`
TOOL span with Genie space, conversation, message and status attributes.
Existing historical trace names are unchanged.

Investigation distinguished these supplied traces:

- `tr-f019d53b798cd3de94a39f24570cedc3`: successful, 26.276 seconds;
  final text includes a Markdown estimate table.
- `tr-ad94d01a79875df8c9e4c3428227ef73`: 194.543 seconds;
  inner execution/worker spans failed because Genie exceeded its 180-second
  wait while still ASKING_AI. This was not a table/chart rendering exception.
  The old outer stream span appeared OK because it emitted an error event.

The revised stream marks execution failure on the trace and sends a sanitized
failure code, trace ID, Genie status and available conversation identifiers.
The UI distinguishes Genie timeout from generic execution failure and offers
trace/conversation links. Case mutation warnings are included in the server's
failure message only when the request carries a mutation approval. No automatic
retry or second Genie conversation is started after timeout. Genie can still
outlive the bounded wait; this does not remediate Genie service latency.

Successful responses preserve bounded query tables (up to 10 attachments,
100 rows, 30 columns per table) in `custom_outputs.genie_result`, separately
from supervisor prose. The UI escapes all cell values and renders tables directly.
Genie native visualization metadata is not an embeddable chart specification;
charts remain available through the Genie conversation link. The integration
requests visualization support but does not fabricate charts from arbitrary
numeric columns. The API returns the available result chunk; full results and
native visualizations remain in Genie. No existing space/index is modified.

### Version 6 live validation

Serving version 6 is READY / NOT_UPDATING; App display is Databricks Specialists.
Model URI: `models:/m-3d6d29cf58d645f98fd08dcf537ada07`.

A read-only five-row AWS rate-card query through the deployed App completed in
41.99 seconds. Persisted OK trace: `tr-bbea0f36d4694f50803df42839135de6`.
It contains `workflow.execution`, `workflow.databricks_specialists`, and
`genie.query`, with no `agent.supervisor` wrapper. The App received a structured
five-row table plus Genie identifiers, as well as live FETCHING_METADATA,
ASKING_AI, PENDING_WAREHOUSE and COMPLETED status events.

A deliberate invalid-role request against the serving transport produced the
expected input-validation error, a sanitized stream failure and an ERROR trace:
`tr-964cbea210fb5207a1faa5f0bf0ff3fb`. This trace is tagged as an expected
error-boundary validation failure. App routes reject that role before invocation.
No tool ran or case update occurred. This checks the live generic error boundary;
the exact Genie timeout branch was exercised with a controlled local SDK waiter,
not a new three-minute production timeout. The original failed conversation was
not restarted, and its eventual completion was not verified.

Evidence: local ignored `validation-results/v6-live-genie.json` and
`validation-results/v6-error-boundary.json`. Native chart embedding is not claimed;
the UI links to the Genie conversation for native visualizations.

## App history and Genie charts — November 9 update

The App now stores per-user conversation snapshots and preferences in the existing
Lakebase database, in isolated `app_history` tables. The App service principal has
SELECT, INSERT and UPDATE permission only on these history tables. API ownership
uses the authenticated `x-forwarded-user` subject; a supplied email does not select
another user's history. Optimistic revisions prevent silent overwrites across tabs.

New conversation retains prior threads in the sidebar. Reopening restores the saved
answers, Genie tables/charts and trace links without executing agents. Mobile users
open the Conversations drawer. Preferences include display name and default bar/line
chart type. Saved history starts with this release; prior cleared browser state was
not persisted. The UI saves the submitted question before invoking serving, then
saves the received answer. Closing the page or stopping the stream can leave a
question without its final answer; saved history does not poll or replay execution.
Approvals and private reasoning are not stored in conversation snapshots.

Genie links now use `/genie/rooms/{space}/monitoring?o=...&mc=...&m=...`, matching
this workspace's native route. The public message API for the supplied conversation
exposes a visualization query reference, not its chart configuration. The App draws
bar/line charts directly from returned query rows, with selectable axes; this does
not reproduce Genie's native chart settings. Charts appear before expandable tables.

Validation: 48 Python tests passed, 6 environment-dependent tests skipped; four Node
chart/link tests passed. Local browser checks used the supplied conversation's actual
three-row query result with mocked transport: exact native link, chart controls,
new/reopen, saved chart preference and mobile drawer passed. Real Lakebase checks
verified save/reload, user isolation, stale revisions and preferences. The deployed
App returned 200 for identity, history, preferences and assets; a synthetic thread
was saved and reloaded through its actual authenticated API and then removed.

### Grouped cost-scenario charts

Trace `tr-8d44ddd7d5a050305fbf19551df9fa06` completed OK and includes
EXPECTED, HIGH and LOW rows with all three measures in the reported Genie chart.
The earlier App renderer selected a single numeric measure, which explained the
visual mismatch. Cost tables containing `scenario`, `recurring_monthly_usd`,
`annual_usd` and `first_year_usd` now default to a grouped-column presentation:
blue monthly recurring, orange annual and green first-year cost, with Scenario / USD
axes, legend and the title Cost Estimate by Scenario. This explicit schema preset
takes precedence over the generic bar/line preference. It is an App rendering of
query data using the requested layout, not an import of Genie's native chart spec.

Users can select multiple numeric measures, switch between grouped columns,
horizontal bars and lines, and download SVG with the title and legend included.
Existing saved conversations render through the updated chart code when reopened.
Seven Node tests passed. A browser test with the actual trace's saved result and
mocked transport verified nine bars, colors/series, measure controls, SVG export,
saved-thread reopening and mobile page width. No agent or SQL query was rerun.

### Interactive query-data charts

The shipped chart implementation now uses a locally served Plotly.js basic 3.3.0
bundle (MIT licensed) for hover values, drag zoom, pan controls, reset, legend
series toggles and image export. It uses existing `genie_result.tables` rows and
columns from serving; no direct Genie chat or query rerun was introduced. The
cost-scenario preset retains the requested three grouped series, colors and axes.
Repeated category labels receive distinct row positions so values do not aggregate;
null values remain missing. Generic charts still expose selectable axes/measures.
The SVG renderer remains a fallback if Plotly fails to load or render.

A native PNG API was investigated and successfully downloaded the existing Genie
chart, but was removed before App activation following the interactive-chart
requirement. There are no PNG chart routes or native-image scripts in the App.
The unused investigation-only chart-access table was removed. The public download
API returns an image, not an interactive chart configuration; Plotly uses the App's
explicit schema-based defaults instead of claiming automatic native spec import.

Eight Node tests and 15 App API tests passed. A local browser check with real rows
from the supplied trace and mocked transport verified three series/nine bars,
actual hover value 72.08, mouse drag zoom, clicking the legend to hide a series,
line switching, restored saved conversation and no whole-page mobile overflow.

### ResponsesAgent usage accounting

Version 7 passed packaged FEVM validation with provider-reported request totals
and separate supervisor/worker/model breakdowns. The collector counts each model
callback run once, including structured supervisor routing. Standard response
`usage` and final `response.completed` events carry totals; the App preserves them
without duplicating final output items. AgentOPS trace tags include prompt,
completion, total, cached and reasoning tokens, request count, coverage, indexed
agent/model fields, and configured-rate cost estimates. Rates are configurable;
unknown usage/prices are explicitly incomplete or unavailable. Genie internal
usage remains unmetered when its API supplies no token counts. These request
aggregates must not be added to MLflow child-span or AI Gateway totals again.

### Conversation names and Genie agent mode

The App now offers a pencil button beside each saved conversation. Names are
persisted independently of message content and retained on follow-up saves and
reopen. Rename is scoped to the authenticated owner and uses optimistic revision
checks. The deployed API passed synthetic rename/reload, unchanged-message and
stale-update checks; the test row was removed. Browser checks with mocked storage
passed rename, cancel, follow-up preservation, reopening and mobile use.

The earlier cost worker used the classic chat `start_conversation` API. The new
worker explicitly submits `execution_mode: STANDARD` and `enable_viz: true` to
`/api/2.0/genie/agents/{space_id}/responses`. It consumes terminal SSE events and
reports public execution statuses without exposing private reasoning text. No
chat fallback or automatic query retry is used. Query-result Markdown previews
are adapted into bounded rows/columns for existing interactive Plotly charts;
SQL is not re-executed for rendering. Agent-mode conversation links use the
`/genie/rooms/{space_id}/chats/{conversation_id}` route; historical chat-mode
results retain their monitoring links. The agent API was verified on FEVM and
returned no internal usage counts, so Genie remains explicitly unmetered.

The initial combined usage-breakdown trace tag exceeded Databricks' 255-byte
value limit and caused trace export rejection in version 7. The correction uses
`agentops.usage.agent_model.<index>.<field>` tags, leaving full structured metrics
in response custom outputs. Regression checks cover all four agent roles and
assert the tag-value limit. Rejected historical traces cannot be backfilled from
this change.

Version 9 is deployed and READY. The final authenticated App smoke test routed
straight to cost_estimation, streamed AGENT_STARTED / EXECUTING_QUERY /
QUERY_COMPLETED / PREPARING_ANSWER / COMPLETED, and returned two query previews.
The scenario preview contains three rows and the exact recurring_monthly_usd,
annual_usd and first_year_usd columns used by the three-series grouped chart.
Trace `tr-dd8d9e5c265faec07d29d1490b65b8df` was retrieved and its Genie TOOL span
confirms `genie.mode=agent`, `genie.execution_mode=STANDARD` and completed status.
Response usage, AgentOPS tags and MLflow totals all match: 5,550 input + 1,184
output = 6,734 observed tokens. Genie is explicitly unmetered; USD 0.03441000 is
only the configured-rate estimate for observed model calls, marked partial.
Final local suite: 62 passed / 6 environment-dependent skips, plus 9 Node tests.

### Native MLflow cost fallback

The older trace `tr-549316cb188e8acbd622151efc1aed9f` has native costs on
13 GPT-OSS model spans but none on six Claude spans. Its USD 0.00195612725
native total is therefore incomplete. This does not establish that GPT-OSS
returned dollar costs in its response: MLflow can calculate native costs from
its own pricing catalog. The later trace
`tr-354c82c2b33609e67418b34d1e5c3546` contains two `system.ai.*` Claude calls
with tokens but no native cost fields.

The new processor preserves existing `mlflow.llm.cost` and uses explicit
request-scoped rates only when native cost is absent. It prices LLM/CHAT_MODEL
spans, not parent wrappers, preserving other installed span processors and the
AI Gateway model names. Native totals and configured-rate AgentOPS totals may
differ when an existing native cost uses another rate. Genie-internal usage
remains unavailable; observed costs are estimates, not invoices.

FEVM serverless run `489319210036894` passed packaged-model validation and
registered version 10 (`models:/m-71cd849b44084d618a4918bbb90185eb`).
Trace `tr-15556b09613d13a3e19c1017abd805df` has four priced model calls:
two Claude and two GPT-OSS. Native input USD 0.006923 + output USD 0.003064 =
USD 0.009987, matching the configured estimate. Tokens: 2,743 input + 284 output =
3,027. Parent spans have no duplicated costs. The smoke test reads the synthetic
case without mutation. Local Python tests: 65 passed, 6 environment-dependent
skips; all 9 Node chart/link tests passed. Deployment verification follows.


Version 10 reached READY, but live App trace
`tr-ee33db2b5c690f15c476bc995258fed6` had token totals and AgentOPS estimates
without native span costs. The synchronous packaging test did not cover the
serving/streaming lifecycle. Processor installation now reconciles the shared
processor at request start under a lock, preserving other processors and keeping
rates request-scoped. The replacement packaging test explicitly clears processor
configuration and uses predict_stream. A concurrent reset/reinstall regression
check was added; 66 Python tests passed with 6 environment-dependent skips.


FEVM run `335066841702445` passed the reset-and-streaming packaging check and
registered version 11 (`models:/m-abfaa4bdccfa4a48bee0bd3d478c4a89`).
Trace `tr-8541d33e981da4332e3b928e00a302ca` contains four costed model calls
(two Claude, two GPT-OSS), with native input USD 0.006923 + output USD 0.003104 =
USD 0.010027. This matches the configured estimate for 3,021 observed tokens,
with no costs duplicated on parent spans. Deployment/live verification pending.


The native cost fix is deployed as version 11 and was verified through the live
App. Trace `tr-0dc357ec936fce9b4f7bcd92a6361970` contains four priced model spans
(two Claude, two GPT-OSS). Native input USD 0.006905 + output USD 0.003104 =
USD 0.010009; 3,015 tokens match the response and AgentOPS tags. All costs were
configured-rate fallbacks and no parent span was priced again. This read-only
check validates the preceding deployed architecture; the supervisor-only team
refactor is separately prepared for review with external worker bindings.
