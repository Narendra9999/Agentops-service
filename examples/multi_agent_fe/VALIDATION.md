> Current source layout: the supervisor-only implementation is under
> `src/teams/platform-intelligence`. It references existing Platform knowledge and
> Cost intelligence workers; the case worker is removed. The records below retain
> historical deployment evidence and do not imply deployment of the new team config.
> See the team README for current architecture, bindings and remaining gaps.

# Validation record

## Agent-managed memory and embedded Platform knowledge — current validation

- Agent/App suite with disposable local PostgreSQL: **69 passed, 1 skipped**.
- Framework resource/text contracts: **7 passed**. Chart checks: **9 passed**.
- Real PostgreSQL verified cross-session recall, owner isolation, pending-pattern
  exclusion, explicit approval, expiry, deletion/audit and concurrent deduplication.
  This database was temporary and local, not FEVM Lakebase.
- FEVM run **280309112395701** passed with real Claude/GPT-OSS and the existing
  documentation index. Trace: `tr-58fe2655f10425a5400b259c5143abb2`.
  Memory storage was a process-local stub: model tool use and recall across new
  graph instances passed; this is not evidence of durable FEVM memory.
- Gateway rejected `parallel_tool_calls`; the graph enforces one call per step
  without that unsupported request parameter. Required tool choice was verified.
- Persistent FEVM memory tables, secret scope/ACL and runtime grants were blocked
  by automatic approval review pending explicit approval. They have not run.
  No new worker endpoint or memory-enabled model/App rollout has occurred.
- The current implementation uses structured facts/pattern records. A Markdown
  file-curation interface is a separate interface decision; the storage backend
  can persist file paths/content/versions in Lakebase or use actual Volume files.

## Supervisor-only AgentOPS integration — current local validation

- Agent/App suite: **60 passed, 1 skipped** (optional local database coverage).
- Framework team/resource and text-extraction checks: **6 passed**.
- Interactive chart checks: **9 passed**.
- Headless browser: conversation rename, follow-up title preservation, reopen,
  mobile rename, composer clearing and Stop passed against mocked APIs.
- Staged source scan found no matching access-token or private-key patterns.
- Root-wide checks remain blocked by existing issues: a bare `test` expression in
  `create_chunk.py`, missing local pandas, and a production-monitor resource filename
  mismatch in an existing test. Root bundle validation requires the Jenkins template
  render (`!sp_id` is not valid standalone bundle YAML).
- Existing worker calls and answer quality require binding the real knowledge
  endpoint. These local checks do not establish deployed-worker authorization.

## FEVM standard ResponsesAgent packaging contract

Serverless run **52486441721419** completed successfully using profile `fevm-naren`.
Model artifact: `models:/m-d3b3acc40dba4a58a04e0b366ecccea1`.
The actual AgentOPS `CostTrackingResponsesModel` loaded the team's `ChatModel`
and existing supervisor ResponsesAgent. Assertions passed for public streaming
progress, standard usage and trace ID, input guardrails preventing worker calls,
and post-guardrails filtering sensitive output before final events were released.
Workers were stubbed: **zero workers deployed**. This proves packaging and framework
contracts, not live worker quality, permissions or external trace propagation.

## Historical validation

## Local checks

Combined suite: **26 passed**, Python 3.11, disposable PostgreSQL database.
One dependency deprecation warning from Starlette's HTTPX TestClient integration.

```sh
FE_MULTI_TEST_DSN='postgresql://USER@127.0.0.1:PORT/postgres' \
PYTHONPATH=src:examples/multi_agent_fe python -m pytest examples/multi_agent_fe/tests -q
```

- Dispute contracts: valid transitions, rejected transitions, required approval fields,
  and idempotency digests.
- Real PostgreSQL: duplicate and concurrent duplicate delivery commit once; stale
  revisions fail; failed audit insertion rolls back state; tenant scoping rejects
  cross-tenant reads and mutations.
- Optional AgentOPS observer: disabled mode, telemetry startup/export failure,
  preservation of application errors, nested caller lineage and thread isolation.
- FastAPI: missing authentication, cross-origin writes and instruction-role
  injection rejected; a valid chat request invokes the endpoint once without
  forwarding a browser-supplied identity as a trusted principal.

These tests do not exercise the LangGraph model loop or remote Databricks services.

## Native graph update

Latest local suite: **28 passed, 5 PostgreSQL tests skipped**, with one dependency
deprecation warning. The database tests passed in the earlier live-database run.

The supervisor package has been removed. Native `StateGraph` routing tests execute
multi-worker handoffs, synthesis, no-delegation answers, bounded loops and invalid
routing rejection. A prebuilt agent test exercises a real tool-call loop with a
local deterministic model. These tests make no remote model calls.

The current local graph test environment contains LangGraph 0.6.11,
langgraph-prebuilt 0.6.5 and langgraph-sdk 0.2.12, but uses langchain-core 1.7.6
because the mirror lacks the requested 0.3.81. This is separate from the customer
candidate in `requirements.in`; it does not establish full candidate compatibility.
The candidate pins databricks-langchain 0.8.0, core 0.3.81, MLflow 3.11.1 and
FastAPI 0.123.0. Full resolution failed because core 0.3.81 is absent from the
mirror; public PyPI connections were refused. No dependency lock was generated.
The earlier 26-test run included five live PostgreSQL tests. Subsequent runs skip
those five unless FE_MULTI_TEST_DSN points to a disposable database.

## FEVM validation remaining

The selected workspace and bootstrap evidence are recorded below. The following
end-to-end checks remain pending:

1. Model availability, full supervisor routing, Genie SQL/results and docs retrieval.
2. Lakebase OAuth, network reachability and least-privilege runtime role.
3. Prompt Registry registration and packaged ResponsesAgent loading/prediction.
4. Three paired before/after scenarios with persisted evaluation evidence.
5. Model Serving deployment and AI Gateway PII, rate limits, inference rows and usage.
6. Review App session, daily monitor execution and enabled schedule.
7. Strict bundle validation, Databricks App deployment and browser smoke test.

The UI approval mechanism is for a shared synthetic demo tenant. Production
authorization requires trusted identity propagation and application-specific
permissions. Telemetry payloads require their own capture/redaction policy;
gateway PII enforcement alone does not sanitize every internal trace.

## FEVM execution evidence

Selected profile: `fevm-naren`, host
`https://fevm-classic-stable-naren.cloud.databricks.com`.

- Bootstrap job run `640242267406577`, task `1061037323529852`: **SUCCESS**.
- Installed exact candidate: LangGraph 0.6.11, prebuilt 0.6.5, SDK 0.2.12,
  langchain-core 0.3.81, databricks-langchain 0.8.0, MLflow 3.11.1, FastAPI 0.123.0.
- Seven graph/helper tests passed on serverless compute. ResponsesAgent and
  ChatDatabricks imports succeeded.
- Live docs retrieval succeeded against
  `fevm_catalog_naren.agentops.databricks_docs_index` (81 indexed rows).
- Both requested model endpoints report READY.
- Isolated Lakebase project `projects/fe-agentops-multi`, primary endpoint in its
  default branch: initialized dispute schema and seeded `fe-demo/DEMO-001`.
  Compute is 0.5 CU with 300-second idle suspension.
- Strict bundle validation passed against FEVM with placeholder endpoint/experiment
  bindings; no App deployment is implied by validation.
- Integration run `803698262588184` failed at prompt registration: the workspace
  returned `FEATURE_DISABLED: Prompt registry is not enabled`. Retries were cancelled.
- Run `238360293828268` failed on an existing-schema check; setup was corrected.
- Current integration run `171765661045903` uses the explicit hash-verified prompt
  artifact fallback. It failed before evaluation with SIGABRT while importing the
  native psycopg adapter; retries were cancelled. Driver isolation is in progress.
  This fallback does not implement managed Prompt Registry.
- Existing DCP Calculator Genie agent is reused without modification. Cost scenario
  checks clarification for incomplete workload inputs; synthetic USD176 is no
  longer asserted against that existing agent.
- Dedicated serving identity/credential creation was rejected by automatic approval
  review pending explicit user approval. No identity or secret scope was created.

Earlier local dependency limitations above are historical; exact candidate graph
compatibility is now established in FEVM. Full endpoint integration remains pending.

Live FEVM Lakebase transaction suite: **5 passed in 16.29s**, including concurrent
duplicate delivery, stale write rejection, audit failure rollback and tenant isolation.

MLflow experiment created: `4384417075366092`. Setup artifact run
`30ccfb8e36c84cbb8938df61849df42e` completed and stores the dependency snapshot and
prompt/runtime artifacts. No paired evaluation scores have been produced yet.

Driver diagnostic job `266479502086016`, task `1036947793287283`: **SUCCESS**.
The psycopg-binary 3.3.8 child process aborted with
`OpenSSL internal error: FATAL FIPS SELFTEST FAILURE`. The pure-Python psycopg
3.3.8 adapter loaded successfully against system `libpq.so.5` (version 160015).
The implementation now defaults to `PSYCOPG_IMPL=python` and omits the binary
extra. FIPS settings are not disabled or changed. Model Serving still needs its
system-libpq availability verified in the packaged endpoint environment.

Dependency export job `394612485824227`, task `1121829002257132`: **SUCCESS**.
`requirements.lock` now contains 164 exact package versions derived from the
installed dependency closure on FEVM Python 3.12.3. This is an environment snapshot,
not yet a clean Model Serving installation test.

The first live prediction exposed Claude's assistant-prefill restriction after a
worker return. The native supervisor now appends an application-owned user turn
when resuming from worker evidence. All seven local graph/helper tests pass after
this fix. Current integration run: `398173878154163`; outcome pending.

Partial run `398173878154163` captured successful docs and dispute traces.
Trace `tr-549316cb188e8acbd622151efc1aed9f` verified a committed investigating/version2
mutation with one audit row. It also exposed repeated supervisor delegation after
completion and empty text extraction by the evaluation adapter. The run was stopped
for correction; no final paired metrics are claimed. The router now receives an
explicit completed-worker ledger and removes completed agents from its output enum.
Responses text is extracted from serialized content dictionaries. Current local
suite: 31 passed, six skipped (five live database tests and one full-MLflow type
test unavailable in the lightweight client environment).

## Final checkpoint for this deployment stage

- Citation-preserving paired evaluation `444199922956807`: SUCCESS; all six
  predictions completed with the intended single worker and nonempty answers.
- Before metrics: routing, correctness, relevance and Safety all 3/3; audit 1/1.
- After metrics: routing and Safety 3/3; correctness/relevance 2/3; audit 1/1.
  The failing case is upstream Genie clarification that omitted run frequency.
  No prompt-quality improvement is claimed from this three-case sample.
- Review setup `548605307479893`: SUCCESS on retry after a library-install attempt
  failed; a labeling session with three traces was created.
- Model registration `69904416915088`: SUCCESS. UC model
  fevm_catalog_naren.agentops_multi.platform_team version 1 loaded and predicted
  with a literal docs URL. This ran under the notebook identity, not Model Serving.
- Monitor job 629083761269610: created PAUSED and bound to the example DAB resource.
  Manual run `389173406827940`: SUCCESS / no_data, correctly excluding evaluation
  and packaging traces. Production scoring remains unvalidated.
- Model Serving, Gateway enforcement/inference tables and chat App remain pending
  explicit runtime identity/credential approval. No identity or secret was created.

FEVM_STATUS.md is the current concise checkpoint; earlier entries above preserve
the validation history and intermediate failures.


## Deployment identity checkpoint

User-approved runtime identity and 30-day secret created successfully. Secret stored only in Databricks Secrets. Runtime grant script execution rejected by automatic approval review for lack of explicit privilege-scope approval; none of its grants ran. Strict FEVM bundle validation and git diff --check passed. Serving, Gateway and App remain undeployed.


## Approved resource grants and deployment

- Approved resource grants completed, including the isolated Lakebase OAuth role.
- Runtime API calls fail before resource authorization because the identity has no workspace access entitlement. Automatic approval review rejected adding workspace-access; no entitlement change was made.
- Serving endpoint fe-platform-agent-team created with model version 1; NOT_READY / IN_PROGRESS, container creation pending.
- Outer endpoint Gateway separately rejects usage tracking, rate limits and AI Guardrails as unsupported. Inference-table logging alone was accepted and persisted; actual rows remain unverified.
- Bundle deployed; existing bound monitor updated and kept paused. App fe-platform-team-fevm RUNNING. Authenticated GET /health and / both returned HTTP 200. End-to-end chat remains unverified.
- Existing Review App endpoint attachment succeeded in run 85266302557611.
- User corrected model invocation to use Gateway model names. Verified direct owner-identity call through /ai-gateway/mlflow/v1/chat/completions to system.ai.gpt-oss-120b returned OK and usage: 106 prompt, 29 completion, 135 total tokens.
- Source clients now use system.ai.claude-sonnet-4-6 and system.ai.gpt-oss-120b via Gateway; SDK refreshing authentication retained. Local suite 32 passed, 6 skipped. FEVM registration/smoke run 369338691345341 in progress.
- System model-service effective privileges are inherited from account users. Attempt to grant on protected system catalog was denied; no system grants/policies changed. Dedicated per-demo Gateway policies remain unconfigured.

Gateway registration run 369338691345341 completed SUCCESS: UC version 2, packaged supervisor/docs worker/retrieval/citation smoke passed through Gateway under notebook identity. Serving still version 1 provisioning; runtime entitlement and serving version switch remain pending.

Workspace-access explicitly approved and enabled. Runtime identity, Gateway Claude (15 total tokens), Gateway GPT-OSS (132 total tokens), docs retrieval and Lakebase case read passed. Version 1 READY; version 2 update submitted. Gateway usage-table query succeeded with zero matching rows at initial check.


## Version 2 live serving failures and correction

Version 2 reached READY / NOT_UPDATING. The docs and cost requests returned answers, but case tools failed because the serving image lacks libpq. Mutation and duplicate attempt truthfully reported failure; no successful write is claimed. All five responses returned null trace IDs. Runtime logs confirm tracing was not active; serving needs explicit ENABLE_MLFLOW_TRACING plus experiment configuration.

The correction selects pg8000 1.31.5 with scramp 1.4.5, a pure-Python driver using verified TLS and platform crypto, without changing FIPS settings. scramp 1.5.0 was incompatible and was pinned back before deployment. Live runtime checks passed concurrent duplicate delivery, stale revision rejection, audit rollback, case read and exactly one audit row on isolated fe-driver fixtures. Registration of version 3 is in progress.


## Version 3 serving verification

Version 3 returned answers with non-null trace IDs and successfully read and updated DEMO-001. Direct database verification confirmed status investigating, version 2 and exactly one audit row with the dedicated runtime principal. The duplicate agent request made no additional write but returned a misleading version-conflict answer instead of the committed receipt.

Persisted MLflow traces were NOT found. Logs show `InferenceTableSpanProcessor` lacks `_shutdown`, caused by MLflow 3.11.1's custom processor under the newer platform-injected OpenTelemetry. Source correction uses public direct experiment destination selection; no dependency monkeypatch or FIPS change. The next package also checks exact retry receipts before invoking an LLM. Live runtime receipt/digest checks and the local 32-pass suite succeeded.


## Version 4 deployment and production validation

- Version 4 registration job 1013439741686122 SUCCESS; URI models:/m-d0d96917c8b74533a59d05b4778859b8. Serving READY / NOT_UPDATING confirmed.
- Five initial serving requests returned answers and retrievable OK traces. Docs/disputes worker spans include caller lineage, model inputs and reported token usage.
- Initial missing-input cost question was answered directly by supervisor, failing the expected Genie route. This result is retained in deployment-verification.json; an additional explicit Genie lookup invoked the worker successfully. No blanket routing success or numerical cost validation is claimed.
- DEMO-SERVING-004 persisted investigating/version 2 with exactly one matching audit row under runtime principal e3bcf75e-7b34-4026-8860-d5e4423ea2b7. Exact retry returned the committed receipt with no model spans and no second write.
- Authenticated App API read against version 4 returned HTTP 200 and the persisted version 2 state.
- Production validation 621411458754902 SUCCESS: five captured traces added to existing Review App session and scored; safety/relevance means 1.0. Supplemental run 410350055142480 SUCCESS: two more captured Genie/App traces added and scored with the same means. No predict function or mutation replay was used.
- Nine rows observed in the configured outer endpoint inference table. Gateway runtime-filtered usage query returned no rows; ingestion remains unverified.
- Bundle strict validation and deployment passed. Existing monitor job 629083761269610 confirmed UNPAUSED at 07:00 UTC; no duplicate monitor/session created.
- Remaining limits: per-demo Gateway PII/rate policies not configured or validated, managed Prompt Registry disabled, routing reliability/answer-quality gaps, unvalidated numerical estimates, bounded demo monitoring and no conversational memory persistence. Shared Genie/index/model-service policies unchanged.
- Final evidence verifier checks retrievable OK traces, lineage, context/token visibility, replay without model calls, persisted revision and exact audit actor; reports routing gaps explicitly. git diff --check passed.


## Chat workspace redesign

The existing FastAPI App was redesigned and redeployed with a specialist sidebar,
task starters, responsive conversation layout, safely formatted Markdown/tables,
case-approval dialog, account identity and per-answer execution-trace links.
A streaming request-lifecycle API emits accepted, processing, elapsed-time
heartbeats and result/error events. It invokes Model Serving once and does not
invent worker transitions or expose private reasoning. Named worker/tool activity
is inspected in the completed MLflow trace; the version 4 serving contract still
returns its complete answer rather than incremental graph events.

Seven App tests passed, covering authentication, origin and role checks,
single invocation, lifecycle ordering and failure without retry. JavaScript syntax
and strict FEVM bundle validation passed. Desktop and 390px mobile browser checks
verified no horizontal overflow, Running → Complete transitions, table rendering
and escaped model-supplied HTML using controlled response fixtures.

Live redeployment succeeded. Authenticated HTML, CSS, JS and whoami routes returned
HTTP 200. A real read-only synthetic-case stream delivered accepted/processing
at 0.24s, heartbeats at 2.24/4.24/6.24s and the answer at 7.92s. Returned case state
was investigating/version 2. Evidence: validation-results/ui-stream-smoke.json.
No model redeployment, grant expansion or database mutation was needed.


## Version 5 real execution events, Stop and composer clearing

- Registered models:/m-d1b710e9ca7a4d1abf86230a2e8fbe75 as version 5. Registration job 567845930464598 SUCCESS, including packaged plan/tool streaming, docs and exact-receipt checks. READY / NOT_UPDATING verified before live worker tests.
- Customer-owned graph emits high-level delegated task/next-agent events, worker start/end and actual tool start/end through a request-scoped ContextVar sink. ResponsesAgent publishes response.in_progress/custom_outputs.activity events; the App forwards these without inventing stages. No private reasoning or raw tool results are streamed as activity.
- Live case, docs and Genie streams all passed and produced retrievable OK traces. Named tools observed: read_dispute, search_databricks_docs, genie_query. Evidence: validation-results/live-worker-stream.json.
- Live App forwarded plan at 2.80s, read_dispute at 3.77s and answer at 6.22s; first execution activity at 1.54s. Evidence: validation-results/app-live-activity.json. App HTML and JavaScript assets return HTTP 200.
- Composer clears at submission, allows the next draft during execution, and does not clear that draft when the previous request finishes. Stop uses AbortController; no automatic retry. Browser checks verified clearing, real-event rendering, Stop with one invocation and preserved drafts using fixture streams.
- Stop closes the UI wait and upstream stream on disconnect. The agent checks a closed-stream flag at subsequent progress boundaries; in-flight model/tool calls may finish and committed mutations are never undone. Cancellation propagation is best effort, not an atomic workflow rollback.
- Local suite: 38 passed, 6 skipped. Progress order, concurrent request isolation, stream proxy metadata, authentication and no retry on error covered. JavaScript syntax, strict bundle validation and git diff --check passed. App deployment succeeded. No new grants, shared-resource changes or database mutations were needed.

## Shared App transport regression checks (2026-11-09)

Python 3.11 suite: **40 passed, 6 skipped** (environment-dependent tests), one
LangGraph dependency warning. App tests verify legacy final-result/trace metadata
collection, upstream cleanup, one invocation on success and failure, timeout
HTTP 504, generic failure HTTP 502, and explicit serving stream timeout settings.
Strict FEVM bundle validation passed. Live Genie evidence and limits are recorded
in `FEVM_STATUS.md`; no database mutation was needed for these checks.

## Workflow naming and structured Genie results (2026-11-09)

Local suite: **45 passed, 6 skipped** (environment-dependent tests), one dependency
warning. Added checks cover Genie wait timeout without restart, original
conversation/status retention, result-table schema/nulls/limits, preserving query
previews through supervisor synthesis, sanitizing App errors and retaining upstream
failure metadata. Read-only errors omit the case-update warning; approved requests
retain it. JavaScript syntax and table escaping/link generation checks passed.

Packaged version 6 validation succeeded in serverless job `148511758614522`:
documentation citation, existing mutation-receipt replay (no new write), and streamed
case read. Model URI: `models:/m-3d6d29cf58d645f98fd08dcf537ada07`.
These packaged checks do not themselves validate Genie response tables or native
chart embedding; live evidence is recorded separately in `FEVM_STATUS.md`.

Live version 6 checks also passed: five-row Genie result through the deployed App,
renamed workflow/Genie spans in the persisted OK trace, and a controlled serving
validation error with an ERROR trace. See the version 6 section of FEVM_STATUS.md
for trace IDs, timing and limits. The original long-running query was not replayed.

### Saved history and query charts

Run `node --test tests/test_charts.cjs` for monitoring-route, numeric-data,
HTML escaping and line-gap checks. History authorization and revision checks are
included in `tests/test_app.py`. Provision isolated history tables with
`PYTHONPATH=. python scripts/fevm_history_setup.py --profile <approved-profile>`.

FEVM validation passed authenticated App save/reload with the App's Lakebase role;
synthetic verification rows were removed. Direct database API tests also checked
two-user isolation, stale revisions and preference persistence. Browser rendering
checks used the actual existing Genie conversation's rows with mocked serving and
history responses; no Genie query was rerun. The original native chart specification
is not returned by that public message API, so native chart reproduction is not
claimed. See FEVM_STATUS.md for persistence boundaries and release details.

The grouped cost chart update has seven passing Node tests, including the three
cost-series defaults, nine grouped bars, legends, units, relative values and
independent gaps for multiple line series. Browser validation used the returned
Genie rows from trace `tr-8d44ddd7d5a050305fbf19551df9fa06`, with mocked serving
and history transport, and verified exported SVG is valid XML containing the title,
legend and nine data bars. Saved snapshots were reopened through the new renderer.

The final interactive implementation has eight passing Node tests and 15 passing
App API tests. Browser interaction checks exercised Plotly hover, actual pointer
zoom, actual legend click, chart-type switching and saved-thread restoration with
the existing trace rows. Plotly is pinned and served locally; no browser CDN request
or new Genie invocation is required. The PNG experiment was removed before activation.

### ResponsesAgent token and cost tags

The usage update passes 56 local tests (6 environment-dependent skips), covering
provider aliases, cached/reasoning details, callback deduplication, agent/model
attribution, missing usage, receipt replay, final streaming totals, unchanged
wrapper events and original stream errors. The packaged model also passed FEVM
serverless validation (run 615245174027011): 2,719 input + 275 output = 2,994
reported tokens across four calls, split between supervisor (2,263) and disputes
(731). The configured-rate estimate was USD 0.009962; this is not billing evidence.
Both models matched the configured rate card, and no usage was missing in that
read-only case request. The test did not exercise Genie internal token reporting.

Conversation renaming adds title-only owner/revision-checked API coverage.
Browser verification passed chosen-name preservation on follow-up and reopen,
cancel behavior and mobile dialog access. A live authenticated synthetic thread
passed rename/reload, preserved messages and stale-write rejection, then cleanup.

Genie agent-mode tests cover canonical STANDARD request fields, SSE framing,
terminal-success enforcement, timeout identifiers without retries, query-preview
conversion, private-reasoning exclusion and agent-mode deep links. The direct
FEVM agent API returned a successful assistant answer and one query table. A
four-role regression test ensures usage breakdown tag values fit Databricks'
255-byte limit; one large JSON tag is deliberately avoided.

Final local verification: 61 Python tests passed, 6 environment-dependent skips,
and 9 Node chart/link tests passed. FEVM run 635003557058473 registered version 9
and verified that trace `tr-81518735e30bc97ad3fe0e7facc16e2b` exists with
`agentops.tokens.total=2995`, input=2727 and output=268. Its four CHAT_MODEL spans
sum to the same 2,995 tokens (1,114 + 346 + 383 + 1,152), confirming the aggregate
is not counting wrapper spans again. Indexed agent/model tags and the USD 0.009995
configured-rate estimate were independently read back through MLflow. The same
run verified a STANDARD Genie agent response with one query-result preview table.

Final deployed version 9 validation passed through the actual App and supervisor
with a synthetic cost workload. Trace `tr-dd8d9e5c265faec07d29d1490b65b8df`
proves STANDARD agent mode, two SQL-result status cycles, terminal completion,
returned scenario rows and preserved usage. The App chart adapter selects all
three expected cost measures from the actual agent-mode table, producing three
grouped series and nine values. Response usage equals persisted AgentOPS tags
and MLflow totals (6,734). Genie-internal usage is unavailable and coverage is
partial. Final Python suite: 62 passed, 6 environment-dependent skips; Node: 9 passed.

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

## File-style memory workspace pivot

- Local agent/App suite: 71 passed, one optional test skipped; includes disposable
  PostgreSQL checks for persistent files, revision conflicts, owner isolation,
  history, restore, expiry and concurrent creation.
- Framework adapter/text contracts: seven passed.
- Approved FEVM Lakebase provisioning completed: file/version tables, App summaries,
  scoped grants, isolated signing secret and App READ ACL.
- Direct FEVM persistence validation passed in isolated namespace
  `evaluation:file-memory:431b4eaf-71f6-4ac8-b936-4c5026958d13`.
  Fresh-store recall, isolation, stale writes and restoration passed. Fixture is
  soft-deleted; version history retained. Test ran under the setup identity.
- Model-tool compatibility run: `614766712455059` (result recorded below when complete).
- No serving traffic change or App redeployment in this pivot validation.
