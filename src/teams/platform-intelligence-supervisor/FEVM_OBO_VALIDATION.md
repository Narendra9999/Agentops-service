# FEVM OBO and Genie conversations validation

## Implementation under test

- Branch: `feat/platform-intelligence-supervisor`, user commits through `8f4097e`.
- Endpoint: `fe-platform-intelligence-supervisor`.
- Registered model: `fevm_catalog_naren.agentops_multi.platform_intelligence_supervisor`, version **11**.
- Registration MLflow run: `4690479a4c224ed882aa2eb5443c91af`.
- [AgentOPS pipeline run](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/852118393555051/runs/457835384027871?o=7474644082064164).
- Original six-stage AgentOPS pipeline; no alternate deployment path.
- User's `model.auth_policy.user_api_scopes: [genie, sql]` registration format.
- System identity for supervisor model/platform worker; caller OBO for Genie.
- Genie uses SDK conversations, completion polling and query-result extraction.
  This is not the Genie Agent Responses API or STANDARD agent mode.

The local draft's alternative `deployment.authentication` configuration and
per-request supervisor graph have been superseded by the user's committed
implementation. The draft and its earlier validation notes remain preserved in
the Git stash created before the fast-forward merge; they are not the deployed
candidate source.

## Local validation

- **190 integration tests passed**, excluding the separate live-endpoint module.
- **35 focused unit tests passed**.
- Includes middleware request/stream handling, guardrails, module-only packaging,
  native usage regression checks, unchanged system-only registration defaults,
  actual MLflow policy serialization, SDK text/query-result contracts,
  missing/invalid results, per-invocation callers and permission/credential denial
  without system-identity retry.

## Workspace prerequisite

The `agents_obo` preview was disabled and has now been enabled by the workspace
admin API; its effective value was verified true. A separate diagnostic model
registered after enablement returned HTTP 200 and confirmed credentials available
at the serving entry point and in `asyncio.to_thread`. The temporary endpoint was
removed. Earlier version 9 could not acquire invoker credentials; a fresh version
10 deployment passed smoke validation. These earlier results do not establish
that this conversations-based candidate works.

## Live validation

All six pipeline stages passed: vector setup, registration, predeployment evaluation,
deployment, smoke test and postdeployment evaluation. Version 11 is READY with
NOT_UPDATING configuration and 100% traffic; version 10 has been removed.

The saved artifact contains user scopes `[genie, sql]`, declared system resources
and no mutually exclusive top-level `resources`. Postdeployment evaluation passed
quality and guardrail gates (guardrail accuracy 1.0).

All six live scenarios passed:

| Scenario | Trace | Result |
|---|---|---|
| platform | `tr-456aa56f99a066e062c3141ffd82db10` | Passed |
| combined_stream | `tr-0b40c31092b18f8a212ae745d27fef9c` | Passed |
| platform_stream | `tr-c28b8e9271a58f242dc80ee20b2493a3` | Passed |
| guardrail | `tr-256595a3f0449d4192895413b297fc7f` | Passed |
| cost | `tr-1fcf889d6d5bc1b40c36166023b5a68f` | Passed |
| cost_sql | `tr-343a5cfd31a7f9f51c87c2f45185bebf` | Passed |

Regular and streaming responses conform to the Responses contract. Genie SQL
attachments contain schema and data rows. Genie message user IDs match the
signed-in invoker. Native token totals and estimated costs match AgentOPS tags;
blocked input produces no model calls. Usage covers observed supervisor calls.

The broad cost-clarification prompt also invoked platform engineering. The SQL
scenario called cost intelligence twice. These orchestration decisions are
reported observations; routing exclusivity and exactly-once worker execution are
not asserted by these tests, and the user's orchestration was left unchanged.

The first ownership checks failed due to validation-script SDK assumptions.
The corrected checks use `list_conversation_messages(...).messages[].user_id`.
Only the affected checks were repeated; no runtime changes were needed.

Independent service-principal caller verification passed in run
`387679742456543`: HTTP 403 before the temporary CAN_QUERY grant, then HTTP 200
and Genie message owner `74101833841131` matching that caller. The signed-in
user's Genie messages instead belong to `78943176438521`. The temporary grant
was removed and endpoint ACL cleanup independently verified.

Second-caller trace: `tr-d0eb0780d36e4560f867d33ca9a83833`.
Workspace OBO was enabled as a deployment prerequisite; the user's runtime
implementation was deployed unchanged through commit `8f4097e`.


## Boundaries

The worker returns an explicit `Skipping Genie` result when caller credentials or
permissions are missing. HTTP 200 alone is therefore insufficient: successful
Genie calls and their conversation owner must be verified. Each tool invocation
starts a new conversation. The returned SQL result is the SDK's supplied chunk;
no automatic additional-chunk fetch or chart rendering is included.

The App has not been updated for human-user token forwarding. An App service
principal is the caller unless user authorization forwards the human identity.
Remote platform internals retain that endpoint's own authentication policy.
Supervisor usage/cost metrics cover observed model calls; costs are configured
rate estimates and do not establish complete remote-worker billing.
