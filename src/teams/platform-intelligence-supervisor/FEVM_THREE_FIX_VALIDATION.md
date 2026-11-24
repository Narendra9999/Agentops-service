# Framework correction validation — 24 November 2026

This record concerns the non-split `feat/platform-intelligence-supervisor` branch,
endpoint `fe-platform-intelligence-supervisor`, and experiment `968004603421016`.
The historical `FEVM_VALIDATION.md` describes an earlier deployment.

## Corrections

1. Ordinary team ResponsesAgents now use framework-owned, request-local LangChain
   usage capture. Regular and streaming calls account for each local model result
   once. Existing AgentOPSBase and Chat Completions accounting paths are retained
   without a second capture callback. Native model-span pricing uses the shared
   configured rate card and preserves existing provider costs. The request
   boundary restores the pricing processor if runtime initialization removed it.
2. A complete `inference.target_module` / `target_class` pair takes precedence over
   conventional `Agent.py`. If either field is absent, use `Agent.py` with the
   specified class, or `ChatModel` when unspecified. Invalid explicit modules,
   imports and classes fail rather than silently selecting another agent.
   Nested modules and relative imports remain supported.
3. Predeployment evaluation resolves each team fixture before shared defaults and
   refreshes its evaluation tables from those fixtures. Existing UC table rows
   no longer override packaged fixtures. Defaults apply independently to each
   absent fixture, not to an entire team merely because one file is missing.

## Tests

- Local integration suite: 174 passed.
- Focused unit suite: 35 passed.
- [FEVM serverless regressions](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/runs/1024382423895386?o=7474644082064164): succeeded.
  Covered regular/streaming usage, configured/incomplete inference, invalid
  targets, nested modules, team/default/partial fixtures, stale-table prevention,
  rubric runtime loading and runtime cost-processor restoration.

## Deployment evidence

[Initial pipeline run](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/852118393555051/runs/681883487993369?o=7474644082064164)
registered and deployed version 6. Predeployment gates and serving smoke tests
passed. Live Responses output and streaming completion were valid and AgentOPS
usage/count/cost tags matched observed local model calls. Native MLflow cost
attributes remained absent in serving, motivating the request-boundary repair.
The postdeployment quality gate failed; an evaluated platform answer omitted
requested tool-tracing guidance. Thresholds were not relaxed.

[Final correction pipeline](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/852118393555051/runs/136508571861060?o=7474644082064164)
registered version 7. Its predeployment correctness gate failed (`0.667 < 0.8`):
a platform answer omitted actual documentation URLs. Domain rubrics passed
(cost estimation `4.667/5`, platform evidence `5/5`), relevance/safety were `1.0`,
and guardrail accuracy was `1.0`. Unchanged automatic retries were canceled;
version 7 was not promoted to the main endpoint.

[Packaged-model check](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/runs/1043725454593161?o=7474644082064164)
loaded version 7 directly on FEVM without working-tree imports. It passed:
2 local model calls, 3,529 tokens, USD 0.019239 estimated cost. Native model span
costs, trace-header cost and AgentOPS tags agreed. Trace:
`tr-eb0557d0e1f57364d52e43eacedfe480`.

An isolated `fe-platform-intelligence-supervisor-validation` endpoint runs the
candidate for serving regressions without bypassing the main quality gate.
All five live response/metrics scenarios succeeded on isolated version 7.
This is targeted framework validation, not a successful main-endpoint promotion.

| Scenario | Response/usage checks | Local calls | Tokens | Estimated USD | Trace |
|---|---|---:|---:|---:|---|
| platform | Succeeded | 2 | 3289 | 0.018747 | `tr-cb2015bc66f8d718a98aff091e933e30` |
| cost | Succeeded | 2 | 6937 | 0.038127 | `tr-352b41ad7bbab9e305375bb3363e18d2` |
| combined_stream | Succeeded | 2 | 3401 | 0.018663 | `tr-245925fc05174ad62a5fdc25b9301755` |
| platform_stream | Succeeded | 2 | 3337 | 0.018831 | `tr-abbde0177bf746eee18c2280cc0c4402` |
| guardrail | Succeeded | 0 | 0 | 0 | `tr-fbff48eee1160ae281eb08dc80454c65` |

For each unblocked request, assertions verified HTTP 200, a valid Responses
output/completed SSE item, nonempty answer, standard AgentOPS tags, native model
usage, equal native/tag token totals and call counts, positive equal native/tag
costs, and native trace-header cost. No second accounting of parent/child usage
was observed. The guardrail request produced an answer without any model/tool
calls. Streaming here is buffered completed output, not incremental token delivery.

### Separate failures

- Main rollout: failed correctness gate; main endpoint remains version 6.
- Strict cost-only routing: failed. Trace `tr-352b41ad7bbab9e305375bb3363e18d2`
  contains both `cost_intelligence` and `platform_engineering` tool invocations.
  The cost metrics passed for the calls that actually occurred; this is not a
  claim that the routing policy passed. Team orchestration was not changed to
  hide or enforce this behavior within AgentOPS middleware.

## Requested checklist

- [x] a. Trace and usage capture: succeeded in isolated FEVM serving version 7,
  including native costs and regular/streaming paths.
- [x] b. Inference selection/fallback: succeeded in local and FEVM regressions.
- [x] c. Team/default fixture fallback: succeeded in local and FEVM regressions.
- [ ] Overall main-endpoint pipeline promotion: failed quality gate.
- [ ] Cost-only worker exclusivity: failed routing policy check.

## Cost scope

Token usage comes from model responses; dollar amounts are configured-rate
estimates for observed local supervisor calls. Remote platform and Genie internal
billing is outside this scope. Parent and child totals must not be added together.

Runtime OAuth secret bindings and AI Gateway governance remain configured. The
supervisor now calls the standard `databricks-claude-sonnet-4-6` serving resource.
This validation does not establish secretless authentication.
