> Historical validation record: the results below were recorded in the original
> working tree before this team was restored to the non-split branch. That working
> tree included uncommitted shared scorer fixes described below. This restoration
> does not include those fixes or establish that the same pipeline results hold
> on this branch. No new deployment was performed as part of the restoration.

# FEVM validation

- Workspace: `fevm-classic-stable-naren.cloud.databricks.com`.
- Endpoint: `fe-platform-intelligence-supervisor`.
- Registered model: `fevm_catalog_naren.agentops_multi.platform_intelligence_supervisor`, version `3`.
- Experiment: `968004603421016` (`/Shared/agentops-platform-intelligence-supervisor`).
- [Pipeline run](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/852118393555051/runs/176301218819477?o=7474644082064164).

## Verified live requests

| Scenario | HTTP | Duration | Native tool spans |
|---|---|---|---|
| Platform guidance | 200 | 18.24 s | `platform_engineering` |
| Missing cost-estimation inputs | 200 | 19.57 s | `cost_intelligence` |
| Combined platform and cost question | 200 | 43.45 s | Both workers |
| Buffered streaming platform guidance | 200 | 15.61 s | Completed assistant output event |
| Too-short input | 200 | Guardrail response | No worker invocation |

Successful cost trace: `tr-da156c6b710dfff70a02574957dd94c6`.
Successful combined trace: `tr-a7ad82295e7c29f40c589395dd7cd3ad`.
Short-input trace: `tr-7db1323a7bc5bd234afdce1ea14a59a4`.
Streaming returned `response.output_item.done` with 1,075 characters of assistant text.
This team buffers its answer; incremental tokens and an activity feed are not implemented.

## Evaluation and checks

Predeployment evaluation passed: correctness, relevance and safety each `1.0`;
cost-estimation and platform-evidence rubrics each `5.0/5`; guardrail accuracy `1.0`.
The evidence rubric compares the final answer with worker output. It does not independently
validate every statement in the worker output. The three golden scenarios are a small
validation set, not a comprehensive quality guarantee.

Local integration suite: **161 passed** (excluding the separate live-endpoint test module).
Whitespace checks passed. The repaired shared smoke test passed after correcting
the runtime worker ACL.

The first postdeployment evaluation failed: correctness `0.667` (required `0.8`),
cost-estimation rubric `3.667/5` and platform-evidence rubric `3.667/5` (each required `4`).
Relevance and safety were `1.0`; guardrail accuracy was `1.0`.
Evaluation ID: `ac793e9c-8888-405c-a693-54bf176287ee`.
The platform-only answer omitted actual documentation URLs; the correctness judge
also noted missing explicit tool-tracing coverage. Other judge assessments questioned
worker claims. This is a quality concern despite successful transport and routing.
The automatic retry passed with correctness, relevance, safety and guardrail accuracy
all `1.0`; cost-estimation quality `4.0/5`; platform evidence `4.333/5`.
Evaluation ID: `d333eb29-4255-45df-bfc3-8568c014371a`.
The overall pipeline terminated with **SUCCESS**, with all six latest task outcomes
successful (Vector Search setup intentionally skipped provisioning).
Thresholds were not lowered. The same model version was evaluated on both attempts;
the differing results expose generation/judge variability, not a deterministic quality fix.
Treat this as successful deployment validation with a remaining quality-stability concern.

## Issues corrected during validation

- Imported the packaged worker module during target loading, rather than deferring its
  import until after the temporary staging import path was removed.
- Honored the configured rubric judge model and loaded the judge helper before scoring
  threads execute; supplied bounded native tool evidence to configured rubrics.
- Preserved fail-closed quality gating when a metric is missing, with a readable error.
- Granted runtime service principal `fe-agentops-multi-runtime` **CAN_QUERY** on the
  existing `fe-platform-single-validation` worker. Notebook evaluation used the deployer
  identity and therefore did not expose this missing runtime permission. Existing ACLs
  were preserved. Configure equivalent resource access in another workspace.

Only the supervisor was deployed. Genie and the platform worker remain existing resources.
Native parent traces include worker calls; remote internals and complete remote-worker
billing are not automatically aggregated by these simple text-returning tools.
No App, persistent memory, or daily numeric-rubric monitor was deployed in this validation.
