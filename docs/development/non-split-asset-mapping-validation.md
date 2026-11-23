# Non-split asset mapping: FEVM validation

Branch: `feat/platform-intelligence-supervisor` (isolated non-split worktree).
Combined team `config.yaml` is retained. The split experiment is untouched.

## Change and isolation

`model.asset_mappings` explicitly copies team assets into new locations in a
private staging tree before MLflow packaging. No customer implementation is
rewritten. Default/empty mappings retain the original code paths and skip staging.
Only the shared packaging helper and registration notebook change framework
behavior. Middleware, routing, instrumentation, guardrails, evaluation, and serving
contracts are unchanged.

Two isolated teams exercise the shared six-stage pipeline:

| Case | Team | Endpoint | MLflow experiment |
|---|---|---|---|
| Unchanged supervisor implementation moved below `agent/supervisors`, with `configs -> agent/configs` mapping | `platform-intelligence-assets` | `fe-platform-intelligence-assets` | `4253071280209070` |
| Unchanged existing single-agent implementation, mappings omitted | `platform-single-assets-control` | `fe-platform-single-assets-control` | `4253071280209091` |

The nested supervisor and worker file contents match the original clean team's
implementation. Its new `Agent.py` imports through `agent.supervisors` so the test
actually executes the nested implementation rather than a flattened import alias.
The fixture uses existing Guidelines domain scorers, builtin correctness/relevance/
safety, the original golden/adversarial cases, and the existing 0.8 gates. It does
not import the split experiment's uncommitted rubric-judge changes or lower gates.

Existing endpoints remain separate: `fe-platform-single-validation` version 17,
and `fe-platform-intelligence-supervisor` version 3, both at 100% traffic.

## Local checks

- Integration suite: **157 passed**, excluding the live-serving test module.
- Real MLflow save/load checks remove the original source checkout before loading.
  They cover unchanged top-level relative paths, nested local assets, and mapped
  nested prompt paths. The nested prompt read demonstrably fails before mapping.
- Validation covers omitted/empty mappings, malformed entries, missing sources,
  path escapes, external symlinks, existing/duplicate destination rejection,
  directory/file copying, preserved import aliases, and unchanged source files.
- Registration regression tests check the prepackaging import-root block and
  preservation of the prompt-manifest directory binding.
- Unit baseline comparison: both committed HEAD and the modified worktree have
  **147 passes and the same 13 failures** when excluding `test_chunking.py`.
  That excluded module cannot collect because the committed `create_chunk.py`
  contains only `test`. The 13 failures concern existing dashboard functions and
  the missing `resources/production-monitoring-resource.yml`. These files were
  not changed for this task. The complete unit suite is not green.

## FEVM runs and iterative fixes

- [Supervisor pipeline](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/20690260735812/runs/984463038573837?o=7474644082064164)
- [Single-agent control pipeline](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/321643752339694/runs/80605624177599?o=7474644082064164)
- Both bundles pass `bundle validate --strict` using the explicitly selected
  `fevm-naren` profile.

Initial FEVM attempts exposed two registration defects, retained in run history:

1. Inline ResponsesAgent validation runs before MLflow copies code. The mapped
   staging tree's package root must be on `sys.path` for `agent.supervisors...`
   imports. Registration now adds this root only when mappings are enabled.
2. The prior non-split helper refactor removed `team_configs_dir`, still referenced
   by the existing Prompt Registry-disabled manifest check. Registration restores
   that binding for both default and mapped packaging.

Successful registration after repair produced version 3 for both isolated models.
The earlier registered versions belong to attempts that failed later notebook
checks; they are not claimed as successful deployments.

## Registered artifact verification

The actual Unity Catalog model artifacts were downloaded using MLflow's supported
SDK artifact transport and checked against team source bytes:

- `fevm_catalog_naren.agentops_multi.platform_intelligence_assets/3`, registration
  run `d224bda70d2c40ce932297549136c6c1`.
- `fevm_catalog_naren.agentops_multi.platform_single_assets_control/3`, registration
  run `ae05bc3dcb2c4d1f97e0bbe8381713cc`.

Supervisor source SHA-256:
`09a2b853b58321128377d67984bc12308191cece9426824ff065df99da253953`.
The nested supervisor, worker source, sibling prompts, and mapped prompts match.
No `agent/configs` directory is created in the original team checkout.
The single-agent artifact has its original source and no injected `agent/configs`.

## Supervisor quality-gate result

The nested supervisor registered and ran successfully in FEVM, including native
traces with successful `cost_intelligence` and `platform_engineering` tool spans.
It did **not** pass the answer-quality gate, so the shared pipeline correctly
blocked deployment. No supervisor test endpoint was created and its live-serving
stream was not tested. The gate was not bypassed or lowered.

| Assessment | First attempt | Configured retry | Threshold |
|---|---:|---:|---:|
| Correctness | 0.667 | 0.667 | 0.8 |
| Cost grounding | 1.000 | 0.667 | 0.8 |
| Platform evidence | 0.667 | 1.000 | 0.8 |
| Relevance | 1.000 | 1.000 | 0.8 |
| Safety | 1.000 | 1.000 | 0.8 |
| Guardrail accuracy | 1.000 | 1.000 | 0.8 |

Evaluation IDs: `daa09bd6-22d6-4eef-98b5-4402bc2cf78a` and
`f91fd908-926a-4ad2-8355-d5f284aaead8`.
The tracing answer omitted explicit tool-tracing coverage. Judge feedback also
questioned specialist documentation and interpreted an optional cost-discussion
offer as violating the cost-grounding guideline. These are answer/evaluator
findings, not file-loading failures; this packaging change does not resolve them.

The actual existing deployed supervisor artifact (version 3) was downloaded and
compared too: supervisor source, worker source, and `configs/prompts.json` are
byte-for-byte identical to the nested fixture. This establishes implementation
and prompt equivalence, but is not a claim of deterministic answer equivalence
or that the original supervisor would pass the same evaluation on another run.

Representative successful runtime traces:

- `tr-cf8193890e8bb5353aecc48c61857f6c`: cost and platform worker spans.
- `tr-b0f491847a60448e894ccbb44832eb4c`: platform worker span.
- `tr-a41440f8a509521fd083101ef7101cff`: both workers on the mixed request.

## Single-agent pipeline and live-serving result

Pre-deployment evaluation passed: correctness, relevance, safety, platform evidence,
and guardrail accuracy were all 1.0. Deployment and smoke testing passed. The endpoint is `READY` and `NOT_UPDATING`,
with version 3 at 100% traffic. Independent live checks passed:

- Regular Responses-compatible output with Databricks documentation citations.
- Streaming `response.output_item.done` output with a nonempty cited answer.
- Short input rejected with: "Your question is too short. Please provide more
  detail so I can help you."

All six shared pipeline stages completed successfully: setup, registration,
pre-deployment evaluation, deployment, smoke testing, and post-deployment evaluation.
Post-deployment correctness, relevance, safety, platform evidence, and guardrail
accuracy were all **1.0**. Evaluation ID:
`918303d1-fc31-4330-a92c-0a422aa7231f`.

The endpoint remains version 3 with 100% traffic. Final read-only checks confirm
that the original single-agent version 17 and original supervisor version 3 remain
`READY` / `NOT_UPDATING` with 100% traffic. No existing team config or agent
implementation was edited.

The mapping behavior is validated in FEVM registration and supervisor execution;
the default path is validated through the complete deployed single-agent pipeline.
This is not an all-green supervisor deployment result: its quality-gate failure
remains unresolved and is reported above.

Machine-readable run history, task outputs, artifact comparisons, and live-check
results are recorded in [the FEVM results JSON](non-split-asset-mapping-fevm-results.json).
