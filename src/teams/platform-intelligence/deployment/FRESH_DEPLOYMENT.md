# Fresh-agent validation — 2026-11-10

The original AgentOPS pipeline successfully registered, evaluated and deployed
`fevm_catalog_naren.agentops_multi.platform_intelligence` version **1** at
`fe-platform-intelligence`.

[Pipeline run 795232273128037](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/701419594686647/runs/795232273128037?o=7474644082064164)

## Configuration and packaging corrections

- Team YAML specifies `/Shared/agentops-platform-intelligence` by name, with no
  numeric experiment ID. Registration resolves/creates it and passes generated
  ID **240926829857302** to model configuration, evaluation and serving.
- The resolved ID overrides stale endpoint environment destinations. Four live
  functional traces were confirmed OK in the new experiment.
- The new endpoint has explicit runtime OAuth configuration using an existing
  secret reference, rather than relying on environment inherited from an older
  endpoint. No plaintext credential was added.
- The sole AgentOPS entry point remains `agent/Agent.py:ChatModel`; the internal
  implementation is now `fe_multi_agent/supervisor_agent.py:SupervisorAgent`.
  No lower-case `agent.py` remains in that package.

## Results

| Quality gate | Before | After |
|---|---|---|
| correctness | 1.0 | 1.0 |
| relevance_to_query | 1.0 | 1.0 |
| safety | 1.0 | 1.0 |
| routing | 1.0 | 1.0 |
| cost_grounding | 1.0 | 1.0 |
| platform_evidence | 1.0 | 1.0 |

Guardrail accuracy was **1.0** on 17 configured fixtures before and after
serving deployment. The original smoke stage passed. Live streaming checks
passed explicit file writes, persistent Lakebase readback, fresh-conversation
recall, embedded Platform retrieval with documentation citations, and Genie
Cost Intelligence routing. All four live traces completed successfully.

Local checks: **74** focused shared-framework tests, **96** example tests, and
strict bundle validation passed. These suites run in separate processes because
shared unit-test MLflow stubs interfere with real MLflow imports when combined.
The earlier full-repository suite limitations remain documented in VALIDATION.md.

## Gateway availability

Inference tables are enabled with prefix `platform_intelligence`. FEVM explicitly
rejected each advanced option on this endpoint type: usage tracking, rate limits,
and AI Guardrails. They were verified separately, not inferred from YAML flags.
The original deployment tolerates these unsupported features, so pipeline success
does not imply that the requested Gateway controls were installed. AgentOPS
application guardrails passed the fixture checks; these are separate controls.
Native MLflow token and cost attributes were verified on six model spans from
the Platform scenario. Genie internal token/cost coverage remains unavailable.

## Scope and follow-up

This proves first deployment of a new model and serving endpoint within FEVM.
It reuses the approved Lakebase project, runtime service principal, secret scopes,
Genie agent and Vector Search index. It does not prove provisioning in a different
workspace. Those dependencies and their permissions must be configured there.

App deployment is deferred at the user's request. The existing App and old
`fe-platform-agent-team` endpoint remain available. App history still uses its
own Lakebase configuration; integrating it with the common `session_history`
configuration remains a follow-up. Shared automatic history/recall stays disabled
while explicit supervisor file memory remains enabled.

The first fresh run was intentionally canceled during registration to rename
the internal supervisor module before packaging; the run linked above completed
the resulting implementation. No quality thresholds were weakened.

Sanitized execution evidence is stored locally in the ignored
`examples/multi_agent_fe/validation-results/` directory.
