# Original AgentOPS pipeline in FEVM

This bundle supplies FEVM compute and team parameters to the **original AgentOPS
notebooks**. It contains no separate registration, evaluation, deployment or smoke
implementation. Its task paths and dependencies are checked against
`resources/pipeline-resource.yml` by `tests/unit/test_team_runtime.py`.

| Stage | Original shared notebook |
|---|---|
| Vector Search setup | `src/data_preparation/vector_search/notebooks/VectorSearch.py` |
| Register model | `src/agent_development/agent/notebooks/RegisterModel.py` |
| Pre-deployment evaluation | `src/agent_development/agent_evaluation/notebooks/PreDeploymentEval.py` |
| Deploy agent | `src/agent_development/agent/notebooks/DeployAgent.py` |
| Smoke test | `src/agent_development/agent_evaluation/notebooks/SmokeTest.py` |
| Post-deployment evaluation | `src/agent_development/agent_evaluation/notebooks/PostDeploymentEval.py` |

Run from this directory with the explicitly selected FEVM profile:

```sh
databricks bundle validate --strict -t fevm --profile fevm-naren
databricks bundle deploy -t fevm --profile fevm-naren
databricks bundle run agentops_pipeline -t fevm --profile fevm-naren --no-wait
```

The deployment notebook uses `databricks.agents.deploy()`. Registration uses the
existing framework wheel and team `agent/Agent.py` through the shared
`CostTrackingResponsesModel`. Evaluation uses the shared `run_evaluation()`, audit
logging, team fixtures and scorer loader. No Genie agent or worker endpoint is
created. Vector Search creation is disabled because the configured index exists.

FEVM uses serverless environments with the team's pinned dependencies, selected
by `dependency_source: environment`. Original customer jobs retain the default
`volume` mode and offline dependency installation. FEVM disables environment
packing in team config; the shared registration path still supports its default
offline `env_pack` behavior.

`framework.team_runtime.TeamRuntime` loads an optional trusted
`teams/<team>/request_adapter.py` to construct signed evaluation requests. This
team's adapter assigns isolated evaluation owners and signs bounded context using
a secret reference; it does not substitute lifecycle steps. The shared scorer
loader loads `scorers/routing.py` from this team. Canonical golden fixtures retain
`request`, `expected_response` and typed `expected_output` fields. Adversarial
fixtures cover this team's enabled length, PII and injection checks.

This team opts out of registration-time promotion. The original notebooks pass
this run's candidate version to evaluation and deployment and advance `champion`
after serving readiness. Existing serving environment/secret references are
preserved. Post-deployment failure does not automatically roll back deployment.

The earlier version 19 rollout used a separate team lifecycle and is historical
functional evidence only. Validation of these original notebooks is recorded
separately after the current run completes.
