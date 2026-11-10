# Team configuration and shared pipeline contract

Teams supply an agent, configuration, fixtures and optional scorers/request
adapters under `src/teams/<team>/`. The original six AgentOPS notebooks own
registration, evaluation, deployment and audit reporting. Customer code owns
routing, delegation and memory tools.

## Notebook startup

Each notebook has a small standard-library locator to import the shared framework
from the deployed bundle. `initialize_notebook(dbutils)` then resolves normalized
Workspace paths, validates `dependency_source` and prepares import paths. Call it
again after `restartPython()`. It does not read secrets, connect to Lakebase, load
an agent or choose workers.

The default dependency source is `volume`, preserving offline customer jobs.
`environment` uses the job's installed dependencies without consulting a volume
or reinstalling packages. Job parameters supply this execution setting because
startup occurs before team configuration and dependency installation.

## Team configuration

- `llm.endpoint` and `llm.api`: the supervisor model identity and transport. The
  team's entry point translates these to its internal settings object.
- `multi_agent.settings`: customer-owned worker bindings, memory policy and graph
  limits. Shared lifecycle code does not interpret specialist names or topology.
- `model.resources`: explicit registration resources. Supported types are
  `serving_endpoint`, `genie_space`, and `vector_search_index`. Use either `name`
  or `name_from` (a dotted config path). Optional `when`/`when_not` mappings match
  exact configuration values. Referenced bindings must exist. Enabled resources
  must have a nonempty name; duplicate declarations are removed.
- `model.pip_dependencies_file`: a team-relative lock file of package specs. It
  replaces repeated package lists when combined with `pip_dependency_mode:
  replace`. Legacy inline dependency lists and extend mode remain supported;
  do not configure both a file and a nonempty inline list.
- `pipeline.request_timeout_seconds`: shared serving read timeout, default 240.
- `pipeline.sanity_queries.pre_deployment` and `.deployment`: optional team-specific
  startup checks. Generic questions remain the default for existing teams.

Example resource declaration:

```yaml
model:
  pip_dependency_mode: replace
  pip_dependencies_file: configs/requirements.lock
  resources:
    - type: vector_search_index
      name_from: multi_agent.settings.docs_index
      when:
        multi_agent.settings.platform_mode: in_process
    - type: genie_space
      name_from: multi_agent.settings.genie_space_id
      when:
        multi_agent.settings.cost_endpoint: ''
```

Single-agent teams without explicit resources keep the existing LLM resource
behavior and legacy Vector Search/table/warehouse declarations. Multi-agent teams
must provide explicit resources; the framework no longer guesses their topology.
Changing a worker transport requires updating the team bindings and matching
resource conditions. UC AI Gateway model services are not serving endpoints and
must not be declared as such; their runtime EXECUTE grants remain required.

## Common execution helpers

`TeamRuntime` loads configuration and an optional trusted team request adapter.
It supplies signed evaluation requests, reusable public input examples and
scenario queries. Its `invoke()` method uses the common inference client for
deployment smoke checks, SmokeTest and post-deployment evaluation. The client
uses WorkspaceClient authentication with one HTTP invocation, a shared timeout
policy and no automatic inference retry. Signed context validation and isolated
evaluation owners remain in the team adapter.

Registration builds and installs one framework wheel in isolated writable
storage, then uploads that exact artifact. All metric-gate consumers use the same
finite-value and threshold checks. These helpers do not replace the original
pipeline stages.

The Platform Intelligence bundle declares common notebook parameters once at job
level, with stage-specific upstream experiment IDs remaining task parameters.
Its serverless environment and registered model reference the same checked-in
`configs/requirements.lock`; package versions are unchanged by this refactor.

## Validation and rollout

Refactor validation: 89 focused shared tests, 19 ResponsesAgent integration tests
and 109 agent/App tests passed;
strict FEVM bundle validation passed. A separate existing notification test
requires the absent `resources/production-monitoring-resource.yml` and was
excluded after confirming that failure. Four provider integration tests require
the local `openai` package and are outside this validation.

Bundle validation does not execute serverless requirements-file installation.
This refactor has not been deployed or run in FEVM. Use the existing full pipeline
to validate registration, both evaluation phases, endpoint readiness and smoke
checks before claiming a successful rollout. App deployment remains separate.
