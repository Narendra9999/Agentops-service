# Platform Intelligence Supervisor

A team-owned LangGraph supervisor calls two existing workers:

- `platform_engineering`: a Responses-compatible serving endpoint.
- `cost_intelligence`: an existing Genie space, using STANDARD agent mode.

Only this supervisor is registered and deployed. The platform endpoint defaults to
`fe-platform-single-validation`; the Genie binding uses the existing FEVM space.
Change `supervisor.workers` in `config.yaml` for your resources. The platform endpoint
must be a worker endpoint, not this supervisor's own endpoint. Runtime identity needs
access to both resources and the configured AI Gateway model.

## Ownership

`agent/Agent.py` imports the configured target class from
`agent/PlatformintelligenceSupervisor.py`, which owns graph construction, routing
instructions and worker tools.
`agent/workers/platform_engineering.py` and `agent/workers/cost_intelligence.py`
own the two worker API calls; `workers/__init__.py` exports them. The team is an ordinary MLflow
ResponsesAgent; it imports no AgentOPS orchestration or supervisor base class.

The AgentOPS registration loader loads `agent/Agent.py`, which imports the team
supervisor. `inference.target_class` selects `PlatformIntelligenceSupervisor`.
Keep `Agent.py` on this non-split branch; module-only loading is not available here.
The framework loads its context and wraps its public calls. The shared framework handles middleware,
configured guardrails, response adaptation, native instrumentation, usage handling,
evaluation and deployment. Only the shared serving entry point calls `set_model`.

## Configuration and evaluation

- `configs/prompts.json`: existing packaged prompt format with verified SHA256.
- `guardrails`: existing input/output checks; no custom guardrail implementation.
- `scorers/domain/cost_estimation_quality.yaml`: custom 1–5 rubric, threshold 4.
- `scorers/domain/platform_evidence.yaml`: documentation-evidence rubric.
- `evaluation.domain_scorers`: selection, weights and thresholds using the shared loader.
- `fixtures`: cost-only, platform-only and mixed requests plus adversarial inputs.
- `deployment/databricks.yml`: original six-stage shared pipeline, distinct endpoint,
  experiment name and evaluation tables. Vector Search setup is skipped because the
  platform worker is already deployed. Registration resolves the experiment ID.

The existing locked dependency set is reused. FEVM runtime environment bindings are
examples; rebind authentication and resources for a customer workspace. Deployment
credentials are not automatically inherited by model-serving runtime processes.

## Deliberate limits

No memory, signing key, App deployment, custom activity feed or token-by-token stream.
The target exposes standard `predict()` and `predict_stream()` methods only;
private `_run()` executes its async graph. AgentOPS bridges these synchronous
entry points into its async middleware. Streaming returns the final output item
through the shared buffered middleware.
Worker exceptions propagate to AgentOPS error handling; calls are not automatically
retried. Routing is model-driven and bounded by the configured graph step limit;
the prompt's once-per-worker rule is an instruction, not an enforced call quota.

Native LangChain tracing observes the supervisor and its tool calls. Remote worker
internals are not automatically joined to this trace. Worker token usage is not
returned by these simple text tools, so supervisor trace usage is not complete system
billing. Answer rubrics do not prove that a particular worker was invoked. The
current framework production-monitor loader excludes custom numeric rubric closures;
this rubric is for pre/post deployment evaluation, not an automatically registered
daily numeric-rubric monitor.

Local contract tests use controlled workers and graphs. No shared graph builder
or supervisor base is required. See [FEVM validation](FEVM_VALIDATION.md) for
historical results and the limitations of this restored non-split branch.

## Non-split branch limitations

This restoration adds team files only; it does not copy uncommitted framework
changes from the split-config worktree. The existing numeric domain scorer loader
does not forward `evaluation.judge_model` to its judge helper or supply native
worker-span evidence to the rubric. These evidence-oriented rubrics therefore
require validation against this branch before production use. No new FEVM run
or serving endpoint update is claimed by restoring these files.

`platform-intelligence-assets` remains a separate nested-path packaging fixture.
This team keeps its implementation directly under `agent/`, so sibling prompt
loading needs no asset mapping. Both teams use one combined `config.yaml`.

## Restoration checks

On this non-split branch, 165 local integration tests passed (excluding the
separate live-endpoint test module). These include the restored team contract,
request/stream middleware, injection guardrail, both worker HTTP contracts, Genie
failure propagation, rubric resolution, and class/worker imports from an isolated
packaged copy. Providers and workers are controlled in these tests; these results
do not establish live FEVM deployment or rubric judge quality on this branch.
