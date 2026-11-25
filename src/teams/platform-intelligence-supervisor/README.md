# Platform Intelligence Supervisor

A team-owned LangGraph supervisor calls two existing workers:

- `platform_engineering`: a Responses-compatible serving endpoint.
- `cost_intelligence`: an existing Genie space, using the SDK conversations API.

Only this supervisor is registered and deployed. The platform endpoint defaults to
`fe-platform-single-validation`; the Genie binding uses the existing FEVM space.
Change `supervisor.workers` in `config.yaml` for your resources. The platform endpoint
must be a worker endpoint, not this supervisor's own endpoint. System identity
provides the supervisor model and platform worker access. Genie uses the invoking
caller's credentials and permissions.

## Ownership

`agent/PlatformintelligenceSupervisor.py` owns graph construction, routing
instructions and worker tools. `agent/workers/platform_engineering.py` and
`agent/workers/cost_intelligence.py` own the worker API calls. This is an ordinary
MLflow ResponsesAgent with no AgentOPS orchestration base class.

This team intentionally has no `Agent.py`. `inference.target_module` locates
`PlatformintelligenceSupervisor.py`; `inference.target_class` selects its class.
An explicit `inference.target_module` / `target_class` pair takes precedence.
If either field is absent, AgentOPS uses `Agent.py` (class: configured
`target_class`, otherwise `ChatModel`).
An invalid explicit target fails loading instead of falling back to another agent.
Dotted module names support nested packages (for example,
`supervisors.PlatformintelligenceSupervisor`). Prompt paths must still match the
packaged layout; `model.asset_mappings` can accommodate existing relative paths.
The framework loads context and wraps public calls. The shared framework handles
middleware,
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
Unhandled worker exceptions propagate to AgentOPS error handling; calls are not automatically
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

On this non-split branch, 176 local integration tests passed (excluding the
separate live-endpoint test module). These include the restored team contract,
request/stream middleware, injection guardrail, both worker HTTP contracts, Genie
failure propagation, rubric resolution, module-only registration selection, entry-point precedence, nested relative imports,
and class/worker imports from an isolated
packaged copy. Providers and workers are controlled in these tests; these results
do not establish live FEVM deployment or rubric judge quality on this branch.


## Caller-authorized Genie access

`model.auth_policy.user_api_scopes` declares `genie` and `sql` scopes at model
registration. Registration verifies that the saved policy matches the configured
policy. Other teams retain resources-only registration unless they opt in.
`deployment.environment_vars` configures the system identity separately; those
credentials are not used as a fallback for the Genie worker.

A workspace administrator must enable the Model Serving preview
`Agent Framework: On-Behalf-Of-User Authorization` (`agents_obo`) before deploying.
App OBO is a separate setting. The caller needs Genie, SQL warehouse and Unity
Catalog data permissions; scopes do not grant those permissions.

The Genie tool creates `ModelServingUserCredentials` inside each invocation.
It starts a conversation, waits for completion, and returns text plus query schema
and result data to the supervisor. This is the conversations API, not the Genie
Agent Responses API or STANDARD agent mode. Each tool call starts a new Genie
conversation. The connector returns the query-result chunk supplied by the SDK;
it does not fetch additional result chunks or render charts.

Missing caller credentials or permission denial produces the explicit
`Skipping Genie` tool result. Other worker failures propagate. Live validation
must check successful Genie execution and conversation ownership, rather than
accepting a successful HTTP response containing a skipped worker.

An App invoking the endpoint with its service principal represents that principal.
Forwarding a human user's identity requires separate App user authorization.
Remote platform internals continue to use that endpoint's own authentication
policy. Supervisor trace costs cover observed model calls, not complete remote
worker billing.

See [current FEVM OBO validation](FEVM_OBO_VALIDATION.md) for the merged branch test results.
