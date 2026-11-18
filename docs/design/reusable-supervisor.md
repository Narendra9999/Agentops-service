# Reusable supervisor configuration

AgentOPS middleware remains independent of orchestration. The optional shared
LangGraph supervisor lives in `src/framework/orchestration/`; customers can also
wrap their own supervisors without using it. Only `pattern: supervisor` with
`implementation: langgraph` is implemented by this reusable builder.

```yaml
multi_agent:
  pattern: supervisor
  implementation: langgraph
  supervisor:
    model: system.ai.claude-sonnet-4-6
    api: ai_gateway
    instructions: configs/supervisor.md
  workers:
    platform_docs:
      description: Answer platform engineering questions with documentation evidence.
      transport: serving_endpoint
      endpoint_name: existing-platform-engineering
    cost_estimation:
      description: Analyze costs and estimate spending.
      transport: genie
      space_id: your-existing-space-id
```

Worker identifiers are arbitrary, unique tool-compatible names. Descriptions help
routing; instructions define team policies. Endpoint bindings contain serving
endpoint names, not URLs or registered model names. Genie bindings reference
existing spaces and use the team's existing agent-mode transport, not classic
chat. No workers or spaces are deployed by the supervisor.

The migrated platform-intelligence configuration retains its packaged retrieval
worker using `transport: in_process`, `implementation: platform_knowledge`, and
worker-local `settings.docs_index` / `settings.platform_model`. Local code must be
explicitly registered by the host; configuration cannot import arbitrary classes.
The previous `platform_mode`, `platform_endpoint`, `cost_endpoint`, and `agents`
block are replaced by the worker registry in the team YAML.

## Execution and ownership

1. Team `Agent.py` validates the schema, resolves its instructions file inside the
   team directory and initializes the existing ResponsesAgent target.
2. Team factories bind the configured transports to worker implementations. The
   shared registry validates all bindings before constructing workers.
3. Shared `supervisor.build_supervisor_graph` routes to registered workers and
   synthesizes their answers. Each worker runs at most once per turn. Async
   `ainvoke` is preferred; synchronous workers use the existing async bridge.
4. With authenticated memory enabled, shared `memory_supervisor.build_memory_graph`
   advertises the same configured worker names as tools and enforces memory-first
   execution. The team supplies the Lakebase workspace, identity checks, evidence
   policy, progress callbacks, and usage scopes. No second middleware lifecycle is
   introduced. Without authenticated context, the existing stateless policy applies.
5. AgentOPS retains outer guardrails, protocol adaptation, tracing and usage
   accounting. Existing worker transport adapters retain remote usage reporting.

For another team, use the shared builders with a model supporting structured
output/tool binding and worker factories returning `invoke`/`ainvoke` objects.
Workers accept `{'messages': [...]}` and return messages with a final AI answer;
optional `source_urls` and `genie_result` presentation metadata are preserved.
The shared graph does not import platform-intelligence modules.

Team-specific model creation, Genie/serving transport adapters, retrieval, memory
persistence and evaluation rules remain in the team package. This change provides
a reusable orchestration engine, not an automatic loader for arbitrary frameworks.

## Registration and prompts

The original AgentOPS registration pipeline and ResponsesAgent serving wrapper
are unchanged. Declare existing dependencies in `model.resources`; the migrated
team declarations bind to `multi_agent.workers.*` and `multi_agent.supervisor.*`.
When changing worker names or transports, update resource declarations, prompts
and team-specific evaluation rules accordingly. Keep the pipeline `llm` model/API
settings aligned with `multi_agent.supervisor`.

An explicit `supervisor.instructions` file overrides the pinned supervisor prompt
loaded from the manifest. Omit it to retain the existing Prompt Registry/artifact
selection. File instructions are versioned with the deployment; they are not
implicitly registered in Prompt Registry.

## Validation

Offline tests exercise a three-worker registry with unrelated identifiers,
invalid bindings rejected before factories run, arbitrary-name graph execution,
source metadata, prevention of repeated delegation, and memory-first tool dispatch.
Resource tests cover local retrieval, direct Genie and remote endpoint bindings.
FEVM deployment is required to validate the migrated artifact against live
services; local validation alone does not establish serving readiness.
