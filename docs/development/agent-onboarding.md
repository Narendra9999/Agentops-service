# Integrating an independent agent with AgentOPS

AgentOPS owns the outer request lifecycle. Team agents own orchestration, model
calls and tools. A team does not need to inherit `AgentOPSBase` or
`AsyncResponsesAgent` to use the shared pipeline.

## Minimal ResponsesAgent integration

For an existing ResponsesAgent implementation:

```python
# src/teams/my-team/agent/Agent.py
from customer_package import MySupervisorAgent

class ChatModel(MySupervisorAgent):
    pass
```

The example assumes the customer agent has a no-argument constructor. Override
`load_context(context)` if it needs configuration or packaged artifacts. Declare
its dependencies in the team's requirements file and configure guardrails,
tracing, history and deployment settings in the team configuration. Use the
shared pipeline's team-agent registration path, which packages the framework
serving wrapper as the MLflow model entry point. Registering only the customer
agent directly with MLflow does not apply AgentOPS middleware.

`RegisterModel.py` packages `CostTrackingResponsesModel`; its `load_context()`
loads `ChatModel`, calls the target's loader, then constructs
`AgentOPSMiddleware(config, target)`.

Prediction follows:

```text
MLflow loader
  -> CostTrackingResponsesModel.predict
  -> wrapper.apredict
  -> middleware.execute(invoke=adapter.invoke)
  -> AgentOPSBase._run_lifecycle
  -> input checks / history / memory
  -> adapter.invoke -> target prediction
  -> output checks / metrics / persistence
  -> ResponsesAgentResponse
```

Streaming follows:

```text
MLflow loader
  -> CostTrackingResponsesModel.predict_stream
  -> wrapper.apredict_stream
  -> middleware.execute_stream(invoke=adapter.invoke, stream=adapter.stream)
  -> AgentOPSBase._run_lifecycle
  -> input checks / history / memory
  -> adapter.stream -> target streaming
  -> output checks / metrics / persistence
  -> checked ResponsesAgent output events
```

If a target does not implement streaming, the wrapper uses one prediction and
emits its completed output. No additional model invocation is needed.

The supervisor never needs to call `AgentOPSBase.predict_stream()`: middleware
enters the common lifecycle through `execute_stream()` instead.

## Optional framework hooks

The included single documentation agent extends `AgentOPSBase` to use framework
model-call helpers and context-aware processing hooks. The adapter calls its raw
hooks with `RequestContext`, bypassing its public lifecycle to avoid duplicate
checks. This is an optional integration style, not a customer inheritance rule.

Our supervisor extends `AsyncResponsesAgent` to provide convenient synchronous
MLflow boundaries over its async implementation. Plain ResponsesAgents are also
supported; their synchronous methods are offloaded by the adapter.

Shared targets keep request state local or receive it through explicit context.
Agent instances and clients are reused. No shallow or deep copies of agents are
needed. Each middleware invocation owns a new `RequestContext`, deadline and
resource stack.

## Chat Completions implementations

Select the protocol and target class explicitly:

```yaml
inference:
  target_protocol: chat_completions
  target_class: ChatCompletionsModel
  max_stream_chars: 1000000
```

`Agent.py` exposes that class with `predict(messages=..., custom_inputs=...)`
and optionally `predict_stream` with the same arguments. Native async variants
`apredict` and `apredict_stream` are also supported. Return decoded SDK objects
or dictionaries, not raw SSE lines. Targets construct their own clients and
execute their own orchestration. The adapter never retries model or tool calls.

The shared adapter converts text messages, input images, function call history,
final text/function calls, streamed text and fragmented function-call arguments.
It supports one completion choice, preserves final reported usage including
cached/reasoning details, and accepts terminal usage-only chunks. Unsupported
content types fail explicitly. Truncated/content-filtered/refused streams cannot
be persisted as successful answers. Buffered output is released only after checks.
Tools returned for the caller are handoffs, not automatically executed actions.

The existing single-agent validation team includes a plain
`ChatCompletionsModel` using its existing RAG preparation and native model spans.
Its configured public protocol is translated by the shared framework. The model
span owns native usage; the outer adapter records one aggregate for tags without
annotating a parent span with those tokens again.

## Internal tools versus the external response

For a request to estimate a batch workload and explain deployment options, a
customer supervisor might execute:

```text
Supervisor -> model: choose next action
Model -> supervisor: tool_call(cost_intelligence, arguments)
Supervisor -> Cost Intelligence: execute worker
Cost Intelligence -> supervisor: result
Supervisor -> model: continue with worker result
Model -> supervisor: tool_call(platform_knowledge, arguments)
Supervisor -> Platform Knowledge: execute worker
Platform Knowledge -> supervisor: retrieved documents
Supervisor -> model: synthesize answer
Model -> supervisor: final text
Supervisor -> adapter: external answer or public stream
```

The adapter invokes the supervisor once. It does not execute those intermediate
model tool calls or feed results back into the supervisor. Taking over that loop
would require knowing the customer's routing, state, retries and worker APIs.
Native instrumentation can still trace those internal operations without making
AgentOPS their orchestrator. Outer guardrails do not automatically validate every
internal tool argument or worker response.

For streaming, a supervisor may intentionally expose public progress events such
as “Calling Cost Intelligence,” followed by answer deltas and a completed answer.
The adapter translates supported public events; it does not automatically publish
raw internal routing messages or private reasoning. If the target intentionally
returns an unresolved tool request for its caller to execute, the integration
needs an explicit tool-handoff contract; it must not treat that request as a
completed text answer.

## Remote worker trace visibility

A local tool span can record a remote endpoint's request, response and latency
without containing the endpoint's internal spans. Without propagation:

```text
Trace A: supervisor -> remote-worker invocation -> final synthesis
Trace B: worker -> retrieval -> worker model
```

With supported distributed trace continuation:

```text
Trace A: supervisor
  -> remote-worker invocation
     -> worker execution
        -> retrieval
        -> worker model
  -> final synthesis
```

The caller must inject tracing context and the worker must extract it and export
compatible spans. W3C `traceparent` is a common carrier; sending a header alone
does not guarantee that a managed endpoint honors it. If continuation is not
supported, record the worker trace ID and parent execution ID for correlation;
these are still separate traces. Do not assume parent token totals automatically
include usage from an unrelated worker trace.

Our custom App opts into `custom_inputs.stream_activity: true`, forwards
`custom_outputs.activity` from supported stream events, and renders a live
activity timeline. This is independent of MLflow span visibility. Targets must
publish public activity intentionally. Progress delivered before final output
checks should contain safe operational summaries, not raw tool arguments,
sensitive results or private reasoning. Playground/Review App clients should not
be assumed to render this custom activity protocol.
