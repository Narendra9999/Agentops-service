# Async AgentOPS runtime

AgentOPS exposes `await agent.apredict(request)` and
`async for event in agent.apredict_stream(request)` to async applications.
`AgentOPSBase` uses one async middleware lifecycle for prediction and streaming,
shared with its synchronous MLflow entry points. It awaits input checks, the customer hook,
output checks and persistence in their original order. Each async request gets
an isolated turn object, including its request context. Customer-specific shared
clients and mutable fields must still be safe for concurrent access.

## MLflow boundary

The MLflow ResponsesAgent loader used by this deployment calls `predict()`
synchronously and iterates `predict_stream()` synchronously. These methods must
return a ResponsesAgentResponse and a synchronous iterator respectively.
Making either method `async def` would break that contract. The synchronous
prediction method bridges to the async lifecycle; this does not make the managed
serving HTTP server an asyncio server. Async callers should use `apredict`, not
call `predict` on their event-loop thread.

Existing customer `_process_request` hooks remain supported in an executor.
For native async execution, override `_aprocess_request` and await the customer's
async model or graph client there. Framework code does not replace customer
routing or planner logic. Small normalization, regex and configuration functions
remain synchronous.

## Guardrails and storage

The built-in guardrails currently perform local checks, not remote LLM calls.
Their `acheck` adapters run existing checks off-loop, preserving tracing and
supporting existing custom synchronous checks. A remote guardrail integration
should implement native `async def acheck(...)` using an async HTTP/model client.
AgentOPS awaits that implementation. It does not run an agent before the input
checks pass or return a checked answer before output checks finish.

Lakebase/session SDK calls currently remain synchronous and run in the executor.
This is async compatibility, not a native async database driver migration.
Use `run_blocking` for legacy blocking integrations; new integrations should
prefer native awaitable clients. Cancellation cannot roll back an in-flight
SDK/database operation. Existing streaming guardrail release behavior is
preserved; this change does not introduce token buffering for legacy streams.

## Shared clients and usage

- `ainvoke_endpoint`: native `httpx.AsyncClient` HTTP, with SDK credential refresh
  off-loop. Uses the same validation/timeouts as the synchronous client and makes
  one request without automatic replay.
- `TeamRuntime.ainvoke`: the existing team request adapter plus async endpoint I/O.
- `ainvoke_with_usage_tracking`: native LangChain `ainvoke`, with the same callback
  selection and accounting as `invoke_with_usage_tracking`.
- `AsyncResponsesAgent`: compatibility async APIs available to team agents and the
  shared serving wrapper. A team's native implementation can override those APIs.

Streaming compatibility consumes each synchronous iterator on one worker to
preserve generator-local trace context. Its queue holds one event and applies
backpressure. Closing the async iterator requests cooperative cancellation and
closes the upstream iterator on its owning worker after any current blocking call
returns. Callers that stop early should use `contextlib.aclosing`.

This is not a claim that every SDK, tool, scorer, Spark job, or customer graph is
native async. Existing offline pipeline notebooks stay synchronous; inference
consumers and native integrations can use the async APIs without changing them.

## Validation

`tests/integration/test_async_runtime.py` checks event-loop responsiveness,
context propagation, concurrent request isolation, ordered streaming, error
propagation, bounded buffering/cleanup, async HTTP, native async guards/hooks,
and token accounting without model retries. Existing ResponsesAgent save/load,
streaming, guardrail and session tests cover synchronous compatibility.

## Shared lifecycle, deadlines and resource ownership

`AgentOPSBase._run_lifecycle` owns history loading, input guardrails, memory
recall, customer invocation, output guardrails, usage metrics and persistence.
Both prediction and streaming execute this implementation, with isolated turn
objects for async callers. Native streaming integrations override
`_aprocess_request_stream`; existing `_process_request_stream` hooks remain
supported off-loop. The synchronous streaming entry point uses one producer task
on its own loop, with a one-event queue, rather than creating a new task per token.

Optional server-side configuration:

```yaml
middleware:
  timeout_seconds: 120
```

Absent this setting, no additional whole-request deadline is imposed. A configured
value must be positive and finite. The deadline includes checks, agent work and
persistence. Native operations receive cancellation; an already executing blocking
SDK call may continue. Requests are never automatically replayed.

Request integrations can register an async resource with:

```python
client = await self._request_resources.enter_async_context(client_context())
```

The request scope closes resources on completion, errors, cancellation, timeout,
and explicit early stream closure. Creation, use and cleanup occur in the same
task. This provides an MCP-compatible ownership primitive; it is not an MCP
transport, tool registry, authentication implementation or persistent session pool.
Do not share these request resources across turns. Async consumers stopping early
must use `contextlib.aclosing`; synchronous consumers must close their iterator.

Custom storage integrations may expose native async `aget_history`,
`arecall_user_memories` and `asave_full_session`. Middleware prefers those methods;
otherwise it offloads the existing synchronous methods. History and memory
normalization remain shared across both paths. The built-in DatabricksStore
backend has not been migrated to a separate async database driver or schema.

Input blocks prevent invocation. Nonstreaming output blocks replace the answer.
Streaming output checks run after delivery and flag violations in trace tags;
they cannot retract delivered tokens. Interrupted streams do not persist partial
answers as completed sessions. A persistence operation already underway may
complete even if its caller is cancelled.

## Platform Intelligence async team

The Platform Intelligence team now implements native `apredict` and
`apredict_stream`. Its synchronous ResponsesAgent entry points are adapters.
Both supervisor graphs use async model/worker dispatch; their synchronous graph
APIs delegate to the same node implementation. The embedded documentation worker
awaits its prebuilt graph. Blocking SDK, Genie and Lakebase operations remain
thread-adapted with no claim of remote cancellation or rollback.

The pinned `databricks-langchain==0.8.0` implements synchronous model transport.
The team's `chat_model` factory adds `_agenerate` using a request-scoped
`httpx.AsyncClient`, refreshed SDK authentication and the configured serving or
AI Gateway base URL. It reuses the pinned package's input/tool/response converters
and makes one HTTP request without automatic retries. OpenAI typed responses are
used for conversion, without invoking an additional instrumented OpenAI client:
LangChain remains the sole model usage span. Async client contexts close on each model
call so clients are not reused across the synchronous serving adapter's loops.

Activity streaming uses a request-owned async task and a bounded activity queue.
Activity may be coalesced for slow consumers; final responses and failures are
never dropped. Final content and Genie artifacts remain buffered until the
AgentOPS prediction lifecycle and artifact checks finish. Closing the stream
cancels the execution task and awaits cleanup; an already executing blocking SDK
call may still complete. No private model reasoning is emitted.

## Serving trace export

`tracing.exporter: experiment` selects the public MLflow experiment exporter using
this deployment run's `MLFLOW_EXPERIMENT_ID`, with the registered model's resolved
`tracing.experiment_id` as fallback. An experiment name is resolved during
registration; customer configs do not need a pre-existing ID. This avoids the
MLflow 3.11 inference-table span processor's incompatibility with the serving
platform's injected OpenTelemetry version. Endpoint inference payload logging
remains an independently configured AI Gateway feature. Omitting the exporter
preserves the existing platform selection. The supervisor uses the same shared
experiment-export helper.

The shared documentation retrieval tool now uses `WorkspaceClient` authentication
for Vector Search rather than constructing a PAT-based client. This supports
serving OAuth identities and maps result rows using the returned column manifest.

### Provider streams and Playground

The shared Responses provider adapter emits `response.output_text.delta`,
`response.output_text.annotation.added` and `response.output_item.done` events.
Provider lifecycle envelopes such as `response.created` and `response.in_progress`
are consumed internally: Playground can reject them even though MLflow's generic
stream-event model permits them. `response.completed` records final usage without
being forwarded. Provider failure and truncated completion still raise errors.
The live single-agent check rejects unsupported event types anywhere in the
stream, rather than only validating the final completed item.

### Native usage on shared provider calls

The shared Responses API adapter creates one `LLM` span per provider request,
including streaming requests. The existing usage recorder exports provider token
counts to `mlflow.chat.tokenUsage` on that span and uses the existing TokenTracker
rate card through the common native-cost fallback. Cached input and reasoning
counts are retained; cost remains a configured-rate estimate. Missing usage or
pricing does not create a native zero-cost assertion.

AgentOPS summary tags and native MLflow header metrics describe the same calls.
No usage is attached to parent request/agent spans. Streaming records final usage
when the provider completes; truncated streams do not invent counts. Customer
agents with native LangChain tracing retain their existing model spans and do
not invoke this provider adapter. This does not intercept their model/tool calls.

Local regression checks verify actual exported trace totals for prediction and
streaming, cached-token cost parity, one provider request and one usage-bearing
span. FEVM uses the previous authenticated-deployer identity while the explicit
service-principal migration is deferred. See [deployment identities](deployment-identity.md).

Both direct Responses adapters and supervisor native-span processing call
`framework.observability.token_usage.export_native_model_usage`. The exporter
preserves existing model identity, native usage and cost; it only fills missing
attributes on LLM/CHAT_MODEL spans. Supervisor autologging remains the owner of
LangChain span creation. Its request-scoped rate card is passed to the shared
exporter; direct calls use TokenTracker's existing rate card. No additional
configuration or AgentObserver is required.

Configured per-call cost estimates are finalized after MLflow closes model spans,
so its built-in price lookup cannot silently replace the selected AgentOPS rate
card. Existing native costs remain untouched; parent spans are not priced.
