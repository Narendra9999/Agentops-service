# Async AgentOPS runtime

AgentOPS exposes `await agent.apredict(request)` and
`async for event in agent.apredict_stream(request)` to async applications.
`AgentOPSBase` uses one async request lifecycle for prediction, shared with its
synchronous `predict` entry point. It awaits input checks, the customer hook,
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
- `AsyncResponsesAgent`: compatibility async APIs for existing team agents and the
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
