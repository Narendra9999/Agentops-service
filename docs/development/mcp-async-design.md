# MCP support in the async AgentOPS runtime

Async MCP tools are compatible with the synchronous MLflow ResponsesAgent
boundary. The entire MCP connection lifecycle must run inside an async runtime;
only the completed agent response or public stream crosses the serving boundary.
The Python return type at that boundary does not determine whether MCP can run.

This is a design and source review. MCP support has not been implemented or
validated in FEVM. The current async-runtime refactor is a prerequisite, not an
MCP session manager. The reviewed PicoAgents checkout was read statically; its
MCP dependencies were not installed or its MCP tests executed.

## The lifetime constraint

A connection has an owning event loop. MCP/AnyIO context managers may also require
enter and exit in the same task. Locks, transport tasks and live sessions cannot
be passed between fresh event loops. In particular, do not initialize an MCP
session in model load_context and call it from per-request asyncio.run loops.

The existing AgentOPS run_sync bridge creates a request loop. It supports MCP
only when session creation, discovery, invocation and cleanup all occur inside
that request. Existing run_blocking adapters are for blocking SDK calls, not for
moving live async MCP sessions between worker threads.

## Recommended first implementation: request-scoped sessions

```text
MLflow predict(request)
  -> one async request lifecycle
     -> await input guardrails
     -> open MCP session(s)
        -> discover/filter tools
        -> await customer agent / LangGraph ainvoke
           -> await MCP tool calls
        -> await output checks and persistence
     -> close MCP session(s) in their owning task
  -> ResponsesAgentResponse
```

Use explicit async context management with cleanup in finally, including tool
errors, guardrail rejection, timeout and cancellation. Opening/closing a context
within one coroutine keeps its task ownership clear. If a shared manager has
connect/disconnect invoked from different tasks, use one dedicated owner task per
connection, signaled to close by the manager.

This avoids global connection state and is the easiest mode to validate in
Serverless Model Serving. It repeats connection setup and tool discovery; measure
that overhead with the target servers. Retain the MCP session for the full turn,
not separately for every tool call. Customer-owned async orchestration receives
adapters for the configured tools; AgentOPS does not select tools or plan work.

The shared middleware owns only the integrations explicitly configured for it.
If the customer agent owns its MCP clients, its adapter must keep those resources
inside the customer's runtime; AgentOPS must not move them across event loops or
replace the customer's execution loop. Framework MCP support can be delivered
without changing the existing supervisor or its graph.

## Optional persistent sessions

For a FastAPI application, open connections in the application lifespan and close
them at shutdown on that application's event loop. Separate user credentials and
server sessions by authenticated identity; never use an email typed into a chat
as an authorization boundary.

For synchronous Model Serving with connection reuse, a dedicated runtime loop
thread per worker can host owner tasks. Sync predict submits the request with
run_coroutine_threadsafe; no request creates or destroys that shared loop. This
requires worker-start/shutdown integration, cancellation forwarding, bounded
in-flight work, health/reconnect handling and credential isolation. Create it
after worker startup, not before a process fork. Do not add this complexity until
request-scoped performance shows a need and serving lifecycle behavior is tested.

| Mode | Session scope | Main tradeoff |
|---|---|---|
| Request-scoped serving | One complete turn | Repeated handshake; simple cleanup/isolation |
| Async FastAPI lifespan | Worker loop plus auth scope | Reuse; explicit lifecycle and identity isolation |
| Dedicated loop for sync serving | Worker loop plus auth scope | Reuse; more lifecycle/cancellation engineering |

## PicoAgents findings

Reviewed checkout:
`/Users/narendra.merla/Documents/GitHub/multi_agent_systems/designing-multiagent-systems`.

| Component | Useful pattern | AgentOPS application |
|---|---|---|
| tools/_mcp/_client.py | Dedicated task owns client context; connection locks; discovery; managed cleanup | MCP lifecycle manager independent of customer orchestration |
| tools/_mcp/_tool.py | Namespaced tools; async execution; structured-result preference | Adapt discovered tools to LangChain/customer tool contracts |
| tools/_mcp/_config.py and _transports.py | Transport-specific configuration and construction | Team config with secret references and explicit transport |
| _middleware.py | Awaitable before/after/error processing for operations | Optional shared hooks around registered MCP calls |
| agents/_agent.py | Awaits tool execution and exposes cancellation/events | Reference for async contracts, not an agent-loop replacement |
| eval/_runner.py | Bounded async execution; fresh loop-local semaphores | Optional async evaluation runner with configured concurrency |

The checkout declares PicoAgents 0.5.0 as educational/alpha and requires
`mcp>=2.0.0` for this integration. It imports SDK 2.0 APIs such as
`mcp.client.Client`; do not assume those imports exist in the customer SDK or
copy older ClientSession examples without checking the deployed package.
Its optional dependencies also include model/telemetry pins that should not
replace AgentOPS's Databricks-compatible lock file.

The MCP package is MIT licensed. If source is copied, preserve required copyright
and license notices. Adapting the patterns does not require adopting PicoAgents
as a framework dependency.

Specific gaps to address rather than copy verbatim:

- create_mcp_tools connects servers in a loop without enclosing rollback. If a
  later connection fails, previously opened connections need explicit cleanup.
- The manager catches Exception in several lifecycle paths. Audit CancelledError
  separately, including cancellation while awaiting connection readiness and
  cleanup of an owner task not yet registered in the manager.
- The HTTP transport wrapper closes its custom client on context exit; an error
  during context entry also needs cleanup.
- MCPTool primarily returns structured content or concatenated text. Preserve
  image/resource/content blocks and metadata where the customer UI needs them.
- Add explicit connect/call/cleanup deadlines and bounded concurrency. Do not
  automatically replay a tool mutation after an uncertain timeout.
- Secret references, per-user credentials, complete schema validation, trace
  redaction and Databricks packaging must be integrated with AgentOPS config.

## AgentOPS integration shape

Keep a small optional package under framework/integrations/mcp with configuration,
session management and tool adapters. Team config declares servers and exposed
tools; teams retain their existing agent graph. Avoid wrapping tools that the
customer already manages unless they explicitly opt in.

Illustrative config, not yet an implemented schema:

```yaml
integrations:
  mcp:
    enabled: true
    session_scope: request
    max_concurrency: 4
    connect_timeout_seconds: 20
    call_timeout_seconds: 60
    servers:
      platform:
        transport: streamable_http
        url: ${PLATFORM_MCP_URL}
        auth_secret_ref: platform-mcp-credentials
        allowed_tools:
          - search_documentation
```

Resolve credentials using existing deployment secret mechanisms; the example
names do not create environment variables, secrets or authorization policies.
Prefer remote Streamable HTTP for the first FEVM serving implementation. Stdio
requires packaging and supervising a subprocess and is better introduced in local
tests separately. Confirm network reachability and authentication for each target.

An MCP call boundary can await configured input validation/authorization, record
one MLflow TOOL span, await invocation and inspect the result. Remote checks use
native async clients. Keep regex/schema operations synchronous. Reuse existing
tracing rather than introducing AgentObserver wrappers or duplicate spans.
Progress events expose tool name and status; arguments/results are logged only
according to configured redaction. MCP output is tool evidence, not instructions
that override the customer's system prompt.

## Validation before rollout

1. Async in-memory MCP server: discovery, schema mapping and successful calls.
2. Call error, connect timeout, mid-call cancellation, partial startup failure and
   cleanup; assert no pending session tasks or cross-loop/cancel-scope errors.
3. Concurrent requests with distinct user credentials; no cross-user sessions or
   request context. Test server restarts and auth refresh separately.
4. Sync ResponsesAgent and async caller parity, including stream close and trace
   parentage. Exercise a full async graph turn containing at least one MCP call.
5. FEVM HTTP MCP smoke through the original registration/evaluation/deploy pipeline;
   check guardrails, trace schema, tool provenance and cost/token coverage.
6. Measure handshake overhead and throughput before selecting persistent sessions.

No PicoAgents orchestrator, memory store, planner or UI is required for this work.
