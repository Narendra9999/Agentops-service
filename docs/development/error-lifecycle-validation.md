# Shared error lifecycle implementation and validation

Implemented in the existing AgentOPS callback lifecycle; no middleware chain or target retry was introduced.

## Behavior

- `process_error(error, context)` classifies exceptions using stable allowlisted codes; exception text is not put into diagnostic tags. The original exception propagates.
- Request state tracks preparation, input guardrails, memory, target, output guardrails, persistence, and completion stages.
- A once-only best-effort finalizer runs after success, rejection, failure, cancellation, or generator close. Existing successful usage emission is retained; missing finalization paths export available usage and outcome tags without adding model calls.
- `CancelledError` and `GeneratorExit` retain control-flow semantics. Diagnostic failures do not replace the original exception.
- Failed/cancelled execution marks usage coverage partial. This is conservative; it does not assert that every provider call was metered.
- Error codes: `deadline_or_dependency_timeout`, `dependency_unavailable`, and `execution_failed`. The combined timeout code avoids incorrectly attributing a dependency timeout to the outer request deadline.
- Standard response processing/redaction stays unchanged. No new successful response envelope is fabricated for operational failures.

Observable tags: `agentops.status`, `agentops.execution.stage`, `agentops.error.code` on failures, `agentops.latency_ms`, and conservative `agentops.usage.status` for interrupted execution.

Supervisor usage helpers are imported during module initialization rather than during request finalization. The registration check exposed a late import failure for the remote usage helper; the repaired rollout will validate this change.

Registration also uses a private temporary directory per run for runtime configuration. The previous supervisor registration failed because the fixed `/tmp/runtime_config.yaml` path was not writable on reused serverless compute.

## Local evidence

- 110 local integration tests passed, including prediction/streaming wrappers, Chat Completions conversion, cancellation and runtime bridges, controlled errors, and buffered-stream failure withholding output.
- 43 focused unit tests passed for usage, inference text, team runtime, model dependencies, and pipeline startup.
- Broader unit suite: one pre-existing collection error in `create_chunk.py` (a bare `test` expression); after excluding that module, 147 passed and 13 failed due missing dashboard/production-monitoring assets. These unrelated failures are not represented as passes.

## FEVM validation

Runs submitted through original team pipelines with the explicit `fevm-naren` profile:

- Single agent: `800909970315904`.
- Supervisor: `292302771102679`.
- Additional serverless controlled-fault run: `927369038752282` (success, exception, deadline, cancellation, buffered-stream exception).

Controlled-fault serverless validation: **SUCCESS**. All five outcomes and real trace tags were asserted. Endpoint pipeline validation remains in progress.

| Scenario | Outcome | MLflow trace |
|---|---|---|
| success | success | `tr-2fbe7556003e51fa54fe8577c7edad12` |
| failure | failed | `tr-c4ec9498c80b8a5995a930bad757ddfc` |
| timeout | failed | `tr-886b6f58b9a1dfbc28e7bc9582c84253` |
| cancelled | cancelled | `tr-d56c8afd9243ebb63cae553aef591cb0` |
| stream_failure | failed | `tr-64f50e76f3b7792cf4eecb293ac9d45a` |

## Limits

This change does not guarantee hard interruption of blocking SDK calls, cancellation of remote work, rollback of side effects, a new public HTTP error schema, or a new Responses failure event. Existing MLflow exception transport behavior remains in place. Cleanup still follows existing AsyncExitStack semantics, without a new cleanup timeout. Finalization is best effort if tracing infrastructure fails or the process is terminated.
