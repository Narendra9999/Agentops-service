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

Supervisor usage helpers are imported during module initialization rather than during request finalization. The registration check exposed a late import failure for the remote usage helper; the repaired registration and pre-deployment evaluation both passed.

Registration also uses a private temporary directory per run for runtime configuration. The previous supervisor registration failed because the fixed `/tmp/runtime_config.yaml` path was not writable on reused serverless compute.

## Local evidence

- 115 local integration tests passed, including prediction/streaming wrappers, Chat Completions conversion, cancellation and runtime bridges, controlled errors, input/output guardrail failures, persistence failures, and buffered-stream failure withholding output.
- 43 focused unit tests passed for usage, inference text, team runtime, model dependencies, and pipeline startup.
- Broader unit suite: one pre-existing collection error in `create_chunk.py` (a bare `test` expression); after excluding that module, 147 passed and 13 failed due missing dashboard/production-monitoring assets. These unrelated failures are not represented as passes.

## FEVM validation

Runs submitted through original team pipelines with the explicit `fevm-naren` profile:

- Single agent initial: `800909970315904` — all six pipeline stages SUCCESS, model v13.
- Single agent corrected accounting: `265743676169662` — all six stages SUCCESS, model v14.
- Supervisor: `292302771102679` — all six stages SUCCESS after registration repair and the configured evaluation retry; model v16.
- Additional serverless controlled-fault run: `927369038752282` (success, exception, deadline, cancellation, buffered-stream exception).

Controlled-fault serverless validation: **SUCCESS**. All five outcomes and real trace tags were asserted. Supervisor v16 is READY / NOT_UPDATING and all three direct endpoint scenarios passed. Single-agent v13 passed all pipeline stages, but direct streaming token reconciliation exposed a cumulative-snapshot issue. Its corrected rollout `265743676169662` passed all six stages with model v14. Direct prediction, streaming, and input-block checks all passed on v14 in READY / NOT_UPDATING state.

| Scenario | Outcome | MLflow trace |
|---|---|---|
| success | success | `tr-2fbe7556003e51fa54fe8577c7edad12` |
| failure | failed | `tr-c4ec9498c80b8a5995a930bad757ddfc` |
| timeout | failed | `tr-886b6f58b9a1dfbc28e7bc9582c84253` |
| cancelled | cancelled | `tr-d56c8afd9243ebb63cae553aef591cb0` |
| stream_failure | failed | `tr-64f50e76f3b7792cf4eecb293ac9d45a` |

## Limits

This change does not guarantee hard interruption of blocking SDK calls, cancellation of remote work, rollback of side effects, a new public HTTP error schema, or a new Responses failure event. Existing MLflow exception transport behavior remains in place. Cleanup still follows existing AsyncExitStack semantics, without a new cleanup timeout. Finalization is best effort if tracing infrastructure fails or the process is terminated.

## Endpoint accounting regression discovered during validation

On single-agent v13, prediction reconciled correctly, but streaming native usage retained the first gateway cumulative snapshot (3,957 tokens) while the response correctly reported the final snapshot (4,854 tokens), trace `tr-d2620c9ffd793016822f3cef8b92def8`. The single-agent Chat Completions demonstration target now exports only its final usage snapshot to the native model span. No inference or accounting assertion was removed. A regression test exercises multiple snapshots and verifies one final native export.

## Supervisor direct endpoint evidence

- cost: `tr-e5cd709da12b067460b2048f1d6c9790`, workers cost_estimation; native token/cost reconciliation and one pre/post guardrail invocation passed.
- platform: `tr-f17a03f81d43f337ad40766f85174bb5`, workers platform_docs; native token/cost reconciliation and one pre/post guardrail invocation passed.
- combined: `tr-c0addfb9729574548357a0304974b90c`, workers cost_estimation, platform_docs; native token/cost reconciliation and one pre/post guardrail invocation passed.

Supervisor citation-quality caveat: the first post-deployment evaluation scored platform_evidence 0.667 against 0.8, while other metrics and guardrails passed. The configured retry passed unchanged gates. Preserve the initial failure for follow-up: evaluation ID `d3c69ea5-4636-4110-9eef-33e2dba2201e`. This rollout does not claim the intermittent citation-quality issue is resolved.

## Final single-agent endpoint evidence

- prediction: `tr-0bcbb37866ccc8fd355716217a89a5bc`, status passed, tokens 3399, estimated cost USD 0.004337.
- streaming: `tr-af57d2aabeb5eb40c509639794aaafbd`, status passed, tokens 5024, estimated cost USD 0.006092.
- input_block: `tr-48095cc11c8cac8457fb0b6af75a536c`, status passed, tokens None, estimated cost USD None.

Final result: both original AgentOPS pipelines completed successfully; single-agent v14 and supervisor v16 were verified after endpoint updates settled. All six direct endpoint scenarios passed, as did five controlled FEVM serverless fault scenarios. Local regression evidence: 115 integration tests and 43 focused unit tests passed. Broader unrelated unit-suite failures remain documented above.

Commits: `6d1ba90` (shared lifecycle and error diagnostics), `c9c00b6` (final streaming usage snapshot and expanded tests).
