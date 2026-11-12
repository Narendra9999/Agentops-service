# Chat Completions and remote usage integration

The shared target adapter selects `responses` (default) or `chat_completions`
from `inference.target_protocol`. `inference.target_class` selects an alternate
class in the team's `Agent.py`; the default remains `ChatModel`. The registration
pipeline packages the shared serving wrapper, so the target does not need to
inherit an AgentOPS class or call middleware itself.

See [onboarding](agent-onboarding.md) for signatures and supported protocol fields.
The validation single-agent config selects the plain Chat Completions target.

## Remote accounting

`ExistingResponsesWorker` supplies a stable caller-span invocation ID to the
usage meter. Shared `remote_usage_record` reads standard response usage and
AgentOPS usage metrics/tags. Shared `rollup_remote_usage` adds each response
aggregate once, including known estimated USD cost, retaining worker trace IDs.
Repeated reports of the same execution/invocation are excluded. Distinct worker
calls sharing a distributed trace are not collapsed just because the trace ID
matches. Nested worker breakdowns are not summed again.

Where model-call IDs overlap with accounted calls, the ambiguous aggregate is
excluded and coverage marked partial. Missing worker tokens or prices remain
explicitly partial/unavailable. Unknown underlying model-call counts are reported
separately instead of assuming one model call per worker invocation.

This rollup changes response summaries and tags only. It does not manufacture
model spans carrying remote totals, which would double count joined worker spans
in MLflow's native aggregation. Parent native totals can differ from end-to-end
summary tags when worker traces are separate. Genie internals remain unmetered
unless that service reports usage.

## Validation

Local: 231 integration/supervisor tests and 59 unit tests passed in separate
processes. FEVM deployment and live remote-worker checks are in progress.
