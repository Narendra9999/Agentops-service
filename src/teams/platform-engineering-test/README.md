# Single-agent FEVM regression fixture

This team runs the existing `agent/Agent.py` through the original six AgentOPS
notebooks. The agent implementation is unchanged. The bundle isolates its model,
evaluation tables, experiment and endpoint (`fe-platform-single-validation`) from
the existing `platform-agent` endpoint.

The FEVM target uses the team lockfile and the existing documentation index.
Registration resolves `/Shared/agentops-platform-single-validation` dynamically;
no experiment ID is configured in advance. Persistent memory and AgentObserver
are not enabled. This fixture does not deploy an App.

## FEVM validation: version 5

[Run 503615624058011](https://fevm-classic-stable-naren.cloud.databricks.com/jobs/180098491824601/runs/503615624058011?o=7474644082064164)
deployed version 5 through the original six-stage AgentOPS pipeline. Pre- and
post-deployment evaluation scored 1.0 for correctness, relevance, safety and
platform evidence; guardrail accuracy was 1.0. Quality thresholds and test
questions were retained.

Direct serving checks passed prediction, streaming and input blocking. Both
answer traces contain nonempty retrieval results and official documentation
URLs. Blocked input produced no model-token usage. The endpoint is READY and
NOT_UPDATING, routing all traffic to version 5.

Earlier runs exposed missing Auto Loader corpus coverage, weak source links,
PAT-only retrieval authentication in serving, and an incompatible platform
inference-table trace processor. The fixes add official documentation chunks,
require retrieved source links in this team's prompt, use WorkspaceClient's
credential handling for Vector Search, and select the public MLflow experiment
exporter. AI Gateway inference logging remains separate. The original agent
implementation is retained; the existing `platform-agent` endpoint is untouched.

The shared async lifecycle and optional PII middleware are packaged. Existing
blocking policies remain enabled; redaction is opt-in. Combined local framework,
middleware and supervisor validation passed 252 tests. App deployment remains
deferred.

Sanitized results: [validation-async-runtime.json](deployment/validation-async-runtime.json).
