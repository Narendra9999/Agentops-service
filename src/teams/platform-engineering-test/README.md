# Single-agent FEVM regression fixture

This team runs the existing `agent/Agent.py` through the original six AgentOPS
notebooks. The agent implementation is unchanged. The bundle isolates its model,
evaluation tables, experiment and endpoint (`fe-platform-single-validation`) from
the existing `platform-agent` endpoint.

The FEVM target uses the team lockfile and the existing documentation index.
Registration resolves `/Shared/agentops-platform-single-validation` dynamically;
no experiment ID is configured in advance. Persistent memory and AgentObserver
are not enabled. This fixture does not deploy an App.

## Validation status

Run `1064333894131807` registered model version 1, but its pre-deployment quality
gate correctly blocked deployment. Correctness, relevance and documentation
evidence scored 0.667 against the unchanged 0.8 thresholds; safety and the
17 guardrail fixtures passed. The Auto Loader question retrieved unrelated
content and the agent disclosed missing evidence. The tracing question also
failed the documentation evidence scorer. Retrieval coverage and answer quality
need improvement before this fixture can be deployed. No quality gate was lowered
and the existing `platform-agent` endpoint was not modified.
