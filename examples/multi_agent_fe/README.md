# Platform intelligence supervisor example

The implementation now lives under
[`src/teams/platform-intelligence`](../../src/teams/platform-intelligence/README.md)
and uses the standard AgentOPS `agent/Agent.py` / `ChatModel` contract.

Only the existing supervisor **ResponsesAgent** is packaged and deployed. It calls
two existing workers by configuration: Platform knowledge (ResponsesAgent endpoint)
and Cost intelligence (Genie STANDARD agent or ResponsesAgent endpoint). The case
worker, mutation tools and approval UI are removed. Lakebase App conversation
history and preferences remain independent and unchanged.

The App retains activity events, stop, cleared message composer, conversation
rename/history and interactive Genie charts. `model.py` is the direct ResponsesAgent
entry point for standalone smoke tests. AgentOPS uses the standard framework wrapper
and the team adapter, preserving guardrails, custom outputs and usage.

Install the pinned dependencies with `pip install -r requirements.in` and run:

```bash
python -m pytest examples/multi_agent_fe/tests -q
node --test examples/multi_agent_fe/tests/test_charts.cjs
```

All Databricks scripts require an explicit profile. The FEVM snapshot uploader
includes the canonical team package, framework and contract-test notebook. The
serverless contract check uses mocked existing workers and deploys no worker:

```bash
python examples/multi_agent_fe/scripts/fevm_bootstrap.py \
  --profile fevm-naren \
  --notebook examples/multi_agent_fe/notebooks/12_AgentOPSSupervisorContract.py
```

See the team README for config bindings, pipeline setup, architecture, cost coverage
and remaining deployment gaps. `FEVM_STATUS.md` and `VALIDATION.md` retain historical
rollout evidence. Earlier three-worker versions are historical; the new team setup
requires the existing knowledge endpoint name before real-worker deployment.
