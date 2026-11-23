# Standalone chat App assets

The retired platform-intelligence agent backend, its notebooks, setup scripts
and daily-monitor job have been removed. This directory retains the FastAPI
App, frontend assets and their tests; it does not deploy an agent.

The active team is [platform-intelligence-supervisor](../../src/teams/platform-intelligence-supervisor/README.md).
Its agent and worker connectors use the normal shared AgentOPS pipeline.

The App deployment bundle requires an explicit existing Responses-compatible
endpoint and a context-signing secret binding. The retained App includes
history/memory UI features that the simple supervisor does not implement;
retention here does not claim those features are integrated with that supervisor.
No live App or serving resource was changed by this source cleanup.
