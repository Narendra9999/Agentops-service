# Configuration-driven native instrumentation

AgentOPS activates supported MLflow native instrumentation at model load, before importing the team target. Customers do not need to add observer context managers, tracing callbacks, or model-span decorators for operations supported by the selected integration.

```yaml
tracing:
  instrumentation:
    enabled: true
    integration: langchain  # LangGraph/LangChain; use openai for OpenAI SDK calls
```

Only `langchain` and `openai` are currently supported. Choose the integration at the orchestration boundary: enabling both the LangChain wrapper and its underlying SDK tracer can produce nested spans for the same model call. Instrumentation is process-wide and configured during model initialization, never toggled per request. Separate model runtimes are required for independent settings. An omitted block leaves external instrumentation unchanged. Explicit `enabled: false` disables the selected native integration, while AgentOPS middleware spans remain enabled. The setting cannot remove explicit decorators or tracing installed by customer code.

The original `multi_agent.settings.observer_enabled` remains a separate optional custom-span facility. It does not control native tracing. The FEVM supervisor keeps it disabled; its LangChain native instrumentation is enabled through the shared setting above.

The shared entry point is `CostTrackingResponsesModel.load_context()`. Both single-agent and supervisor deployments use it. The target class and protocol remain selected through `inference`; the supervisor's team entry point no longer enables autologging itself. The single-agent Chat Completions test target no longer creates its own model spans or exports native model usage manually. Its response usage still feeds shared accounting. Native OpenAI spans receive a missing-cost fallback from the same AgentOPS model rate card; provider/native costs take precedence. These costs are estimates, not billing records.

Supported internal failures appear on native child spans. A failure recovered by the agent need not fail the whole request. Errors escaping the agent are handled by the shared middleware. A disabled tracer does not disable middleware error handling. Unsupported custom operations, swallowed exceptions outside instrumented calls, and remote-worker internals cannot be inferred from a public response. Remote workers still require their own tracing integration and supported trace propagation.

## Validation

`tests/integration/test_native_instrumentation.py` checks configuration validation, enabled/disabled calls, preservation of external instrumentation when omitted, and initialization before customer code import/load. Existing adapter tests verify that cumulative stream usage is accounted once.

`examples/multi_agent_fe/scripts/validate_native_instrumentation.py` runs eight isolated serverless scenarios: each integration enabled/disabled, each with successful/handled-failure execution. Its mocked HTTP transport injects errors without modifying or disrupting deployed worker endpoints. Real deployed single-agent and supervisor requests are validated separately after the original pipeline quality gates and endpoint rollout.
