"""Public tracing destination configuration shared by team runtimes."""
import os
import threading

_cost_processor_lock = threading.Lock()


def use_experiment_destination(experiment_id):
    import mlflow
    from mlflow.entities import MlflowExperimentLocation
    if not experiment_id:
        raise ValueError("Experiment trace export requires a resolved experiment ID")
    mlflow.set_tracking_uri("databricks")
    mlflow.tracing.set_destination(MlflowExperimentLocation(str(experiment_id)))


def configure_trace_destination(config):
    tracing = config.get("tracing") or {}
    exporter = tracing.get("exporter")
    if exporter is None or exporter == "inference_table":
        return
    if exporter != "experiment":
        raise ValueError("tracing.exporter must be experiment or inference_table")
    use_experiment_destination(os.getenv("MLFLOW_EXPERIMENT_ID") or tracing.get("experiment_id"))


def configure_native_instrumentation(config):
    """Activate a supported native tracer before importing/loading the target.

    Autolog patches are process-wide. Select one integration at the model runtime
    boundary, not per request; omit this block to preserve external instrumentation.
    """
    import importlib
    options = (config.get("tracing") or {}).get("instrumentation")
    if options is None:
        return
    if not isinstance(options, dict) or set(options) - {"enabled", "integration"}:
        raise ValueError("tracing.instrumentation accepts enabled and integration")
    enabled = options.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("tracing.instrumentation.enabled must be boolean")
    integration = options.get("integration")
    if integration not in {"langchain", "openai"}:
        raise ValueError("tracing.instrumentation.integration must be langchain or openai")
    # Fail model loading rather than silently advertise tracing that is unavailable.
    module = importlib.import_module("mlflow." + integration)
    module.autolog(disable=not enabled, log_traces=enabled, silent=False)
    if enabled:
        configure_openai_cost_fallback(config)


def configure_openai_cost_fallback(config):
    """Price native model spans using the same rate card as response accounting."""
    model = (config.get("llm") or {}).get("endpoint")
    if not model:
        return
    from mlflow.tracing.config import get_config
    import mlflow
    from framework.observability.usage_tracking import TokenTracker
    from framework.observability.token_usage import set_native_cost_fallback
    if TokenTracker is None:
        return
    rates = {model: TokenTracker(model_name=model).pricing_per_million()}
    def fill_missing_cost(span):
        set_native_cost_fallback(span, rates)
    marker = "agentops.openai.cost." + model
    fill_missing_cost.__name__ = marker
    with _cost_processor_lock:
        processors = list(get_config().span_processors)
        if any(getattr(p, "__name__", "") == marker for p in processors):
            return
        mlflow.tracing.configure(span_processors=[*processors, fill_missing_cost])
