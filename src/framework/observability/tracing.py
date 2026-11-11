"""Public tracing destination configuration shared by team runtimes."""
import os


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
