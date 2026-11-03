"""
AgentOPS Framework — Unity Catalog Trace Storage

Binds a team's MLflow experiment to a Unity Catalog trace location (OTel
Delta tables) at creation time, so every deployed agent's traces persist
beyond the 100k-per-experiment cap and are queryable via SQL/Genie/AI-BI —
with no per-team wiring. Called once, from RegisterModel.py, the single
place each pipeline run creates/upserts the shared experiment.
"""

import logging
import os

import mlflow

logger = logging.getLogger(__name__)

# Shared fallback warehouse already granted for AgentOPS UC writes (see
# session_history.unity_catalog.warehouse_id in agent config.yaml).
DEFAULT_TRACING_WAREHOUSE_ID = "78563e1df52edcf3"


def ensure_tracing_warehouse_env(warehouse_id: str = None) -> str:
    """Set MLFLOW_TRACING_SQL_WAREHOUSE_ID so MLflow can read/write UC-backed traces."""
    resolved = (
        warehouse_id
        or os.environ.get("MLFLOW_TRACING_SQL_WAREHOUSE_ID")
        or DEFAULT_TRACING_WAREHOUSE_ID
    )
    os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = resolved
    return resolved


def configure_uc_trace_experiment(
    experiment_name: str,
    catalog: str,
    schema: str,
    table_prefix: str,
    warehouse_id: str = None,
    enabled: bool = True,
    spark=None,
):
    """
    Bind ``experiment_name`` to a Unity Catalog trace location so every trace
    lands in ``{catalog}.{schema}.{table_prefix}_otel_*`` Delta tables.

    Falls back to standard experiment-managed trace storage (with a warning)
    when UC storage is disabled, catalog/schema are missing, or the experiment
    is already bound to a different location — an experiment's trace location
    can only be set at creation time, so re-runs against an existing
    experiment created before this wiring existed must not fail the pipeline.
    """
    mlflow.set_tracking_uri("databricks")

    if not enabled or not catalog or not schema:
        logger.info(
            "UC trace storage disabled or catalog/schema missing for '%s' — "
            "using experiment-managed trace storage.", experiment_name,
        )
        return mlflow.set_experiment(experiment_name)

    ensure_tracing_warehouse_env(warehouse_id)

    if spark is not None:
        try:
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
        except Exception as e:
            logger.warning(f"Could not ensure UC schema {catalog}.{schema} exists: {e}")

    try:
        from mlflow.entities.trace_location import UnityCatalog

        experiment = mlflow.set_experiment(
            experiment_name=experiment_name,
            trace_location=UnityCatalog(
                catalog_name=catalog,
                schema_name=schema,
                table_prefix=table_prefix,
            ),
        )
        logger.info(
            "MLflow experiment '%s' bound to UC trace location %s.%s.%s_otel_* "
            "(experiment_id=%s)", experiment_name, catalog, schema, table_prefix,
            experiment.experiment_id,
        )
        return experiment
    except Exception as e:
        logger.warning(
            "Could not bind '%s' to UC trace location %s.%s.%s: %s. "
            "Falling back to experiment-managed trace storage.",
            experiment_name, catalog, schema, table_prefix, e,
        )
        return mlflow.set_experiment(experiment_name=experiment_name)
