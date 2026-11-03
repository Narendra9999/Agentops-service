"""Team config loader.

Resolves team-specific deploy values from ``src/teams/<team_name>/config.yaml``.
Each pipeline notebook calls ``load_team_settings(team_name)`` to get a dict of
resolved values (agent_name, agent_endpoint, schema, vs_index, ...). Empty
``team_name`` returns an empty dict, letting callers fall back to their own
defaults.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

import yaml


def resolve_sql_warehouse_ids(
    config: dict[str, Any], workspace_client: Any = None,
    *, default_warehouse_name: str = "",
) -> dict[str, Any]:
    """Resolve configured warehouse names into IDs without modifying the source config."""
    resolved = deepcopy(config)
    default_name = default_warehouse_name.strip()
    parameters = resolved.get("parameters")
    if parameters is None:
        parameters = {}
        if default_name:
            resolved["parameters"] = parameters
    locations = [
        ("", resolved, "sql_warehouse_id", "sql_warehouse_name"),
        ("", resolved, "mlflow_tracing_warehouse_id", "mlflow_tracing_warehouse_name"),
        ("parameters.", parameters, "sql_warehouse_id", "sql_warehouse_name"),
        ("parameters.", parameters, "mlflow_tracing_warehouse_id", "mlflow_tracing_warehouse_name"),
        ("tracing.", resolved.get("tracing") or {}, "sql_warehouse_id", "sql_warehouse_name"),
        ("session_history.unity_catalog.", (resolved.get("session_history") or {}).get("unity_catalog") or {},
         "warehouse_id", "warehouse_name"),
    ]
    warehouses = None
    for prefix, block, id_key, name_key in locations:
        if prefix != "parameters." and id_key not in block and name_key not in block:
            continue
        warehouse_id = str(block.get(id_key) or "").strip()
        placeholder = warehouse_id.startswith("REQUIRED_")
        if warehouse_id and warehouse_id != "auto" and not placeholder:
            continue
        name = str(block.get(name_key) or "").strip()
        if not name or name.startswith("REQUIRED_"):
            name = default_name
        if not name:
            if placeholder:
                raise ValueError(
                    f"{prefix}{id_key} is a placeholder. Provide a real ID or {prefix}{name_key}."
                )
            continue
        if warehouses is None:
            if workspace_client is None:
                from databricks.sdk import WorkspaceClient
                workspace_client = WorkspaceClient()
            try:
                warehouses = list(workspace_client.warehouses.list())
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot resolve SQL warehouse '{name}': unable to list warehouses. "
                    "Check workspace credentials and warehouse permissions."
                ) from exc
        matches = [warehouse for warehouse in warehouses if warehouse.name == name]
        if len(matches) != 1 or not matches[0].id:
            raise ValueError(
                f"Expected one accessible SQL warehouse named '{name}' for {prefix}{name_key}; "
                f"found {len(matches)}. Check the exact name, workspace and permissions, "
                f"or set {prefix}{id_key} explicitly."
            )
        block[id_key] = matches[0].id
    return resolved


def _find_team_config_path(team_name: str, bundle_root: str | None = None) -> str:
    """Resolve team config path under ``src/teams/<team_name>``.

    ``bundle_root`` defaults to the repo root (two levels above this file).
    """
    if bundle_root is None:
        # src/framework/team_config.py → bundle root is two levels up
        bundle_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    team_dir = os.path.join(bundle_root, "src", "teams", team_name)
    yaml_path = os.path.join(team_dir, "config.yaml")
    yml_path = os.path.join(team_dir, "config.yml")
    if os.path.exists(yaml_path):
        return yaml_path
    if os.path.exists(yml_path):
        return yml_path
    return yaml_path


def load_team_settings(
    team_name: str,
    bundle_root: str | None = None,
    *, sql_warehouse_name: str = "",
) -> dict[str, Any]:
    """Read team config and return a dict of resolved deploy values.

    Returns ``{}`` when ``team_name`` is empty so callers can keep their
    existing widget defaults.

    Keys returned (when team config is found):
        agent_name, agent_endpoint, schema, audit_schema,
        vs_index, vs_endpoint, llm_endpoint, embedding_model,
        team_dir, team_config_path, serving_scale_to_zero,
        champion_model_version, champion_workload_size, champion_traffic_percentage,
        challenger_enabled, challenger_model_version, challenger_workload_size,
        challenger_traffic_percentage, ai_gateway_safety_enabled,
        ai_gateway_inference_tables_enabled, ai_gateway_rate_limit_per_user_per_minute,
        ai_gateway_rate_limit_per_endpoint_per_minute, ai_gateway_usage_tracking_enabled

    Deployment keys resolve to ``None`` (not a default) when the team config
    omits ``deployment:`` or a given field, so callers can distinguish "team
    didn't set this" from an explicit ``false``/``0`` and fall back to their
    own widget/pipeline default accordingly.

    Naming precedence:
        - ``naming.schema`` || ``agent.name``
        - ``naming.audit_schema`` || ``agent.name + "_audit"``
        - ``naming.agent_endpoint`` || ``naming.chatbot`` || ``agent.name``
    """
    if not team_name:
        return {}

    path = _find_team_config_path(team_name, bundle_root)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Team config not found: {path}. "
            f"Check team_name='{team_name}' and that src/teams/{team_name}/config.yaml "
            f"or src/teams/{team_name}/config.yml exists."
        )

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    agent_block = cfg.get("agent") or {}
    agent_name = agent_block.get("name")
    if not agent_name:
        raise ValueError(f"{path} is missing required field agent.name")

    naming = cfg.get("naming") or {}
    vs = cfg.get("vector_search") or {}
    vs_primary = vs.get("primary") or {}
    vs_indexes = vs.get("indexes") or []

    # Primary/fallback extraction for pipelines that still expect one value.
    primary_vs_index = vs_primary.get("index", "")
    primary_vs_endpoint = vs_primary.get("endpoint", "")
    if not primary_vs_index:
        # Backward compatibility: legacy single-index schema
        primary_vs_index = vs.get("index", "")
    if not primary_vs_endpoint:
        # Backward compatibility: legacy single-endpoint schema
        primary_vs_endpoint = vs.get("endpoint", "")

    if not primary_vs_index:
        for item in vs_indexes:
            if isinstance(item, dict) and item.get("index"):
                primary_vs_index = item.get("index")
                if not primary_vs_endpoint:
                    primary_vs_endpoint = item.get("endpoint", "")
                break
    llm = cfg.get("llm") or {}
    audit_schema = naming.get("audit_schema") or f"{agent_name}_audit"

    # Tracing — resolves automatically from naming/audit_schema so UC trace
    # storage wires up per-team with zero required config. Teams can still
    # override the warehouse or opt out via config.yaml `tracing:`.
    tracing = cfg.get("tracing") or {}
    parameters = cfg.get("parameters") or {}
    tracing_warehouse = {
        "sql_warehouse_id": tracing.get("sql_warehouse_id")
        or parameters.get("mlflow_tracing_warehouse_id") or cfg.get("mlflow_tracing_warehouse_id"),
        "sql_warehouse_name": tracing.get("sql_warehouse_name")
        or parameters.get("mlflow_tracing_warehouse_name") or cfg.get("mlflow_tracing_warehouse_name"),
    }
    if tracing.get("uc_trace_storage_enabled"):
        name = str(tracing_warehouse.get("sql_warehouse_name") or "").strip()
        if not name or name.startswith("REQUIRED_"):
            tracing_warehouse["sql_warehouse_name"] = sql_warehouse_name
        tracing_warehouse = resolve_sql_warehouse_ids(tracing_warehouse)

    # Production monitoring — continuously scores a sample of live traces with
    # the same builtin scorers used offline. Resolves from config.yaml
    # `evaluation.production_monitoring`; empty means "use pipeline defaults".
    monitoring = (cfg.get("evaluation") or {}).get("production_monitoring") or {}

    # Deployment — champion/challenger compute size, traffic split, and AI
    # Gateway settings. Missing keys resolve to None (not a default) so
    # callers can tell "team didn't set this" apart from an explicit
    # false/0 value and fall back to the pipeline widget accordingly.
    deployment = cfg.get("deployment") or {}
    champion_cfg = deployment.get("champion") or {}
    challenger_cfg = deployment.get("challenger") or {}
    ai_gateway_cfg = deployment.get("ai_gateway") or {}

    return {
        "agent_name": agent_name,
        "agent_endpoint": naming.get("agent_endpoint") or naming.get("chatbot") or agent_name,
        "catalog": naming.get("catalog", ""),
        "schema": naming.get("schema") or agent_name,
        "audit_schema": audit_schema,
        "vs_index": primary_vs_index,
        "vs_endpoint": primary_vs_endpoint,
        "llm_endpoint": llm.get("endpoint", ""),
        "embedding_model": vs.get("embedding_model", ""),
        "team_dir": team_name,
        "team_config_path": path,
        "tracing_uc_enabled": tracing.get("uc_trace_storage_enabled", False),
        "tracing_warehouse_id": tracing_warehouse.get("sql_warehouse_id") or "",
        "tracing_table_prefix": tracing.get("table_prefix") or agent_name,
        "tracing_schema": tracing.get("schema") or audit_schema,
        "monitoring_enabled": monitoring.get("enabled", True),
        "monitoring_sample_rate": monitoring.get("sample_rate"),
        "monitoring_safety_sample_rate": monitoring.get("safety_sample_rate"),
        "monitoring_results_table": monitoring.get("results_table") or "production_quality_scores",
        "serving_scale_to_zero": deployment.get("serving_scale_to_zero"),
        "champion_model_version": champion_cfg.get("model_version"),
        "champion_workload_size": champion_cfg.get("workload_size"),
        "champion_traffic_percentage": champion_cfg.get("traffic_percentage"),
        "challenger_enabled": challenger_cfg.get("enabled"),
        "challenger_model_version": challenger_cfg.get("model_version"),
        "challenger_workload_size": challenger_cfg.get("workload_size"),
        "challenger_traffic_percentage": challenger_cfg.get("traffic_percentage"),
        "ai_gateway_safety_enabled": ai_gateway_cfg.get("safety_enabled"),
        "ai_gateway_inference_tables_enabled": ai_gateway_cfg.get("inference_tables_enabled"),
        "ai_gateway_rate_limit_per_user_per_minute": ai_gateway_cfg.get("rate_limit_per_user_per_minute"),
        "ai_gateway_rate_limit_per_endpoint_per_minute": ai_gateway_cfg.get("rate_limit_per_endpoint_per_minute"),
        "ai_gateway_usage_tracking_enabled": ai_gateway_cfg.get("usage_tracking_enabled"),
    }


def merge_with_overrides(
    settings: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge resolved team settings with widget/var overrides.

    Override semantics: a non-empty override wins; an empty/None override
    falls back to the team setting.
    """
    merged = dict(settings)
    for key, val in overrides.items():
        if val:
            merged[key] = val
    return merged
