"""Unit tests for framework.team_config.load_team_settings."""

import ast
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from framework.team_config import (
    load_team_settings,
    merge_with_overrides,
    resolve_sql_warehouse_ids,
)


@pytest.mark.parametrize("resource_name", [
    "pipeline-resource.yml", "batch-inference-resource.yml", "dashboard-pipeline-resource.yml",
    "iterative-improvement-resource.yml", "production-monitoring-resource.yml",
])
def test_jobs_notify_email_recipients_on_start_success_and_failure(resource_name):
    root = Path(__file__).resolve().parents[2]
    resource = yaml.safe_load((root / "resources" / resource_name).read_text())
    jobs = resource["resources"]["jobs"]
    assert jobs
    for job in jobs.values():
        assert job["email_notifications"] == {
            event: ["${var.notification_emails}"] for event in ("on_start", "on_success", "on_failure")
        }


def test_notification_recipient_uses_a_single_string_setting():
    root = Path(__file__).resolve().parents[2]
    inputs = json.loads((root / "dab-config.json").read_text())["usecase_specific_variables"]
    assert "alert_mail_id" not in inputs
    assert isinstance(inputs["notification_emails"], str)
    assert "@" in inputs["notification_emails"]
    bundle = yaml.load((root / "databricks.yml").read_text(), Loader=yaml.BaseLoader)
    assert "alert_mail_id" not in bundle["variables"]
    assert bundle["variables"]["notification_emails"].get("type", "string") == "string"
    document = yaml.compose((root / "databricks.yml").read_text())
    variables = next(value for key, value in document.value if key.value == "variables")
    recipients = next(value for key, value in variables.value if key.value == "notification_emails")
    default = next(value for key, value in recipients.value if key.value == "default")
    assert default.tag == "tag:yaml.org,2002:str"
    assert default.value == inputs["notification_emails"]
    targets = next(value for key, value in document.value if key.value == "targets")
    for _, target in targets.value:
        overrides = next(value for key, value in target.value if key.value == "variables")
        for key, value in overrides.value:
            if key.value == "notification_emails":
                assert isinstance(value, yaml.ScalarNode)
                assert value.tag in ("tag:yaml.org,2002:str", "!notification_emails")


def test_resolve_sql_warehouse_names_into_runtime_config():
    client = MagicMock()
    client.warehouses.list.return_value = [
        SimpleNamespace(name="analytics", id="analytics-id"),
        SimpleNamespace(name="tracing", id="tracing-id"),
    ]
    config = {
        "parameters": {
            "sql_warehouse_id": "REQUIRED_PROD_SQL_WAREHOUSE_ID",
            "sql_warehouse_name": "analytics",
            "mlflow_tracing_warehouse_id": "REQUIRED_PROD_TRACING_WAREHOUSE_ID",
            "mlflow_tracing_warehouse_name": "tracing",
        },
        "tracing": {"sql_warehouse_name": "tracing"},
        "session_history": {"unity_catalog": {"warehouse_id": "auto", "warehouse_name": "analytics"}},
    }
    result = resolve_sql_warehouse_ids(config, client)
    assert result["parameters"]["sql_warehouse_id"] == "analytics-id"
    assert result["parameters"]["mlflow_tracing_warehouse_id"] == "tracing-id"
    assert result["tracing"]["sql_warehouse_id"] == "tracing-id"
    assert result["session_history"]["unity_catalog"]["warehouse_id"] == "analytics-id"
    assert config["parameters"]["sql_warehouse_id"] == "REQUIRED_PROD_SQL_WAREHOUSE_ID"
    client.warehouses.list.assert_called_once_with()


@pytest.mark.parametrize("config", [
    {},
    {"sql_warehouse_id": "explicit-id", "sql_warehouse_name": "ignored"},
    {"parameters": {"sql_warehouse_id": "explicit-id"}},
    {"session_history": {"unity_catalog": {"warehouse_id": "auto"}}},
])
def test_warehouse_ids_and_unused_warehouses_do_not_require_lookup(config):
    client = MagicMock()
    assert resolve_sql_warehouse_ids(config, client) == config
    client.warehouses.list.assert_not_called()


@pytest.mark.parametrize("warehouse_id", [None, "", "REQUIRED_PROD_SQL_WAREHOUSE_ID"])
def test_top_level_warehouse_name_resolves_missing_id(warehouse_id):
    client = MagicMock()
    client.warehouses.list.return_value = [SimpleNamespace(name="analytics", id="resolved-id")]
    result = resolve_sql_warehouse_ids({"sql_warehouse_id": warehouse_id, "sql_warehouse_name": "analytics"}, client)
    assert result["sql_warehouse_id"] == "resolved-id"


@pytest.mark.parametrize("matches", [[], [
    SimpleNamespace(name="analytics", id="first"),
    SimpleNamespace(name="analytics", id="second"),
]])
def test_warehouse_lookup_rejects_missing_or_ambiguous_names(matches):
    client = MagicMock()
    client.warehouses.list.return_value = matches
    with pytest.raises(ValueError, match="Expected one accessible SQL warehouse named 'analytics'"):
        resolve_sql_warehouse_ids({"sql_warehouse_name": "analytics"}, client)


def test_warehouse_lookup_reports_permission_errors():
    client = MagicMock()
    client.warehouses.list.side_effect = PermissionError("not allowed")
    with pytest.raises(RuntimeError, match="Check workspace credentials and warehouse permissions"):
        resolve_sql_warehouse_ids({"sql_warehouse_name": "analytics"}, client)


def test_warehouse_placeholder_requires_id_or_name():
    with pytest.raises(ValueError, match="Provide a real ID or parameters.sql_warehouse_name"):
        resolve_sql_warehouse_ids({"parameters": {"sql_warehouse_id": "REQUIRED_PROD_SQL_WAREHOUSE_ID"}})


@pytest.mark.parametrize("parameters", [
    None, {}, {"sql_warehouse_id": ""},
    {"sql_warehouse_id": "REQUIRED_PROD_SQL_WAREHOUSE_ID",
     "mlflow_tracing_warehouse_id": "REQUIRED_PROD_TRACING_WAREHOUSE_ID"},
    {"sql_warehouse_name": "REQUIRED_PROD_SQL_WAREHOUSE_NAME"},
])
def test_bundle_warehouse_name_fills_missing_team_parameters(parameters):
    client = MagicMock()
    client.warehouses.list.return_value = [SimpleNamespace(name="bundle-warehouse", id="bundle-id")]
    config = {"parameters": parameters} if parameters is not None else {}
    result = resolve_sql_warehouse_ids(config, client, default_warehouse_name="bundle-warehouse")
    assert result["parameters"]["sql_warehouse_id"] == "bundle-id"
    assert result["parameters"]["mlflow_tracing_warehouse_id"] == "bundle-id"
    assert "sql_warehouse_id" not in config or config["sql_warehouse_id"] != "bundle-id"
    client.warehouses.list.assert_called_once_with()


def test_team_warehouse_ids_and_names_override_bundle_default():
    client = MagicMock()
    client.warehouses.list.return_value = [SimpleNamespace(name="team-tracing", id="team-tracing-id")]
    config = {"parameters": {"sql_warehouse_id": "team-sql-id", "sql_warehouse_name": "ignored",
                             "mlflow_tracing_warehouse_name": "team-tracing"}}
    result = resolve_sql_warehouse_ids(config, client, default_warehouse_name="bundle-warehouse")
    assert result["parameters"]["sql_warehouse_id"] == "team-sql-id"
    assert result["parameters"]["mlflow_tracing_warehouse_id"] == "team-tracing-id"


def test_existing_warehouse_blocks_use_bundle_fallback_without_creating_unused_blocks():
    client = MagicMock()
    client.warehouses.list.return_value = [SimpleNamespace(name="bundle-warehouse", id="bundle-id")]
    config = {"sql_warehouse_id": "", "tracing": {"sql_warehouse_id": ""},
              "session_history": {"unity_catalog": {"warehouse_id": "auto"}}}
    result = resolve_sql_warehouse_ids(config, client, default_warehouse_name="bundle-warehouse")
    assert result["sql_warehouse_id"] == "bundle-id"
    assert result["tracing"]["sql_warehouse_id"] == "bundle-id"
    assert result["session_history"]["unity_catalog"]["warehouse_id"] == "bundle-id"
    assert "mlflow_tracing_warehouse_id" not in result


def test_explicit_team_warehouse_ids_skip_bundle_lookup():
    client = MagicMock()
    config = {"parameters": {"sql_warehouse_id": "sql-id", "mlflow_tracing_warehouse_id": "tracing-id"}}
    assert resolve_sql_warehouse_ids(config, client, default_warehouse_name="bundle-warehouse") == config
    client.warehouses.list.assert_not_called()


def test_bundle_warehouse_parameter_reaches_registration():
    root = Path(__file__).resolve().parents[2]
    bundle = yaml.compose((root / "databricks.yml").read_text())
    variables = next(value for key, value in bundle.value if key.value == "variables")
    warehouse = next(value for key, value in variables.value if key.value == "sql_warehouse_name")
    default = next(value for key, value in warehouse.value if key.value == "default")
    assert default.tag == "!sql_warehouse_name"
    pipeline = yaml.safe_load((root / "resources/pipeline-resource.yml").read_text())
    task = next(task for task in pipeline["resources"]["jobs"]["agentops_pipeline"]["tasks"]
                if task["task_key"] == "register_model")
    assert task["notebook_task"]["base_parameters"]["sql_warehouse_name"] == "${var.sql_warehouse_name}"
    nodes = ast.parse((root / "src/agent_development/agent/notebooks/RegisterModel.py").read_text())
    defaults = next(node.value for node in nodes.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "widget_defaults" for target in node.targets))
    assert ast.literal_eval(defaults)["sql_warehouse_name"] == ""
    calls = [node for node in ast.walk(nodes) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id in {"load_team_settings", "resolve_sql_warehouse_ids"}]
    assert len(calls) == 2
    for call in calls:
        keyword = "sql_warehouse_name" if call.func.id == "load_team_settings" else "default_warehouse_name"
        value = next(argument.value for argument in call.keywords if argument.arg == keyword)
        assert isinstance(value, ast.Name) and value.id == "w_sql_warehouse_name"


@pytest.mark.parametrize("tracing", [
    {"uc_trace_storage_enabled": True},
    {"uc_trace_storage_enabled": True, "sql_warehouse_name": "team-tracing"},
    {"uc_trace_storage_enabled": True, "sql_warehouse_id": "explicit-tracing-id"},
    {"uc_trace_storage_enabled": False},
])
def test_tracing_bundle_fallback_respects_team_precedence(tmp_path, tracing):
    _write_team_config(str(tmp_path), "analytics", {"agent": {"name": "analytics"}, "tracing": tracing})
    client = MagicMock()
    client.warehouses.list.return_value = [
        SimpleNamespace(name="bundle-warehouse", id="bundle-id"),
        SimpleNamespace(name="team-tracing", id="team-tracing-id"),
    ]
    with patch("databricks.sdk.WorkspaceClient", return_value=client):
        settings = load_team_settings("analytics", str(tmp_path), sql_warehouse_name="bundle-warehouse")
    if not tracing["uc_trace_storage_enabled"]:
        assert settings["tracing_warehouse_id"] == ""
        client.warehouses.list.assert_not_called()
    elif tracing.get("sql_warehouse_id"):
        assert settings["tracing_warehouse_id"] == "explicit-tracing-id"
        client.warehouses.list.assert_not_called()
    else:
        assert settings["tracing_warehouse_id"] == ("team-tracing-id" if tracing.get("sql_warehouse_name") else "bundle-id")
        client.warehouses.list.assert_called_once_with()


def test_registration_resolves_and_declares_team_agent_warehouses():
    notebook = Path(__file__).resolve().parents[2] / "src/agent_development/agent/notebooks/RegisterModel.py"
    nodes = ast.parse(notebook.read_text()).body
    resolution = next(node for node in nodes if isinstance(node, ast.Assign)
                      and isinstance(node.value, ast.Call)
                      and isinstance(node.value.func, ast.Name)
                      and node.value.func.id == "resolve_sql_warehouse_ids")
    serialization = next(node for node in nodes if isinstance(node, ast.With)
                         and isinstance(node.items[0].context_expr, ast.Call)
                         and isinstance(node.items[0].context_expr.args[0], ast.Name)
                         and node.items[0].context_expr.args[0].id == "runtime_config")
    assert resolution.lineno < serialization.lineno
    client = MagicMock()
    client.warehouses.list.return_value = [SimpleNamespace(name="analytics", id="resolved-id")]
    namespace = {
        "agent_config": {"parameters": {"sql_warehouse_name": "analytics", "mlflow_tracing_warehouse_name": "analytics"}},
        "resolve_sql_warehouse_ids": lambda config, **kwargs: resolve_sql_warehouse_ids(config, client, **kwargs),
        "w_sql_warehouse_name": "",
        "DatabricksSQLWarehouse": lambda **kwargs: kwargs,
        "resources": [],
    }
    start = next(index for index, node in enumerate(nodes) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "session_cfg" for target in node.targets))
    end = next(index for index, node in enumerate(nodes[start:], start) if isinstance(node, ast.For)
               and isinstance(node.target, ast.Name) and node.target.id == "warehouse_id")
    exec(compile(ast.Module(body=[resolution, *nodes[start:end + 1]], type_ignores=[]), str(notebook), "exec"), namespace)
    assert namespace["agent_config"]["parameters"]["sql_warehouse_id"] == "resolved-id"
    assert namespace["agent_config"]["parameters"]["mlflow_tracing_warehouse_id"] == "resolved-id"
    assert namespace["resources"] == [{"warehouse_id": "resolved-id"}]


@pytest.mark.parametrize("tracing_enabled", [True, False])
def test_team_tracing_settings_resolve_names_only_when_enabled(tmp_path, tracing_enabled):
    _write_team_config(str(tmp_path), "analytics", {
        "agent": {"name": "analytics"},
        "tracing": {"uc_trace_storage_enabled": tracing_enabled},
        "parameters": {"mlflow_tracing_warehouse_name": "tracing"},
    })
    client = MagicMock()
    client.warehouses.list.return_value = [SimpleNamespace(name="tracing", id="tracing-id")]
    with patch("databricks.sdk.WorkspaceClient", return_value=client):
        settings = load_team_settings("analytics", str(tmp_path))
    assert settings["tracing_warehouse_id"] == ("tracing-id" if tracing_enabled else "")
    assert client.warehouses.list.call_count == int(tracing_enabled)


def test_registration_preserves_resolved_config_yml_path(tmp_path):
    path = Path(_write_team_config(str(tmp_path), "analytics", {"agent": {"name": "analytics"}}))
    path.rename(path.with_suffix(".yml"))
    settings = load_team_settings("analytics", str(tmp_path))
    notebook = Path(__file__).resolve().parents[2] / "src/agent_development/agent/notebooks/RegisterModel.py"
    nodes = ast.parse(notebook.read_text()).body
    selection = next(node for node in nodes if isinstance(node, ast.If)
                     and any(isinstance(child, ast.Assign)
                             and any(isinstance(target, ast.Name) and target.id == "team_config"
                                     for target in child.targets) for child in node.body))
    namespace = {"team_name": "analytics", "team_config": "", "team_settings": settings,
                 "bundle_root": str(tmp_path), "os": os}
    exec(compile(ast.Module(body=[selection], type_ignores=[]), str(notebook), "exec"), namespace)
    assert namespace["team_config"] == "src/teams/analytics/config.yml"


def _write_team_config(bundle_root: str, team_name: str, cfg: dict) -> str:
    team_dir = os.path.join(bundle_root, "src", "teams", team_name)
    os.makedirs(team_dir, exist_ok=True)
    path = os.path.join(team_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def test_empty_team_name_returns_empty_dict():
    assert load_team_settings("") == {}


def test_missing_team_raises():
    with tempfile.TemporaryDirectory() as root:
        with pytest.raises(FileNotFoundError, match="src/teams/missing"):
            load_team_settings("missing", bundle_root=root)


def test_naming_block_wins_over_derivation():
    with tempfile.TemporaryDirectory() as root:
        _write_team_config(root, "team-a", {
            "agent": {"name": "team_a_agent"},
            "naming": {
                "schema": "team_a_schema",
                "audit_schema": "team_a_audit_custom",
                "chatbot": "team-a-bot",
            },
            "vector_search": {"index": "team_a_idx", "endpoint": "team-a-vs"},
            "llm": {"endpoint": "databricks-gpt-oss-120b"},
        })
        s = load_team_settings("team-a", bundle_root=root)
        assert s["agent_name"] == "team_a_agent"
        assert s["schema"] == "team_a_schema"
        assert s["audit_schema"] == "team_a_audit_custom"
        assert s["agent_endpoint"] == "team-a-bot"
        assert s["vs_index"] == "team_a_idx"
        assert s["vs_endpoint"] == "team-a-vs"
        assert s["llm_endpoint"] == "databricks-gpt-oss-120b"
        assert s["team_dir"] == "team-a"


def test_naming_falls_back_to_derivation():
    with tempfile.TemporaryDirectory() as root:
        _write_team_config(root, "team-b", {
            "agent": {"name": "team_b_agent"},
            "vector_search": {"index": "team_b_idx"},
        })
        s = load_team_settings("team-b", bundle_root=root)
        assert s["agent_name"] == "team_b_agent"
        assert s["schema"] == "team_b_agent"
        assert s["audit_schema"] == "team_b_agent_audit"
        assert s["agent_endpoint"] == "team_b_agent"


def test_naming_uses_agent_endpoint_if_provided():
    with tempfile.TemporaryDirectory() as root:
        _write_team_config(root, "team-d", {
            "agent": {"name": "team_d_agent"},
            "naming": {"agent_endpoint": "team-d-endpoint"},
        })
        s = load_team_settings("team-d", bundle_root=root)
        assert s["agent_endpoint"] == "team-d-endpoint"


def test_missing_agent_name_raises():
    with tempfile.TemporaryDirectory() as root:
        _write_team_config(root, "team-c", {"vector_search": {"index": "x"}})
        with pytest.raises(ValueError, match="agent.name"):
            load_team_settings("team-c", bundle_root=root)


def test_merge_with_overrides_non_empty_wins():
    settings = {"agent_name": "team_a", "schema": "team_a_schema", "vs_index": "team_a_idx"}
    overrides = {"agent_name": "override", "schema": "", "vs_index": None}
    merged = merge_with_overrides(settings, overrides)
    assert merged["agent_name"] == "override"
    assert merged["schema"] == "team_a_schema"
    assert merged["vs_index"] == "team_a_idx"


def test_deployment_block_parses_champion_challenger_and_ai_gateway():
    with tempfile.TemporaryDirectory() as root:
        _write_team_config(root, "team-e", {
            "agent": {"name": "team_e_agent"},
            "deployment": {
                "serving_scale_to_zero": False,
                "champion": {"model_version": "3", "workload_size": "Medium", "traffic_percentage": 80},
                "challenger": {"enabled": True, "model_version": "4", "workload_size": "Small", "traffic_percentage": 20},
                "ai_gateway": {
                    "inference_tables_enabled": True,
                    "rate_limit_per_user_per_minute": 0,
                    "rate_limit_per_endpoint_per_minute": 100,
                    "usage_tracking_enabled": False,
                },
            },
        })
        s = load_team_settings("team-e", bundle_root=root)
        assert s["serving_scale_to_zero"] is False
        assert s["champion_model_version"] == "3"
        assert s["champion_workload_size"] == "Medium"
        assert s["champion_traffic_percentage"] == 80
        assert s["challenger_enabled"] is True
        assert s["challenger_model_version"] == "4"
        assert s["challenger_traffic_percentage"] == 20
        assert s["ai_gateway_rate_limit_per_user_per_minute"] == 0
        assert s["ai_gateway_usage_tracking_enabled"] is False


def test_deployment_block_defaults_to_none_when_absent():
    with tempfile.TemporaryDirectory() as root:
        _write_team_config(root, "team-f", {"agent": {"name": "team_f_agent"}})
        s = load_team_settings("team-f", bundle_root=root)
        assert s["serving_scale_to_zero"] is None
        assert s["champion_workload_size"] is None
        assert s["challenger_enabled"] is None
        assert s["ai_gateway_usage_tracking_enabled"] is None
