import ast
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

from framework import notebook_bootstrap
from framework.inference import extract_request_text, extract_response_text


NOTEBOOK = Path(__file__).resolve().parents[2] / (
    "src/agent_development/agent_evaluation/notebooks/ProductionQualityMonitor.py"
)


@pytest.fixture
def dashboard_functions():
    notebook = Path(__file__).resolve().parents[2] / (
        "src/agent_deployment/monitoring/notebooks/CreateAgentDashboard.py"
    )
    functions = [node for node in ast.parse(notebook.read_text()).body if isinstance(node, ast.FunctionDef)]
    namespace = {"WorkspaceClient": MagicMock(return_value=MagicMock()), "re": re,
                 "catalog": "default_catalog", "schema": "default_schema"}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(notebook), "exec"), namespace)
    return namespace


def test_agent_dashboard_discovers_payload_table_and_rewrites_template(dashboard_functions):
    client = dashboard_functions["WorkspaceClient"].return_value
    client.api_client.do.return_value = {"ai_gateway": {"inference_table_config": {
        "enabled": True, "catalog_name": "logging_catalog", "schema_name": "logging_schema",
        "table_name_prefix": "original-endpoint",
    }}}
    table = dashboard_functions["_resolve_inference_table"]("renamed-endpoint")
    assert table == "logging_catalog.logging_schema.`original-endpoint_payload`"
    client.api_client.do.assert_called_once_with("GET", "/api/2.0/serving-endpoints/renamed-endpoint")
    client.serving_endpoints.get.assert_not_called()
    template_path = Path(__file__).resolve().parents[2] / "resources/agent-dashboard-template.lvdash.json"
    template = json.loads(template_path.read_text())
    original_queries = ["".join(dataset.get("queryLines", [])) for dataset in template["datasets"]]
    result = dashboard_functions["_replace_payload_table"](template, table)
    for original, dataset in zip(original_queries, result["datasets"]):
        if "_payload" in original:
            assert table in "".join(dataset["queryLines"])
    client.warehouses.list.return_value = [SimpleNamespace(id="warehouse-id", state="RUNNING")]
    upsert = MagicMock(return_value=("dashboard-id", "updated"))
    dashboard_functions.update({
        "WorkspaceClient": MagicMock(return_value=client), "agent_endpoint": "renamed-endpoint",
        "inference_table_name": "", "template_path": str(template_path), "json": json,
        "_get_workspace_host_and_token": lambda: ("https://workspace.example", "test-token"),
        "_upsert_lakeview_dashboard": upsert,
    })
    notebook = Path(__file__).resolve().parents[2] / (
        "src/agent_deployment/monitoring/notebooks/CreateAgentDashboard.py"
    )
    nodes = ast.parse(notebook.read_text()).body
    start = next(index for index, node in enumerate(nodes) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "w" for target in node.targets))
    exec(compile(ast.Module(body=nodes[start:], type_ignores=[]), str(notebook), "exec"), dashboard_functions)
    assert upsert.call_args.kwargs["warehouse_id"] == "warehouse-id"
    assert table in upsert.call_args.kwargs["serialized"]
    assert "renamed-endpoint_payload" not in upsert.call_args.kwargs["serialized"]


@pytest.mark.parametrize("override", ["", "custom_catalog.custom_schema.custom_table"])
def test_agent_dashboard_override_and_fallback_match_monitor(dashboard_functions, override, capsys):
    client = dashboard_functions["WorkspaceClient"].return_value
    client.api_client.do.return_value = {"ai_gateway": None}
    dashboard_functions.update({"agent_endpoint": "endpoint", "inference_table_name": override})
    notebook = Path(__file__).resolve().parents[2] / (
        "src/agent_deployment/monitoring/notebooks/CreateAgentDashboard.py"
    )
    nodes = ast.parse(notebook.read_text()).body
    start = next(index for index, node in enumerate(nodes) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "w" for target in node.targets))
    end = next(index for index, node in enumerate(nodes) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "obj" for target in node.targets))
    exec(compile(ast.Module(body=nodes[start:end], type_ignores=[]), str(notebook), "exec"), dashboard_functions)
    assert dashboard_functions["target_table"] == (override or "default_catalog.default_schema.`endpoint_payload`")
    if override:
        client.api_client.do.assert_not_called()
        assert "Using explicit inference_table override:" in capsys.readouterr().out
    else:
        assert "falling back to default guess:" in capsys.readouterr().out


def test_agent_dashboard_resolver_matches_production_monitor():
    dashboard = Path(__file__).resolve().parents[2] / (
        "src/agent_deployment/monitoring/notebooks/CreateAgentDashboard.py"
    )
    resolvers = []
    for notebook in (NOTEBOOK, dashboard):
        resolver = next(node for node in ast.parse(notebook.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_inference_table")
        resolver.body = [node for node in resolver.body if not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )]
        resolvers.append(ast.dump(resolver))
    assert resolvers[0] == resolvers[1]


def test_agent_dashboard_uses_default_prefix_only_when_endpoint_config_omits_it(dashboard_functions):
    client = dashboard_functions["WorkspaceClient"].return_value
    client.api_client.do.return_value = {"ai_gateway": {"inference_table_config": {"enabled": True}}}
    assert dashboard_functions["_resolve_inference_table"]("endpoint") == (
        "default_catalog.default_schema.`endpoint_payload`"
    )


@pytest.mark.parametrize("endpoint", [
    {}, {"ai_gateway": None}, {"ai_gateway": {}},
    {"ai_gateway": {"inference_table_config": None}},
    {"ai_gateway": {"inference_table_config": {"enabled": False}}},
    {"ai_gateway": {"inference_table_config": {"table_name_prefix": "unconfirmed"}}},
])
def test_agent_dashboard_returns_none_for_missing_or_disabled_inference_logging(dashboard_functions, endpoint):
    client = dashboard_functions["WorkspaceClient"].return_value
    client.api_client.do.return_value = endpoint
    assert dashboard_functions["_resolve_inference_table"]("endpoint") is None


def test_agent_dashboard_reports_endpoint_lookup_failure(dashboard_functions, capsys):
    client = dashboard_functions["WorkspaceClient"].return_value
    client.api_client.do.side_effect = PermissionError("not authorized")
    assert dashboard_functions["_resolve_inference_table"]("endpoint") is None
    assert "WARNING: could not query endpoint 'endpoint'" in capsys.readouterr().out


def run_setup(monkeypatch, values, notebook_path):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    widgets = dict(values)
    dbutils = MagicMock()
    dbutils.widgets.text.side_effect = lambda name, default: widgets.setdefault(name, default)
    dbutils.widgets.get.side_effect = widgets.__getitem__
    dbutils.notebook.entry_point.getDbutils.return_value.notebook.return_value.getContext.return_value.notebookPath.return_value.get.return_value = notebook_path
    install = MagicMock()
    monkeypatch.setattr(notebook_bootstrap, "install_air_gapped_packages", install)
    tree = ast.parse(NOTEBOOK.read_text())
    setup_nodes = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "databricks.sdk":
            break
        setup_nodes.append(node)
    namespace = {"dbutils": dbutils}
    exec(compile(ast.Module(body=setup_nodes, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace, dbutils, install


@pytest.mark.parametrize("prefix", ["", "/Workspace"])
def test_monitor_bootstrap_normalizes_paths_and_preserves_widget_values(monkeypatch, prefix):
    namespace, dbutils, install = run_setup(monkeypatch, {
        "agent_name": "widget-agent", "agent_endpoint": "widget-endpoint",
        "results_table": "widget_scores", "sample_fraction": "0",
    }, f"{prefix}/Users/test/files/src/agent_development/agent_evaluation/notebooks/ProductionQualityMonitor")
    assert namespace["_bundle_root"] == "/Workspace/Users/test/files"
    assert namespace["_project_root"] == "/Workspace/Users/test/files/src"
    assert namespace["agent_name"] == "widget-agent"
    assert namespace["agent_endpoint"] == "widget-endpoint"
    assert namespace["results_table"] == "widget_scores"
    assert namespace["sample_fraction"] == 0
    install.assert_called_once()
    assert "mlflow>=3.7.0" in install.call_args.args[0]
    dbutils.library.restartPython.assert_called_once()


def test_monitor_bootstrap_applies_team_overrides(monkeypatch, tmp_path):
    team_dir = tmp_path / "src" / "teams" / "team-a"
    team_dir.mkdir(parents=True)
    (team_dir / "config.yaml").write_text(yaml.safe_dump({
        "agent": {"name": "team-agent"},
        "naming": {"schema": "team_schema", "agent_endpoint": "team-endpoint"},
        "evaluation": {"production_monitoring": {"results_table": "team_scores"}},
    }))
    monkeypatch.setattr(notebook_bootstrap, "resolve_bundle_paths", lambda dbutils: {
        "bundle_root": str(tmp_path), "src_root": str(tmp_path / "src"),
    })
    namespace, _, _ = run_setup(monkeypatch, {
        "team_name": "team-a", "agent_name": "widget-agent", "results_table": "widget_scores",
    }, "/unused")
    assert namespace["agent_name"] == "team-agent"
    assert namespace["schema"] == "team_schema"
    assert namespace["agent_endpoint"] == "team-endpoint"
    assert namespace["results_table"] == "team_scores"


def test_monitor_skip_exits_before_install_or_restart(monkeypatch):
    class NotebookExit(Exception):
        pass

    install = MagicMock()
    monkeypatch.setattr(notebook_bootstrap, "install_air_gapped_packages", install)
    dbutils = MagicMock()
    dbutils.widgets.get.return_value = "yes"
    dbutils.notebook.exit.side_effect = NotebookExit
    with pytest.raises(NotebookExit):
        exec(compile(NOTEBOOK.read_text(), str(NOTEBOOK), "exec"), {"dbutils": dbutils})
    install.assert_not_called()
    dbutils.library.restartPython.assert_not_called()


def test_monitor_scores_responses_and_legacy_payloads_but_skips_invalid_responses():
    records = [
        {"request": {"input": [{"role": "user", "content": "new question"}]},
         "response": {"output": [{"type": "message", "role": "assistant", "content": [
             {"type": "output_text", "text": "new answer"},
         ]}]}},
        {"request": {"messages": [{"role": "user", "content": "old question"}]},
         "response": {"messages": [{"role": "assistant", "content": "old answer"}]}},
        {"request": {"input": [{"role": "user", "content": "no answer"}]}, "response": {}},
        {"request": {"input": [{"role": "user", "content": "null answer"}]}, "response": None},
    ]
    payloads = pd.DataFrame([
        {"request": json.dumps(record["request"]), "response": json.dumps(record["response"]),
         "databricks_request_id": f"request-{index}"}
        for index, record in enumerate(records)
    ])
    spark = MagicMock()
    spark.table.return_value.where.return_value.where.return_value.sample.return_value.toPandas.return_value = payloads
    evaluate = MagicMock(return_value={"metrics": {}, "tables": {
        "eval_results": pd.DataFrame({"outputs": ["new answer", "old answer"]}),
    }})
    namespace = {
        "spark": spark, "inference_table": "catalog.schema.payloads", "sample_fraction": 0.5,
        "json": json, "datetime": datetime, "timezone": timezone,
        "extract_request_text": extract_request_text, "extract_response_text": extract_response_text,
        "run_parallel_evaluation": evaluate, "scorer_groups": {"builtin": ["scorer"]},
        "team_name": "team-a", "agent_name": "agent-a", "environment": "test",
        "_results_fqn": "catalog.schema.scores", "total_scorers": 1,
    }
    score_function = next(
        node for node in ast.parse(NOTEBOOK.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_score_one_day"
    )
    exec(compile(ast.Module(body=[score_function], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    result = namespace["_score_one_day"](date(2026, 10, 21))
    assert result["rows_scored"] == 2
    evaluated = evaluate.call_args.args[0]
    assert [row["outputs"] for row in evaluated] == ["new answer", "old answer"]
    written = spark.createDataFrame.call_args.args[0]
    assert list(written["request_id"]) == ["request-0", "request-1"]
    assert list(written["team_name"]) == ["team-a", "team-a"]
