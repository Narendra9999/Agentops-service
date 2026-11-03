import ast
from enum import Enum
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest


class EndpointMissing(Exception):
    pass


def endpoint(versions, updating=False):
    return SimpleNamespace(
        state=SimpleNamespace(config_update="IN_PROGRESS" if updating else "NOT_UPDATING"),
        config=SimpleNamespace(served_entities=[
            SimpleNamespace(entity_name="catalog.schema.agent", entity_version=version)
            for version in versions
        ]),
        pending_config=None,
    )


def deployment_namespace(snapshots):
    source = Path(__file__).resolve().parents[2] / "src/agent_development/agent/notebooks/DeployAgent.py"
    tree = ast.parse(source.read_text())
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "wait_for_endpoint_update"]
    recovery = next(
        node for node in tree.body if isinstance(node, ast.Try)
        and isinstance(node.body[0], ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "deployment" for target in node.body[0].targets)
    )
    workspace = MagicMock()
    workspace.serving_endpoints.get.side_effect = snapshots
    namespace = {
        "w": workspace,
        "agents": MagicMock(),
        "agent_endpoint": "agent-endpoint",
        "NotFound": EndpointMissing,
        "time": SimpleNamespace(monotonic=MagicMock(side_effect=count()), sleep=MagicMock()),
        "deploy_champion": MagicMock(side_effect=[ValueError("same signature"), SimpleNamespace(endpoint_name="agent-endpoint")]),
    }
    return namespace, compile(ast.Module(body=helpers + [recovery], type_ignores=[]), str(source), "exec")


def test_incompatible_cleanup_waits_for_each_version_to_disappear():
    namespace, code = deployment_namespace([
        endpoint(["4", "5", "9"]),
        endpoint(["4", "5", "9"], updating=True),
        endpoint(["4", "5", "9"]),
        endpoint(["5", "9"]),
        endpoint(["5", "9"], updating=True),
        endpoint(["9"]),
        EndpointMissing("last deployment removed"),
    ])
    exec(code, namespace)
    assert namespace["w"].serving_endpoints.get.call_count == 7
    assert namespace["agents"].delete_deployment.call_args_list == [
        call("catalog.schema.agent", "4"),
        call("catalog.schema.agent", "5"),
        call("catalog.schema.agent", "9"),
    ]
    assert namespace["time"].sleep.call_count == 3
    assert namespace["deploy_champion"].call_count == 2


def test_incompatible_cleanup_does_not_retry_deployment_after_failed_delete():
    namespace, code = deployment_namespace([endpoint(["4", "5"])])
    namespace["agents"].delete_deployment.side_effect = PermissionError("not allowed")
    with pytest.raises(RuntimeError, match="Failed to remove"):
        exec(code, namespace)
    assert namespace["agents"].delete_deployment.call_count == 1
    assert namespace["deploy_champion"].call_count == 1


@pytest.mark.parametrize("update, error", [
    ("IN_PROGRESS", "cleanup did not finish"),
    ("UPDATE_FAILED", "configuration update failed"),
])
def test_endpoint_wait_is_bounded_and_reports_failed_updates(update, error):
    snapshot = endpoint(["4"])
    snapshot.state.config_update = update
    namespace, code = deployment_namespace([snapshot])
    namespace["deploy_champion"].side_effect = None
    exec(code, namespace)
    with pytest.raises(RuntimeError, match=error):
        namespace["wait_for_endpoint_update"]("agent-endpoint", max_wait_seconds=2)


def test_cleanup_accepts_sdk_enum_states_and_legacy_served_models():
    snapshot = endpoint([])
    snapshot.state.config_update = Enum("ConfigUpdate", {"NOT_UPDATING": "NOT_UPDATING"}).NOT_UPDATING
    snapshot.config = SimpleNamespace(served_models=[
        SimpleNamespace(model_name="catalog.schema.agent", model_version="4"),
    ])
    namespace, code = deployment_namespace([snapshot, endpoint([])])
    exec(code, namespace)
    namespace["agents"].delete_deployment.assert_called_once_with("catalog.schema.agent", "4")
    assert namespace["deploy_champion"].call_count == 2
