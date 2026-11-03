"""
AgentOPS Framework — Notebook Bootstrap Utilities

Shared setup boilerplate for the pipeline notebooks (RegisterModel,
PreDeploymentEval, DeployAgent, PostDeploymentEval): widget declaration,
notebook/bundle path resolution, team-config merging, audit-pipeline start,
and air-gapped dependency installation.

Keeping this in one place means every notebook installs from the same
Mastercard shared volume path (``framework.mlops_utils.get_pip_find_links``)
and resolves team overrides with the same precedence rules, instead of each
notebook re-implementing (and slowly drifting from) the same logic.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

from framework.mlops_utils import get_pip_find_links
from framework.team_config import load_team_settings, merge_with_overrides

TRUE_VALUES = {"1", "true", "yes", "y"}


def to_bool(value: str) -> bool:
    """Parse a widget string value as a boolean flag (yes/true/1/y)."""
    return str(value).strip().lower() in TRUE_VALUES


def read_widgets(dbutils, defaults: dict[str, str]) -> dict[str, str]:
    """Declare + read a batch of text widgets in one pass.

    ``defaults`` maps widget name -> default value. Returns the raw
    (unstripped) string read back from each widget, in the same order.
    """
    for name, default in defaults.items():
        dbutils.widgets.text(name, default)
    return {name: dbutils.widgets.get(name) for name in defaults}


def workspace_abs(path: str) -> str:
    """Normalize a notebook path across interactive and job-cluster contexts."""
    if not path:
        return path
    return path if path.startswith("/Workspace/") else f"/Workspace{path}"


def resolve_bundle_paths(dbutils) -> dict[str, str]:
    """Resolve the notebook/bundle/src paths shared by every pipeline notebook.

    Returns a dict with:
        nb_path: Workspace-absolute path of the running notebook
        nb_dir: Directory containing the notebook
        src_root: ``.../files/src`` (the project's ``src`` directory)
        bundle_root: ``.../files`` (bundle root — where ``pyproject.toml`` lives)
    """
    nb_path = workspace_abs(
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    )
    nb_dir = os.path.dirname(nb_path)
    src_root = os.path.dirname(os.path.dirname(os.path.dirname(nb_dir)))
    bundle_root = os.path.dirname(src_root)
    return {
        "nb_path": nb_path,
        "nb_dir": nb_dir,
        "src_root": src_root,
        "bundle_root": bundle_root,
    }


def install_air_gapped_packages(packages: list[str], python_version: str = "312") -> str:
    """Install packages from the shared Mastercard air-gapped volume (no PyPI access).

    Every pipeline notebook must resolve the same volume path — routed through
    ``framework.mlops_utils.get_pip_find_links`` so there is a single source of
    truth instead of each notebook hardcoding its own copy of the path.

    Returns the resolved volume path (callers that need it for a follow-up
    install, e.g. model-dependency installs, can reuse it without recomputing).
    """
    vol_path = get_pip_find_links(python_version)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", *packages,
        "--find-links", vol_path, "--no-index", "-q",
    ])
    return vol_path


def resolve_team_settings(team_name: str, bundle_root: str, widget_values: dict[str, Any]) -> dict[str, Any]:
    """Resolve team config and overlay it onto widget values.

    Team config wins for any key it defines (non-empty); otherwise the widget
    value is kept. Equivalent to the ``_pick(team_val, widget_val)`` pattern
    repeated across notebooks, but backed by the shared/tested
    ``merge_with_overrides`` helper.

    Only suitable for string-valued fields where "" means unset — for
    booleans/numbers where an explicit ``False``/``0`` is a meaningful team
    override, use ``pick_if_not_none`` instead (this truthy-based merge would
    otherwise silently discard them in favor of the widget value).
    """
    settings = load_team_settings(team_name, bundle_root=bundle_root) if team_name else {}
    return merge_with_overrides(widget_values, settings)


def pick_if_not_none(team_val, widget_val):
    """Team config wins when explicitly set (including False/0); otherwise
    the widget value is kept. Use for booleans/numbers, where a falsy-but-set
    team value must not be mistaken for "team didn't set this"."""
    return team_val if team_val is not None else widget_val


def resolve_mlflow_experiment_id(dbutils, provided: str = "", upstream_task_key: str = "register_model") -> str:
    """Return the pipeline-provided MLflow experiment_id, falling back to the
    upstream task's published task value (set by RegisterModel)."""
    provided = (provided or "").strip()
    if provided:
        return provided
    try:
        resolved = dbutils.jobs.taskValues.get(
            taskKey=upstream_task_key, key="mlflow_experiment_id", debugValue="",
        ).strip()
        if resolved:
            print(f"Resolved mlflow_experiment_id from {upstream_task_key} task values: {resolved}")
        return resolved
    except Exception:
        return ""


def start_audit_pipeline(catalog, audit_schema, pipeline_name, agent_name, environment,
                          spark, dbutils, triggered_by="pipeline", depends_on=""):
    """Create and start a ``PipelineStepLogger`` for this notebook's run."""
    from framework.audit.audit_logger import PipelineStepLogger

    pipeline = PipelineStepLogger(
        catalog=catalog, audit_schema=audit_schema,
        pipeline_name=pipeline_name, agent_name=agent_name, environment=environment,
        triggered_by=triggered_by, depends_on=depends_on, spark=spark, dbutils=dbutils,
    )
    pipeline.start()
    return pipeline


def resolve_agent_config_path(bundle_root: str, team_name: str) -> str:
    """Resolve the agent runtime config.yaml to load for guardrails/scorers.

    Team config (``src/teams/<team_name>/config.yaml`` or ``.yml``) wins when
    present, else the shared default at ``src/agent_development/agent/config.yaml``.
    """
    if team_name:
        for ext in ("yaml", "yml"):
            candidate = os.path.join(bundle_root, "src", "teams", team_name, f"config.{ext}")
            if os.path.exists(candidate):
                print(f"Loading config from team config: {candidate}")
                return candidate
    shared_path = os.path.join(bundle_root, "src", "agent_development", "agent", "config.yaml")
    print(f"Loading config from shared config: {shared_path}")
    return shared_path


def resolve_fixture_path(nb_dir: str, bundle_root: str, team_dir: str, filename: str) -> str:
    """Resolve an evaluation fixture file: team fixtures dir first, else shared."""
    if team_dir:
        team_path = os.path.join(bundle_root, "src", "teams", team_dir, "fixtures", filename)
        if os.path.exists(team_path):
            print(f"Using team fixture: {team_path}")
            return team_path
    shared_path = os.path.join(os.path.dirname(nb_dir), "evaluation", filename)
    print(f"Using shared fixture: {shared_path}")
    return shared_path
