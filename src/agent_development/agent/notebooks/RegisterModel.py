# Databricks notebook source
# MAGIC %md
# MAGIC # Register Model
# MAGIC Builds framework wheel, uploads to UC Volume, logs model, registers in UC.
# MAGIC Does NOT deploy — deployment happens after pre-deployment evaluation passes.
# MAGIC
# MAGIC Outputs model_name and model_version for downstream steps.

# COMMAND ----------

# Parameters — pass `team_name` to resolve everything else from src/teams/<team>/config.yaml.
# Other widgets remain as override fallbacks: non-empty overrides the team-config value.
widget_defaults = {
    "catalog": "",
    "schema": "",
    "agent_name": "",
    "llm_endpoint": "",
    "vs_endpoint": "",
    "vs_index": "",
    "embedding_model": "",
    "sql_warehouse_name": "",
    "model_signature_json": "",
    "environment": "dev",
    "team_name": "",
    "team_config": "",  # legacy override
    "skip_register_model": "no",
    "mlflow_tracing_uc_enabled": "no",
}
for name, default in widget_defaults.items():
    dbutils.widgets.text(name, default)

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
agent_name = dbutils.widgets.get("agent_name")
llm_endpoint = dbutils.widgets.get("llm_endpoint")
vs_endpoint = dbutils.widgets.get("vs_endpoint")
vs_index = dbutils.widgets.get("vs_index")
embedding_model = dbutils.widgets.get("embedding_model")
model_signature_json = dbutils.widgets.get("model_signature_json").strip()
environment = dbutils.widgets.get("environment").strip() or "dev"
team_name = dbutils.widgets.get("team_name").strip()
team_config = dbutils.widgets.get("team_config").strip()
skip_register_model = dbutils.widgets.get("skip_register_model").strip().lower() in {"1", "true", "yes", "y"}
w_tracing_uc_enabled = dbutils.widgets.get("mlflow_tracing_uc_enabled").strip().lower() in {"1", "true", "yes", "y"}

if skip_register_model:
    print("Register model skipped as requested (skip_register_model=yes)")
    dbutils.notebook.exit("SKIPPED: skip_register_model=yes")

# COMMAND ----------

# Install dependencies — uses air-gapped volume
import os
import subprocess
import sys

# Serverless compute has no job-cluster wheel library attached yet, so
# `framework` isn't installed — add src/ to sys.path directly (pure stdlib,
# no wheel/install needed) so the shared volume-path constant is the single
# source of truth, same as every other pipeline notebook.
bootstrap_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
if not bootstrap_nb_path.startswith("/Workspace/"):
    bootstrap_nb_path = f"/Workspace{bootstrap_nb_path}"
bootstrap_src_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(bootstrap_nb_path))))
sys.path.insert(0, bootstrap_src_root)
from framework.mlops_utils import get_pip_find_links

vol_path = get_pip_find_links("312")
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "databricks-agents==1.5.0",
    "mlflow>=3.7.0",
    "databricks-langchain[memory]==0.5.1",
    "langchain==0.3.21",
    "langchain-community==0.3.10",
    "langchain-core==0.3.63",
    "langchain-text-splitters==0.3.8",
    "databricks-vectorsearch>=0.40",
    "sqlglot", "tabulate", "numpy", "pandas",
    "--find-links", vol_path, "--no-index", "-q",
])
dbutils.library.restartPython()

# COMMAND ----------
# Re-read after restart
catalog = dbutils.widgets.get("catalog")
w_schema = dbutils.widgets.get("schema")
w_agent_name = dbutils.widgets.get("agent_name")
w_llm_endpoint = dbutils.widgets.get("llm_endpoint")
w_vs_endpoint = dbutils.widgets.get("vs_endpoint")
w_vs_index = dbutils.widgets.get("vs_index")
w_embedding_model = dbutils.widgets.get("embedding_model")
w_sql_warehouse_name = dbutils.widgets.get("sql_warehouse_name").strip()
model_signature_json = dbutils.widgets.get("model_signature_json").strip()
environment = dbutils.widgets.get("environment").strip() or "dev"
team_name = dbutils.widgets.get("team_name").strip()
team_config = dbutils.widgets.get("team_config").strip()
w_tracing_uc_enabled = dbutils.widgets.get("mlflow_tracing_uc_enabled").strip().lower() in {"1", "true", "yes", "y"}

import mlflow
import os, sys, subprocess, glob, shutil

# Resolve notebook-relative paths up-front so the team_config helper can find files
# in the deployed workspace bundle (the helper installed in the wheel can't infer this).
def workspace_abs(path: str) -> str:
    """Normalize workspace paths across interactive and job-cluster contexts."""
    if not path:
        return path
    return path if path.startswith("/Workspace/") else f"/Workspace{path}"

nb_path = workspace_abs(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
nb_dir = os.path.dirname(nb_path)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(nb_dir)))  # .../files/src
bundle_root = os.path.dirname(project_root)  # .../files (where pyproject.toml lives)

# Serverless job environments don't install wildcard whl paths declared in
# environments[].spec.dependencies, so build+install the framework package
# here before any `from framework...` import is attempted. Build from a local
# writable copy — workspace FUSE paths can block setuptools writes.
bootstrap_src = "/tmp/agentops_build_src"
if os.path.exists(bootstrap_src):
    shutil.rmtree(bootstrap_src)
shutil.copytree(bundle_root, bootstrap_src, ignore=shutil.ignore_patterns("build", "*.egg-info"))
bootstrap_wheel_dir = "/tmp/agentops_wheels"
os.makedirs(bootstrap_wheel_dir, exist_ok=True)
subprocess.run(
    ["pip", "wheel", bootstrap_src, "--no-deps", "--no-build-isolation", "-w", bootstrap_wheel_dir],
    check=True,
)
bootstrap_wheel = sorted(glob.glob(os.path.join(bootstrap_wheel_dir, "agentops_framework-*.whl")))[-1]
subprocess.run(["pip", "install", "--force-reinstall", "--no-deps", bootstrap_wheel], check=True)

# Resolve team settings — single source of truth from src/teams/<team_name>/config.yaml.
# When team_name is set, team config values WIN. Widgets only fill gaps the team config
# doesn't specify (empty/missing values). Without team_name, widgets are used as-is.
from framework.team_config import load_team_settings
team_settings = load_team_settings(
    team_name, bundle_root=bundle_root, sql_warehouse_name=w_sql_warehouse_name,
) if team_name else {}

def pick(team_val, widget_val):
    return team_val if team_val else widget_val

catalog = pick(team_settings.get("catalog"), catalog)
schema = pick(team_settings.get("schema"), w_schema)
agent_name = pick(team_settings.get("agent_name"), w_agent_name)
llm_endpoint = pick(team_settings.get("llm_endpoint"), w_llm_endpoint)
vs_endpoint = pick(team_settings.get("vs_endpoint"), w_vs_endpoint)
vs_index = pick(team_settings.get("vs_index"), w_vs_index)
embedding_model = pick(team_settings.get("embedding_model"), w_embedding_model)
audit_schema = pick(team_settings.get("audit_schema"), f"{schema}_audit")
if team_name and not team_config:
    # Set legacy team_config to bundle-relative path so downstream lookup logic works
    team_config = os.path.relpath(team_settings["team_config_path"], bundle_root)
print(f"team_name={team_name!r}  resolved → agent_name={agent_name!r} schema={schema!r} audit_schema={audit_schema!r} vs_index={vs_index!r} llm_endpoint={llm_endpoint!r}")

# Start audit tracking (after pip restart)
from framework.notebook_bootstrap import start_audit_pipeline

pipeline = start_audit_pipeline(
    catalog=catalog, audit_schema=audit_schema,
    pipeline_name="register_model", agent_name=agent_name, environment=environment,
    spark=spark, dbutils=dbutils, triggered_by="pipeline", depends_on="vector_search_setup",
)
step = pipeline.start_step("log_and_register", step_order=1, step_type="registration", depends_on="vector_search_setup")

mlflow.set_registry_uri("databricks-uc")
# Job clusters run under service principals; /Users/<principal> experiment paths
# can be slow or brittle to create. Use a stable shared path instead.
current_user = spark.sql("SELECT current_user()").first()[0]
experiment_name = f"/Users/{current_user}/{agent_name}"

# Bind the experiment to a Unity Catalog trace location automatically for
# whichever team is being registered — traces land in
# {catalog}.{tracing_schema}.{tracing_table_prefix}_otel_* with no per-team
# setup. Falls back to experiment-managed trace storage if disabled or if the
# experiment already exists bound to a different (or no) trace location.
from framework.monitoring.uc_tracing import configure_uc_trace_experiment

tracing_uc_enabled = team_settings.get("tracing_uc_enabled", False) if team_name else w_tracing_uc_enabled
tracing_schema = pick(team_settings.get("tracing_schema"), audit_schema)
tracing_table_prefix = pick(team_settings.get("tracing_table_prefix"), agent_name)

configure_uc_trace_experiment(
    experiment_name=experiment_name,
    catalog=catalog,
    schema=tracing_schema,
    table_prefix=tracing_table_prefix,
    warehouse_id=team_settings.get("tracing_warehouse_id") or None,
    enabled=tracing_uc_enabled,
    spark=spark,
)
print(f"MLflow experiment: {experiment_name}")

# Publish experiment_id for downstream tasks (pre/deploy/post notebooks).
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = (experiment.experiment_id if experiment else "")
    if experiment_id:
        dbutils.jobs.taskValues.set(key="mlflow_experiment_id", value=experiment_id)
        print(f"Published task value mlflow_experiment_id={experiment_id}")
except Exception as exc:
    print(f"Warning: could not publish mlflow_experiment_id task value: {exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Resolve paths

# COMMAND ----------
# nb_path/nb_dir/project_root/bundle_root already computed above for team_config helper
agent_dir = nb_dir
default_config = os.path.join(os.path.dirname(agent_dir), "config.yaml")
tools_dir = os.path.join(os.path.dirname(agent_dir), "tools")
framework_dir = os.path.join(project_root, "framework")

# Use team config if specified, otherwise shared default
if team_config:
    config_path = os.path.join(bundle_root, team_config)
    if not os.path.exists(config_path):
        print(f"WARNING: Team config not found at {config_path}, falling back to default")
        config_path = default_config
    else:
        print(f"Using team config: {config_path}")
else:
    config_path = default_config

# Copy team custom tools into tools/custom_tools/ so they're packaged with the model
# (MLflow can't copy workspace paths as separate code_paths — they must be under tools/)
custom_tools_dest = os.path.join(tools_dir, "custom_tools")
if team_config:
    team_dir_name = os.path.dirname(team_config)  # e.g., src/teams/platform-engineering
    candidate_tools_dir = os.path.join(bundle_root, team_dir_name, "tools")
    if os.path.isdir(candidate_tools_dir):
        tool_files = [f for f in os.listdir(candidate_tools_dir) if f.endswith(".py")]
        if tool_files:
            os.makedirs(custom_tools_dest, exist_ok=True)
            for f in tool_files:
                shutil.copy2(os.path.join(candidate_tools_dir, f), custom_tools_dest)
            print(f"Copied {len(tool_files)} custom tools to {custom_tools_dest}: {tool_files}")

# Resolve the team's custom agent dir — same convention as team fixtures/scorers:
#   src/teams/<team_name>/{fixtures,scorers,agent}. When team_name is provided,
#   model packaging must use the corresponding team Agent.py.
team_agent_dir = os.path.join(bundle_root, "src", "teams", team_name, "agent") if team_name else ""
use_team_agent = bool(team_agent_dir) and os.path.exists(os.path.join(team_agent_dir, "Agent.py"))
cost_wrapper = os.path.join(framework_dir, "inference", "cost_tracking_model.py")

if team_name and not use_team_agent:
    raise FileNotFoundError(
        f"Team agent not found for team_name='{team_name}'. "
        f"Expected Agent.py under {team_agent_dir}."
    )

print(f"Project root (src): {project_root}")
print(f"Bundle root: {bundle_root}")
print(f"Agent dir: {agent_dir}")
print(f"Config: {config_path}")
print(f"Tools dir: {tools_dir}")
print(f"Framework dir: {framework_dir}")
print(f"Team agent dir: {team_agent_dir or '(none)'}  (custom agent: {use_team_agent})")
# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build framework wheel + upload to UC Volume

# COMMAND ----------
wheel_dir = "/tmp/agentops_wheels"
os.makedirs(wheel_dir, exist_ok=True)

# Build from a local writable copy because workspace FUSE paths on classic job
# clusters can block setuptools writes (for build/* and *.egg-info updates).
local_src = "/tmp/agentops_build_src"
if os.path.exists(local_src):
    shutil.rmtree(local_src)
shutil.copytree(
    bundle_root,
    local_src,
    ignore=shutil.ignore_patterns("build", "*.egg-info"),
)

# --no-build-isolation: use cluster's setuptools (avoids PyPI download behind firewalls)
build_result = subprocess.run(
    ["pip", "wheel", local_src, "--no-deps", "--no-build-isolation", "-w", wheel_dir],
    capture_output=True, text=True)
print(build_result.stdout)
if build_result.returncode != 0:
    print(f"Build stderr: {build_result.stderr}")
    raise RuntimeError("Framework wheel build failed")

wheel_files = sorted(glob.glob(os.path.join(wheel_dir, "agentops_framework-*.whl")))
local_wheel = wheel_files[-1] if wheel_files else None
if not local_wheel:
    raise RuntimeError("Framework wheel build succeeded but wheel artifact not found")
wheel_filename = os.path.basename(local_wheel)
print(f"Built: {wheel_filename}")

# Upload to UC Volume with retry and integrity verification
from framework.mlops_utils import (
    create_volume_if_missing, upload_to_volume, compute_sha256, verify_integrity
)
from databricks.sdk import WorkspaceClient

upload_client = WorkspaceClient()
volume_name = "python_dependencies"
create_volume_if_missing(upload_client, catalog, schema, volume_name)
volume_path = f"/Volumes/{catalog}/{schema}/{volume_name}"
volume_wheel = f"{volume_path}/{wheel_filename}"

# Upload with retry
upload_to_volume(upload_client, local_wheel, volume_wheel)

# SHA-256 integrity check
local_hash = compute_sha256(local_wheel)
print(f"Wheel SHA-256: {local_hash[:16]}…")

# Install on cluster for log_model validation
subprocess.run(["pip", "install", "--force-reinstall", "--no-deps", local_wheel],
               capture_output=True, text=True)
sys.path.insert(0, os.path.dirname(tools_dir))
print("Installed on cluster for validation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Merge config (databricks.yml + config.yaml) and log model

# COMMAND ----------

from mlflow.models.resources import (
    DatabricksServingEndpoint, DatabricksVectorSearchIndex,
    DatabricksSQLWarehouse, DatabricksTable,
)
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.utils.environment import _mlflow_conda_env as mlflow_conda_env
import json
import yaml
from framework.team_config import resolve_sql_warehouse_ids

# Merge config: databricks.yml (infra) + config.yaml (agent settings)
with open(config_path) as f:
    agent_config = yaml.safe_load(f)

agent_config = resolve_sql_warehouse_ids(agent_config, default_warehouse_name=w_sql_warehouse_name)
agent_config["catalog"] = catalog
agent_config["schema"] = schema
agent_config["agent"]["name"] = agent_name
if llm_endpoint:
    agent_config["llm"]["endpoint"] = llm_endpoint

vs_cfg = agent_config.setdefault("vector_search", {})
vs_primary = vs_cfg.setdefault("primary", {})
vs_indexes = vs_cfg.setdefault("indexes", [])

# Keep modern multi-index config in sync with pipeline/team overrides.
if vs_endpoint:
    vs_primary["endpoint"] = vs_endpoint
if vs_index:
    vs_primary["index"] = vs_index
if vs_primary.get("index"):
    vs_primary.setdefault("name", "context")
    # Backfill legacy single-index fields consumed by the current agent runtime.
    vs_cfg["index"] = vs_primary.get("index")
if vs_primary.get("endpoint"):
    vs_cfg["endpoint"] = vs_primary.get("endpoint")

if vs_primary.get("index"):
    primary_name = vs_primary.get("name", "context")
    updated = False
    for entry in vs_indexes:
        if isinstance(entry, dict) and entry.get("name") == primary_name:
            if vs_primary.get("endpoint"):
                entry["endpoint"] = vs_primary.get("endpoint")
            entry["index"] = vs_primary.get("index")
            updated = True
            break
    if not updated:
        vs_indexes.insert(0, {
            "name": primary_name,
            "endpoint": vs_primary.get("endpoint", ""),
            "index": vs_primary.get("index"),
        })

runtime_config = "/tmp/runtime_config.yaml"
with open(runtime_config, "w") as f:
    yaml.dump(agent_config, f, default_flow_style=False)

print("Merged runtime config:")
print(yaml.dump(agent_config, default_flow_style=False)[:500])

# Resource declarations — these grant Model Serving system auth access
resolved_llm_endpoint = agent_config["llm"]["endpoint"]

def fq_index_name(index_name: str) -> str:
    if not index_name:
        return ""
    return f"{catalog}.{schema}.{index_name}" if "." not in index_name else index_name

def collect_vs_indexes(cfg: dict) -> list:
    """Collect vector-search index names from modern team config shape."""
    out = []

    vs_cfg = cfg.get("vector_search", {}) or {}
    # Modern primary index shape
    vs_primary = vs_cfg.get("primary") or {}
    if isinstance(vs_primary, dict) and vs_primary.get("index"):
        out.append(vs_primary.get("index"))

    # Modern list shape: vector_search.indexes: [{index: "..."}, ...]
    for item in vs_cfg.get("indexes", []) or []:
        if isinstance(item, dict):
            idx = item.get("index")
            if idx:
                out.append(idx)

    # Preserve order, remove blanks/duplicates
    seen = set()
    deduped = []
    for idx in out:
        fq = fq_index_name(str(idx).strip())
        if fq and fq not in seen:
            seen.add(fq)
            deduped.append(fq)
    return deduped

vs_indexes_fq = collect_vs_indexes(agent_config)
if not vs_indexes_fq:
    raise ValueError(
        "No vector search index resolved from runtime config. "
        "Provide vector_search.primary.index or vector_search.indexes[*].index."
    )

resources = [
    DatabricksServingEndpoint(endpoint_name=resolved_llm_endpoint),
]
resources.extend(DatabricksVectorSearchIndex(index_name=i) for i in vs_indexes_fq)
print(f"Resource: VS indexes {', '.join(vs_indexes_fq)}")

# Declare SQL warehouse so Model Serving can write to UC Delta via SQL Statement API
session_cfg = agent_config.get("session_history", {})
uc_cfg = session_cfg.get("unity_catalog", {})
warehouse_ids = {
    str(block[key])
    for block in (agent_config, agent_config.get("parameters") or {})
    for key in ("sql_warehouse_id", "mlflow_tracing_warehouse_id")
    if block.get(key) and block[key] != "auto"
}
warehouse_id = uc_cfg.get("warehouse_id", "")
if uc_cfg.get("enabled") and warehouse_id and warehouse_id != "auto":
    warehouse_ids.add(str(warehouse_id))
tracing_cfg = agent_config.get("tracing") or {}
if tracing_cfg.get("uc_trace_storage_enabled") and tracing_cfg.get("sql_warehouse_id") not in (None, "", "auto"):
    warehouse_ids.add(str(tracing_cfg["sql_warehouse_id"]))
for warehouse_id in sorted(warehouse_ids):
    resources.append(DatabricksSQLWarehouse(warehouse_id=warehouse_id))
    print(f"Resource: SQL Warehouse {warehouse_id}")

# Declare UC session history table so Model Serving can read/write it
uc_table = uc_cfg.get("table", "")
if uc_cfg.get("enabled") and uc_table:
    uc_table_fq = f"{catalog}.{schema}.{uc_table}"
    resources.append(DatabricksTable(table_name=uc_table_fq))
    print(f"Resource: UC Table {uc_table_fq}")

print(f"Total resources declared: {len(resources)}")

input_example = {"input": [{"role": "user", "content": "What is Delta Lake?"}]}
output_example = {"output": [{
    "type": "message", "id": "example", "role": "assistant",
    "content": [{"type": "output_text", "text": "Delta Lake is an open-source storage layer..."}],
}]}
if model_signature_json:
    try:
        signature = ModelSignature.from_dict(json.loads(model_signature_json))
        print("Using user-provided model signature from widget: model_signature_json")
    except Exception as e:
        raise ValueError(
            "Invalid model_signature_json. Provide valid JSON for ModelSignature.from_dict(...), "
            "or leave it empty to use inferred default signature."
        ) from e
else:
    signature = infer_signature(input_example, output_example)
    print("Using inferred default model signature")
model_name = f"{catalog}.{schema}.{agent_name}"

# Embed wheel in model artifact so serving image builds can resolve it.
wheel_artifact_name = f"artifacts/{wheel_filename}"
model_artifacts = {wheel_artifact_name: volume_wheel}

# Resolve model runtime dependencies.
# Team config can override/extend defaults via:
#
# model:
#   pip_dependency_mode: extend   # extend | replace
#   pip_dependencies:
#     - langchain
#     - langchain-community
#
# Backward-compatible alias is also supported at top-level:
# pip_dependencies: [...]
default_model_pip_deps = [
    "mlflow>=3.7.0",
    "openai>=1.66.0",
    "databricks-agents==1.5.0",
    "databricks-sdk>=0.30.0",
    "databricks-langchain[memory]==0.5.1",
    "langchain==0.3.21",
    "databricks-vectorsearch>=0.40",
    "langchain-community==0.3.10",
    "langchain-core==0.3.63",
    "langchain-text-splitters==0.3.8",
    "sqlglot",
    "tabulate",
    "numpy",
    "pandas",
    "pyyaml>=6.0",
]

model_cfg = agent_config.get("model") or {}
team_pip_deps = model_cfg.get("pip_dependencies")
if team_pip_deps is None:
    team_pip_deps = agent_config.get("pip_dependencies", [])

if team_pip_deps is None:
    team_pip_deps = []
if not isinstance(team_pip_deps, list):
    raise ValueError("model.pip_dependencies must be a list of package spec strings")

dep_mode = str(model_cfg.get("pip_dependency_mode", "extend")).strip().lower()
if dep_mode not in {"extend", "replace"}:
    raise ValueError("model.pip_dependency_mode must be either 'extend' or 'replace'")

if dep_mode == "replace":
    resolved_model_pip_deps = [str(x).strip() for x in team_pip_deps if str(x).strip()]
else:
    resolved_model_pip_deps = []
    seen_deps = set()
    for dep in default_model_pip_deps + [str(x).strip() for x in team_pip_deps if str(x).strip()]:
        if dep not in seen_deps:
            seen_deps.add(dep)
            resolved_model_pip_deps.append(dep)

print(f"Model pip dependency mode: {dep_mode}")
print(f"Model pip deps count: {len(resolved_model_pip_deps)}")

conda_env = mlflow_conda_env(
    additional_pip_deps=[
        f"./{wheel_artifact_name}",
    ] + resolved_model_pip_deps,
)

# Build a helper to expand a team agent directory into code_paths entries.
# Two things are needed:
#   1. The directory itself (staged as `agent/`) → enables `import agent.Agent`
#   2. Each sub-package inside it at the top level → enables `import utils`,
#      `import sql_generator`, etc. that Agent.py uses via absolute imports.
def team_agent_code_paths(agent_dir: str) -> list:
    paths = [agent_dir]  # staged as `agent/`
    for entry in sorted(os.listdir(agent_dir)):
        if entry.startswith(".") or entry == "__pycache__":
            continue
        full = os.path.join(agent_dir, entry)
        # Add sub-packages (directories) so Agent.py can `import utils` etc.
        if os.path.isdir(full):
            paths.append(full)
    return paths

# Select the model entry point + code_paths.
#   - Team custom agent present → register the team's Agent.py wrapped by the
#     cost-tracking model (real-time per-request $).
#   - Otherwise → unchanged shared framework agent.
if use_team_agent and os.path.exists(cost_wrapper):
    # cost_tracking_model.py is the framework entry point (python_model).
    # Agent.py is staged inside code_paths so its __file__ lands at
    # <temp>/code/agent/Agent.py — keeping ../configs/ resolution correct.
    python_model = cost_wrapper
    code_paths = [tools_dir] + team_agent_code_paths(team_agent_dir)
    # include team configs/ (prompt_templates.yml etc.) so they stage alongside Agent.py
    team_configs_dir = os.path.join(os.path.dirname(team_agent_dir), "configs")
    if os.path.isdir(team_configs_dir):
        code_paths.append(team_configs_dir)
    if team_config and os.path.isfile(config_path):
        code_paths.append(config_path)
    print(f"Registering TEAM custom agent (cost-tracked): {team_agent_dir}")
elif use_team_agent:
    python_model = os.path.join(team_agent_dir, "Agent.py")
    code_paths = [tools_dir] + team_agent_code_paths(team_agent_dir)
    # include team configs/ (prompt_templates.yml etc.) if present
    team_configs_dir = os.path.join(os.path.dirname(team_agent_dir), "configs")
    if os.path.isdir(team_configs_dir):
        code_paths.append(team_configs_dir)
    if team_config and os.path.isfile(config_path):
        code_paths.append(config_path)
    print(f"Registering TEAM custom agent (no cost wrapper found): {team_agent_dir}")
else:
    python_model = os.path.join(agent_dir, "Agent.py")
    code_paths = [tools_dir]
    print(f"Registering shared framework agent: {python_model}")

print(f"  python_model: {python_model}")
print(f"  code_paths: {code_paths}")

# MLflow's inline validation executes the python_model file before staging, so
# code_paths are not yet on sys.path. Pre-add them here so Agent.py's imports work.
for p in code_paths:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

with mlflow.start_run(run_name="agent_registration"):
    log_model_kwargs = {
        "artifact_path": "agent",
        "python_model": python_model,
        "model_config": runtime_config,
        "code_paths": code_paths,
        "resources": resources,
        "conda_env": conda_env,
        "artifacts": model_artifacts,
    }
    try:
        model_src = open(python_model, encoding="utf-8").read()
        if not any(agent_type in model_src for agent_type in ("ChatAgent", "ResponsesAgent", "AgentOPSBase")):
            log_model_kwargs["signature"] = signature
        else:
            print("Detected agent model — omitting explicit signature (MLflow sets it automatically).")
    except Exception:
        log_model_kwargs["signature"] = signature
    model_info = mlflow.pyfunc.log_model(**log_model_kwargs)

print(f"Logged: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Register to UC + set aliases

# COMMAND ----------

# env_pack packages the notebook's already-installed (offline, volume-sourced)
# environment into the model artifact, so Model Serving restores it directly
# instead of resolving conda.yaml over the network at container-build time.
# install_dependencies=False skips env_pack's own validation reinstall, which
# would otherwise require network/private-mirror access this cluster lacks.
try:
    from mlflow.utils.env_pack import EnvPackConfig
    env_pack = EnvPackConfig(name="databricks_model_serving", install_dependencies=False)
except ImportError:
    # Older MLflow (e.g. 3.3.2) only supports the string shorthand.
    env_pack = "databricks_model_serving"

uc_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=model_name,
    env_pack=env_pack,
)
print(f"Registered: {model_name} version {uc_model.version}")

from mlflow import MlflowClient
client = MlflowClient()
client.set_registered_model_alias(model_name, "champion", uc_model.version)
print(f"Alias set: {model_name}@champion → v{uc_model.version}")

try:
    old_champion = client.get_model_version_by_alias(model_name, "previous")
except Exception:
    try:
        old = client.get_model_version_by_alias(model_name, "champion")
        if old.version != uc_model.version:
            client.set_registered_model_alias(model_name, "previous", old.version)
            print(f"Alias set: {model_name}@previous → v{old.version}")
    except Exception:
        pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4b. Register system prompt in MLflow Prompt Registry

# COMMAND ----------

# Register the system prompt so it can be versioned, tracked, and loaded at runtime
prompt_name = f"{catalog}.{schema}.{agent_name}_system_prompt"
system_prompt = agent_config.get("system_prompt", "")

try:
    prompt_version = mlflow.genai.register_prompt(
        name=prompt_name,
        template=system_prompt,
        commit_message=f"Registered with model v{uc_model.version}",
        tags={
            "model_version": str(uc_model.version),
            "llm_endpoint": agent_config["llm"]["endpoint"],
        },
    )
    print(f"Prompt registered: {prompt_name} v{prompt_version.version}")

    # Set production alias
    mlflow.genai.set_prompt_alias(
        name=prompt_name,
        alias="production",
        version=prompt_version.version,
    )
    print(f"Prompt alias set: {prompt_name}@production → v{prompt_version.version}")
except Exception as e:
    # Prompt registry is required — fail the task rather than silently skipping.
    pipeline.end_step(step, status="FAILED", output_summary={"error": str(e)[:500]})
    pipeline.end(status="FAILED")
    raise RuntimeError(f"Prompt registration failed for '{prompt_name}': {e}") from e

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Complete audit tracking

# COMMAND ----------

pipeline.end_step(step, status="COMPLETED", output_summary={
    "model_name": model_name,
    "model_version": str(uc_model.version),
    "model_uri": model_info.model_uri,
    "champion_alias": f"@champion → v{uc_model.version}",
})
pipeline.end(status="COMPLETED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Output for downstream steps

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({
    "model_name": model_name,
    "model_version": str(uc_model.version),
    "model_uri": model_info.model_uri,
}))
