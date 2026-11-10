# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Agent
# MAGIC Deploys the registered model to a serving endpoint.
# MAGIC Runs AFTER pre-deployment evaluation passes.
# MAGIC Supports champion/challenger traffic splits.

# COMMAND ----------

# Minimal locator: make the shared startup helper importable before installation.
import sys
from pathlib import Path
_notebook = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
sys.path.insert(0, str(Path(_notebook if _notebook.startswith('/Workspace/') else '/Workspace' + _notebook).parents[3]))
from framework.notebook_bootstrap import initialize_notebook
notebook_runtime = initialize_notebook(dbutils)


dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("agent_name", "")
# Deployment / champion-challenger settings
dbutils.widgets.text("champion_model_version", "latest")
dbutils.widgets.text("champion_workload_size", "Small")
dbutils.widgets.text("champion_traffic_percentage", "100")
dbutils.widgets.text("challenger_enabled", "false")
dbutils.widgets.text("challenger_model_version", "")
dbutils.widgets.text("challenger_workload_size", "Small")
dbutils.widgets.text("challenger_traffic_percentage", "0")
dbutils.widgets.text("serving_scale_to_zero", "true")
dbutils.widgets.text("agent_endpoint", "agentops-docs-chatbot")
# AI Gateway settings
dbutils.widgets.text("ai_gateway_safety_enabled", "true")
dbutils.widgets.text("inference_tables_enabled", "true")
dbutils.widgets.text("rate_limit_per_user_per_minute", "10")
dbutils.widgets.text("rate_limit_per_endpoint_per_minute", "100")
dbutils.widgets.text("usage_tracking_enabled", "true")
dbutils.widgets.text("environment", "dev")
dbutils.widgets.text("team_name", "")
dbutils.widgets.text("mlflow_experiment_id", "")
dbutils.widgets.text("skip_deploy_agent", "no")

catalog = dbutils.widgets.get("catalog")
w_schema = dbutils.widgets.get("schema")
w_agent_name = dbutils.widgets.get("agent_name")
champion_version_param = dbutils.widgets.get("champion_model_version")
champion_workload = dbutils.widgets.get("champion_workload_size")
champion_traffic = int(dbutils.widgets.get("champion_traffic_percentage"))
challenger_enabled = dbutils.widgets.get("challenger_enabled").lower() == "true"
challenger_version = dbutils.widgets.get("challenger_model_version")
challenger_workload = dbutils.widgets.get("challenger_workload_size")
challenger_traffic = int(dbutils.widgets.get("challenger_traffic_percentage"))
scale_to_zero = dbutils.widgets.get("serving_scale_to_zero").lower() == "true"
w_agent_endpoint = dbutils.widgets.get("agent_endpoint")
ai_gateway_safety = dbutils.widgets.get("ai_gateway_safety_enabled").lower() == "true"
inference_tables = dbutils.widgets.get("inference_tables_enabled").lower() == "true"
rate_limit_user = int(dbutils.widgets.get("rate_limit_per_user_per_minute"))
rate_limit_endpoint = int(dbutils.widgets.get("rate_limit_per_endpoint_per_minute"))
usage_tracking = dbutils.widgets.get("usage_tracking_enabled").lower() == "true"
environment = dbutils.widgets.get("environment").strip() or "dev"
team_name = dbutils.widgets.get("team_name").strip()
mlflow_experiment_id = dbutils.widgets.get("mlflow_experiment_id").strip()
skip_deploy_agent = dbutils.widgets.get("skip_deploy_agent").strip().lower() in {"1", "true", "yes", "y"}

if skip_deploy_agent:
    print("Deploy agent skipped as requested (skip_deploy_agent=yes)")
    dbutils.notebook.exit("SKIPPED: skip_deploy_agent=yes")

# Pre-restart placeholders; resolve via team_config helper after restart
schema = w_schema
agent_name = w_agent_name
agent_endpoint = w_agent_endpoint
model_name = f"{catalog}.{schema}.{agent_name}"

# COMMAND ----------

# Install dependencies — uses air-gapped volume
import os

from framework.notebook_bootstrap import install_air_gapped_packages

install_air_gapped_packages(["databricks-agents", "mlflow>=3.7.0"])
dbutils.library.restartPython()

# COMMAND ----------

# Re-read after restart
# Minimal locator: make the shared startup helper importable before installation.
import sys
from pathlib import Path
_notebook = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
sys.path.insert(0, str(Path(_notebook if _notebook.startswith('/Workspace/') else '/Workspace' + _notebook).parents[3]))
from framework.notebook_bootstrap import initialize_notebook
notebook_runtime = initialize_notebook(dbutils)

import os

from framework.notebook_bootstrap import (
    resolve_bundle_paths, resolve_mlflow_experiment_id, pick_if_not_none, start_audit_pipeline,
)
from framework.team_config import load_team_settings

catalog = dbutils.widgets.get("catalog")
w_schema = dbutils.widgets.get("schema")
w_agent_name = dbutils.widgets.get("agent_name")
champion_version_param = dbutils.widgets.get("champion_model_version")
champion_workload = dbutils.widgets.get("champion_workload_size")
champion_traffic = int(dbutils.widgets.get("champion_traffic_percentage"))
challenger_enabled = dbutils.widgets.get("challenger_enabled").lower() == "true"
challenger_version = dbutils.widgets.get("challenger_model_version")
challenger_workload = dbutils.widgets.get("challenger_workload_size")
challenger_traffic = int(dbutils.widgets.get("challenger_traffic_percentage"))
scale_to_zero = dbutils.widgets.get("serving_scale_to_zero").lower() == "true"
w_agent_endpoint = dbutils.widgets.get("agent_endpoint")
ai_gateway_safety = dbutils.widgets.get("ai_gateway_safety_enabled").lower() == "true"
inference_tables = dbutils.widgets.get("inference_tables_enabled").lower() == "true"
rate_limit_user = int(dbutils.widgets.get("rate_limit_per_user_per_minute"))
rate_limit_endpoint = int(dbutils.widgets.get("rate_limit_per_endpoint_per_minute"))
usage_tracking = dbutils.widgets.get("usage_tracking_enabled").lower() == "true"
environment = dbutils.widgets.get("environment").strip() or "dev"
team_name = dbutils.widgets.get("team_name").strip()
mlflow_experiment_id = resolve_mlflow_experiment_id(
    dbutils, dbutils.widgets.get("mlflow_experiment_id"), upstream_task_key="register_model",
)

# Resolve team settings — team config wins when team_name is set.
bundle_paths = resolve_bundle_paths(dbutils)
bundle_root = bundle_paths["bundle_root"]

team_settings = load_team_settings(team_name, bundle_root=bundle_root) if team_name else {}


def pick(team_val, widget_val):
    return team_val if team_val else widget_val


catalog = pick(team_settings.get("catalog"), catalog)
schema = pick(team_settings.get("schema"), w_schema)
agent_name = pick(team_settings.get("agent_name"), w_agent_name)
agent_endpoint = pick(team_settings.get("agent_endpoint"), w_agent_endpoint)

# Compute size / traffic split / AI Gateway — booleans and percentages can be
# explicitly 0/False in team config, so these use pick_if_not_none (not the
# truthy pick() above) to avoid mistaking a real override for "unset".
champion_version_param = pick(team_settings.get("champion_model_version"), champion_version_param)
champion_workload = pick(team_settings.get("champion_workload_size"), champion_workload)
champion_traffic = pick_if_not_none(team_settings.get("champion_traffic_percentage"), champion_traffic)
challenger_enabled = pick_if_not_none(team_settings.get("challenger_enabled"), challenger_enabled)
challenger_version = pick(team_settings.get("challenger_model_version"), challenger_version)
challenger_workload = pick(team_settings.get("challenger_workload_size"), challenger_workload)
challenger_traffic = pick_if_not_none(team_settings.get("challenger_traffic_percentage"), challenger_traffic)
scale_to_zero = pick_if_not_none(team_settings.get("serving_scale_to_zero"), scale_to_zero)
ai_gateway_safety = pick_if_not_none(team_settings.get("ai_gateway_safety_enabled"), ai_gateway_safety)
inference_tables = pick_if_not_none(team_settings.get("ai_gateway_inference_tables_enabled"), inference_tables)
rate_limit_user = pick_if_not_none(team_settings.get("ai_gateway_rate_limit_per_user_per_minute"), rate_limit_user)
rate_limit_endpoint = pick_if_not_none(team_settings.get("ai_gateway_rate_limit_per_endpoint_per_minute"), rate_limit_endpoint)
usage_tracking = pick_if_not_none(team_settings.get("ai_gateway_usage_tracking_enabled"), usage_tracking)

print(f"team_name={team_name!r}  resolved → catalog={catalog!r}, agent_name={agent_name!r} schema={schema!r} agent_endpoint={agent_endpoint!r}")
print(f"  champion: workload={champion_workload!r} traffic={champion_traffic}% scale_to_zero={scale_to_zero}")
if challenger_enabled:
    print(f"  challenger: workload={challenger_workload!r} traffic={challenger_traffic}%")
model_name = f"{catalog}.{schema}.{agent_name}"

# Start audit tracking
pipeline = start_audit_pipeline(
    catalog=catalog, audit_schema=f"{schema}_audit",
    pipeline_name="deploy_agent", agent_name=agent_name, environment=environment,
    spark=spark, dbutils=dbutils, triggered_by="pipeline", depends_on="pre_deployment_eval",
)
step = pipeline.start_step("deploy_endpoint", step_order=1, step_type="deployment", depends_on="pre_deployment_eval")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Resolve champion version

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks import agents
from mlflow import MlflowClient

w = WorkspaceClient()
client = MlflowClient()

# "latest" means the most recent registered version (set by RegisterModel)
if champion_version_param in ("latest", "0", ""):
    from framework.team_runtime import candidate_model_uri
    candidate = candidate_model_uri(dbutils, model_name)
    if candidate.endswith('@champion'):
        champion_version_resolved = client.get_model_version_by_alias(model_name, "champion").version
    else:
        champion_version_resolved = candidate.rsplit('/', 1)[-1]
    print(f"Resolved champion from @champion alias: v{champion_version_resolved}")
else:
    champion_version_resolved = champion_version_param

print(f"=== Deployment Config ===")
print(f"  Model: {model_name}")
print(f"  Champion: v{champion_version_resolved} → {champion_traffic}% traffic ({champion_workload})")
if challenger_enabled:
    print(f"  Challenger: v{challenger_version} → {challenger_traffic}% traffic ({challenger_workload})")
    assert champion_traffic + challenger_traffic == 100, \
        f"Traffic must sum to 100%, got {champion_traffic + challenger_traffic}%"
    client.set_registered_model_alias(model_name, "challenger", challenger_version)
    print(f"  Alias set: {model_name}@challenger → v{challenger_version}")
else:
    print(f"  Challenger: disabled")
    try:
        client.delete_registered_model_alias(model_name, "challenger")
    except Exception:
        pass
print(f"  Scale to zero: {scale_to_zero}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Deploy champion

# COMMAND ----------

try:
    previous_version = client.get_model_version_by_alias(model_name, "champion").version
except Exception:
    previous_version = None

def deploy_champion(with_tags=True):
    kwargs = dict(
        endpoint_name=agent_endpoint,
        workload_size=champion_workload,
        scale_to_zero_enabled=scale_to_zero,
    )
    from framework.team_runtime import serving_environment
    kwargs["environment_vars"] = serving_environment(bundle_root, team_name, w, agent_endpoint, experiment_id=mlflow_experiment_id)
    if with_tags:
        kwargs["tags"] = {"environment": environment, "framework": "agentops"}
    return agents.deploy(model_name, champion_version_resolved, **kwargs)


def wait_for_endpoint_update(endpoint, removed_deployment=None, max_wait_seconds=600, poll_interval=10):
    deadline = time.monotonic() + max_wait_seconds
    while time.monotonic() < deadline:
        try:
            current = w.serving_endpoints.get(endpoint)
        except NotFound:
            return None
        state = getattr(current, "state", None)
        update = getattr(state, "config_update", None)
        update = getattr(update, "value", update)
        if update == "UPDATE_FAILED":
            raise RuntimeError(f"Endpoint '{endpoint}' configuration update failed; inspect its deployment logs before retrying.")
        configurations = (getattr(current, "config", None), getattr(current, "pending_config", None))
        served = [
            entity for config in configurations if config is not None
            for entity in (getattr(config, "served_models", None) or getattr(config, "served_entities", None) or [])
        ]
        removal_pending = removed_deployment is not None and any(
            (
                getattr(entity, "entity_name", None) or getattr(entity, "model_name", None),
                str(getattr(entity, "entity_version", None) or getattr(entity, "model_version", None)),
            ) == removed_deployment
            for entity in served
        )
        if update == "NOT_UPDATING" and not getattr(current, "pending_config", None) and not removal_pending:
            return current
        print(f"  Waiting for {endpoint}: config_update={update}, removal_pending={removal_pending}")
        time.sleep(poll_interval)
    raise RuntimeError(
        f"Endpoint '{endpoint}' cleanup did not finish within {max_wait_seconds}s "
        f"(deployment awaiting removal: {removed_deployment}). Deployment retry aborted."
    )


import mlflow
if mlflow_experiment_id:
    mlflow.set_experiment(experiment_id=mlflow_experiment_id)
    print(f"MLflow experiment set from pipeline experiment_id: {mlflow_experiment_id}")
else:
    print("Using default notebook MLflow experiment (no custom set_experiment)")

try:
    deployment = deploy_champion()
except Exception as deploy_exc:
    # Tags already exist on endpoint from prior deploy — retry without tags.
    if "Duplicate key" in str(deploy_exc):
        print(f"Tags already present on endpoint, retrying without tags...")
        deployment = deploy_champion(with_tags=False)
    # The endpoint has incompatible model versions (different signature or pre-MLflow-2.20).
    # Remove all served versions of this model from the endpoint, then retry.
    elif "same signature" in str(deploy_exc) or "incompatible" in str(deploy_exc).lower() or "InvalidParameterValue" in type(deploy_exc).__name__:
        print(f"Incompatible model versions on '{agent_endpoint}'. Removing stale versions...")
        ep = wait_for_endpoint_update(agent_endpoint)
        cfg = getattr(ep, "config", None)
        served = getattr(cfg, "served_models", None) or getattr(cfg, "served_entities", None) or []
        for sm in served:
            name = getattr(sm, "entity_name", None) or getattr(sm, "model_name", None)
            ver = getattr(sm, "entity_version", None) or getattr(sm, "model_version", None)
            if not (name and ver):
                raise RuntimeError(f"Cannot identify a served deployment on '{agent_endpoint}'; cleanup aborted.")
            print(f"  Deleting {name} v{ver} from {agent_endpoint}")
            try:
                agents.delete_deployment(name, str(ver))
            except Exception as delete_exc:
                raise RuntimeError(
                    f"Failed to remove {name} v{ver} from '{agent_endpoint}'. Deployment retry aborted."
                ) from delete_exc
            wait_for_endpoint_update(agent_endpoint, removed_deployment=(name, str(ver)))
        print("Retrying deploy...")
        deployment = deploy_champion()
    else:
        raise

endpoint_name = deployment.endpoint_name
print(f"Champion deployed: {endpoint_name} (v{champion_version_resolved})")
# agents.deploy() auto-enables inference tables under its own table_name_prefix —
# the real name is confirmed by querying the endpoint in the AI Gateway step below.
payload_table_name = f"{catalog}.{schema}.`{endpoint_name}_payload`"


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configure challenger (if enabled)

# COMMAND ----------
if challenger_enabled and challenger_version:
    from databricks.sdk.service.serving import ServedEntityInput, TrafficConfig, Route

    print(f"Waiting for champion to be READY before adding challenger...")
    champion_ready = False
    for i in range(120):
        ep = w.serving_endpoints.get(endpoint_name)
        state = ep.state
        if state and str(state.ready).endswith("READY") and \
           (state.config_update is None or str(state.config_update).endswith("NOT_UPDATING")):
            champion_ready = True
            break
        time.sleep(30)

    if not champion_ready:
        raise RuntimeError(
            f"Endpoint {endpoint_name} not READY after waiting for champion deployment. "
            f"Aborting challenger config so pipeline fails fast."
        )

    w.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=[
            ServedEntityInput(entity_name=model_name, entity_version=champion_version_resolved,
                              workload_size=champion_workload, scale_to_zero_enabled=scale_to_zero),
            ServedEntityInput(entity_name=model_name, entity_version=challenger_version,
                              workload_size=challenger_workload, scale_to_zero_enabled=scale_to_zero),
        ],
        traffic_config=TrafficConfig(routes=[
            Route(served_model_name=f"{agent_name}-{champion_version_resolved}", traffic_percentage=champion_traffic),
            Route(served_model_name=f"{agent_name}-{challenger_version}", traffic_percentage=challenger_traffic),
        ]),
    )
    print(f"A/B config applied: {champion_traffic}% → v{champion_version_resolved}, {challenger_traffic}% → v{challenger_version}")
# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Wait for endpoint ready

# COMMAND ----------

max_wait_seconds = 4000
poll_interval = 30
elapsed = 0
endpoint_ready = False

print(f"Waiting for {endpoint_name} (max {max_wait_seconds}s)...")
while elapsed < max_wait_seconds:
    try:
        ep = w.serving_endpoints.get(endpoint_name)
        state = ep.state
        if state and str(state.ready).endswith("READY") and \
           (state.config_update is None or str(state.config_update).endswith("NOT_UPDATING")):
            endpoint_ready = True
            print(f"READY after {elapsed}s")
            break
        print(f"  {elapsed}s: ready={state.ready if state else '?'}, config={state.config_update if state else '?'}")
    except Exception as e:
        print(f"  {elapsed}s: check failed ({e})")
    time.sleep(poll_interval)
    elapsed += poll_interval

if not endpoint_ready:
    raise RuntimeError(
        f"Endpoint {endpoint_name} not ready after {max_wait_seconds}s. "
        f"Failing deploy step to prevent downstream pipeline execution."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4b. Remove stale served versions
# MAGIC Only runs once the new deployment is confirmed READY. Keeps the versions
# MAGIC currently carrying traffic (champion + challenger) and tears down any
# MAGIC older served versions left idle (0% traffic) from prior deploys.

# COMMAND ----------
if previous_version and str(previous_version) != str(champion_version_resolved):
    client.set_registered_model_alias(model_name, "previous", previous_version)
client.set_registered_model_alias(model_name, "champion", champion_version_resolved)
keep_versions = {str(champion_version_resolved)}
if challenger_enabled and challenger_version:
    keep_versions.add(str(challenger_version))

removed_versions = []
try:
    ep = w.serving_endpoints.get(endpoint_name)
    cfg = getattr(ep, "config", None) or getattr(ep, "pending_config", None)
    served = getattr(cfg, "served_models", None) or getattr(cfg, "served_entities", None) or []
    for sm in served:
        name = (getattr(sm, "entity_name", None)
                or getattr(sm, "model_name", None))
        ver = (getattr(sm, "entity_version", None)
               or getattr(sm, "model_version", None))
        if not (name and ver) or name != model_name or str(ver) in keep_versions:
            continue
        print(f"  Removing stale served version {name} v{ver} from {endpoint_name}")
        try:
            agents.delete_deployment(name, str(ver))
            removed_versions.append(str(ver))
        except Exception as delete_exc:
            print(f"    Warning (delete {name} v{ver}): {delete_exc}")
except Exception as inspect_exc:
    print(f"  Stale-version cleanup skipped: {inspect_exc}")

if removed_versions:
    print(f"Removed {len(removed_versions)} stale version(s): {', '.join(removed_versions)}")
else:
    print("No stale served versions to remove.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Configure AI Gateway
# MAGIC Sets inference tables, rate limits, safety filter, and usage tracking.

# COMMAND ----------
if endpoint_ready:
    import requests

    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    host = spark.conf.get("spark.databricks.workspaceUrl", "")
    if not host.startswith("http"):
        host = f"https://{host}"

    # Apply AI Gateway config incrementally — some features may not be supported
    # on all workspace/endpoint types (e.g., agent endpoints don't support
    # rate limits, usage tracking, or safety guardrails on FEVM workspaces).

    # agents.deploy() already turns on inference tables (under its own
    # table_name_prefix) when the champion is deployed — read that back
    # instead of blindly resending a no-prefix config, which the API rejects
    # once a table under the default prefix already exists from an earlier run.
    existing_gateway = requests.get(f"{host}/api/2.0/serving-endpoints/{endpoint_name}",
        headers={"Authorization": f"Bearer {token}"}).json().get("ai_gateway", {}) or {}
    existing_itc = existing_gateway.get("inference_table_config", {}) or {}
    if existing_itc.get("enabled"):
        payload_table_name = f"{catalog}.{schema}.`{existing_itc.get('table_name_prefix', endpoint_name)}_payload`"

    # Step 1: Inference tables (most widely supported) — only if agents.deploy()
    # hasn't already enabled them.
    if inference_tables and not existing_itc.get("enabled"):
        gw = {"inference_table_config": {
            "catalog_name": catalog, "schema_name": schema, "enabled": True,
        }}
        response = requests.put(f"{host}/api/2.0/serving-endpoints/{endpoint_name}/ai-gateway",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json=gw)
        if response.ok:
            print(f"  Inference tables: enabled → {payload_table_name}")
        else:
            print(f"  Inference tables: FAILED ({response.status_code}) — {response.json().get('message', '')[:100]}")
    elif inference_tables:
        print(f"  Inference tables: already enabled by deployment → {payload_table_name}")

    # Step 2: Try adding rate limits + usage tracking + safety (may fail on some endpoint types)
    gw_full = {}
    if inference_tables:
        # Reuse the already-applied config (preserves agents.deploy()'s
        # table_name_prefix) instead of resending a no-prefix config that the
        # API rejects once the default-prefix table already exists.
        gw_full["inference_table_config"] = existing_itc if existing_itc.get("enabled") else {
            "catalog_name": catalog, "schema_name": schema, "enabled": True,
        }
    gw_full["rate_limits"] = [
        {"key": "user", "renewal_period": "minute", "calls": rate_limit_user},
        {"key": "endpoint", "renewal_period": "minute", "calls": rate_limit_endpoint},
    ]
    gw_full["usage_tracking_config"] = {"enabled": usage_tracking}
    if ai_gateway_safety:
        gw_full["guardrails"] = {
            "input": {"safety": True, "pii": {"behavior": "BLOCK"}},
            "output": {"safety": True, "pii": {"behavior": "BLOCK"}},
        }

    response = requests.put(f"{host}/api/2.0/serving-endpoints/{endpoint_name}/ai-gateway",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json=gw_full)
    if not response.ok and "Usage tracking" in response.text:
        print("  Usage tracking not supported on this endpoint type — retrying without it")
        gw_full.pop("usage_tracking_config", None)
        usage_tracking = False  # reflects what was actually applied
        response = requests.put(f"{host}/api/2.0/serving-endpoints/{endpoint_name}/ai-gateway",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json=gw_full)

    if response.ok:
        print(f"=== AI Gateway fully configured ===")
        print(f"  Rate limits: {rate_limit_user}/user/min, {rate_limit_endpoint}/endpoint/min")
        print(f"  Safety filter: {ai_gateway_safety}")
        print(f"  Usage tracking: {usage_tracking}")
    else:
        msg = response.json().get("message", "")[:100]
        print(f"  Advanced AI Gateway features not supported on this endpoint type: {msg}")
        print(f"  Inference tables: {'enabled' if inference_tables else 'not requested'} → {payload_table_name}")
# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Smoke test

# COMMAND ----------
if endpoint_ready:
    from framework.inference import extract_response_text
    from framework.team_runtime import TeamRuntime
    try:
        runtime = TeamRuntime(bundle_root, team_name, dbutils)
        data = runtime.invoke(endpoint_name, runtime.query('deployment', 'What is Unity Catalog?'))
        content = extract_response_text(data)
        print(f"Smoke test PASSED: {content[:200]}...")
    except Exception as e:
        print(f"Smoke test FAILED: {e}")
else:
    print("Smoke test skipped — endpoint not ready yet")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Complete audit tracking

# COMMAND ----------
pipeline.end_step(step, status="COMPLETED", output_summary={
    "endpoint_name": endpoint_name,
    "champion_version": champion_version_resolved,
    "champion_traffic": champion_traffic,
    "challenger_enabled": bool(challenger_enabled),
    "endpoint_ready": bool(endpoint_ready),
    "removed_stale_versions": removed_versions,
    "inference_table": payload_table_name,
    "ai_gateway_safety": ai_gateway_safety,
    "rate_limit_user": rate_limit_user,
    "rate_limit_endpoint": rate_limit_endpoint,
})
pipeline.end(status="COMPLETED")

# COMMAND ----------

try:
    dbutils.jobs.taskValues.set(key="endpoint_name", value=endpoint_name)
    dbutils.jobs.taskValues.set(key="inference_table_name", value=payload_table_name)
except Exception as task_value_exc:
    print(f"Warning: failed to publish deploy task values: {task_value_exc}")

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({
    "endpoint_name": endpoint_name,
    "champion_version": champion_version_resolved,
    "endpoint_ready": bool(endpoint_ready),
}))
