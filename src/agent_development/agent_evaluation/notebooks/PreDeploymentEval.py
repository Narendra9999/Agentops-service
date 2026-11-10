# Databricks notebook source
# MAGIC %md
# MAGIC # Pre-Deployment Evaluation
# MAGIC Evaluates the agent locally (no serving endpoint needed).
# MAGIC Runs BEFORE deployment as a quality gate — blocks bad models from deploying.
# MAGIC
# MAGIC Uses Agent.py directly on the cluster with the framework wheel installed.

# COMMAND ----------

# Minimal locator: make the shared startup helper importable before installation.
import sys
from pathlib import Path
_notebook = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
sys.path.insert(0, str(Path(_notebook if _notebook.startswith('/Workspace/') else '/Workspace' + _notebook).parents[3]))
from framework.notebook_bootstrap import initialize_notebook
notebook_runtime = initialize_notebook(dbutils)


import os

widget_defaults = {
    "agent_name": "databricks_docs_agent",
    "catalog": "classic_stable_cykcbe_catalog",
    "schema": "agentops",
    "audit_schema": "agentops_audit",
    "environment": "dev",
    "eval_golden_table": "eval_golden_dataset",
    "eval_adversarial_table": "eval_adversarial_dataset",
    "eval_results_table": "eval_results",
    "team_name": "",
    "team_dir": "",  # legacy override
    "mlflow_experiment_id": "",
    "skip_pre_deployment_eval": "no",
}
for name, default in widget_defaults.items():
    dbutils.widgets.text(name, default)

w_agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
w_schema = dbutils.widgets.get("schema")
w_audit_schema = dbutils.widgets.get("audit_schema")
environment = dbutils.widgets.get("environment")
eval_golden_table = dbutils.widgets.get("eval_golden_table")
eval_adversarial_table = dbutils.widgets.get("eval_adversarial_table")
eval_results_table = dbutils.widgets.get("eval_results_table")
team_name = dbutils.widgets.get("team_name").strip()
team_dir = dbutils.widgets.get("team_dir").strip() or team_name
mlflow_experiment_id = dbutils.widgets.get("mlflow_experiment_id").strip()
skip_pre_deployment_eval = dbutils.widgets.get("skip_pre_deployment_eval").strip().lower() in {"1", "true", "yes", "y"}

if skip_pre_deployment_eval:
    print("Pre-deployment evaluation skipped as requested (skip_pre_deployment_eval=yes)")
    dbutils.notebook.exit("SKIPPED: skip_pre_deployment_eval=yes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install dependencies

# COMMAND ----------

# Install the registered model's exact environment before restarting Python.
# Reinstalling MLflow later and evicting sys.modules leaves protobuf's global
# descriptor pool populated by the old MLflow version.
import glob
import os
import subprocess
import sys

import mlflow
import mlflow.pyfunc

from framework.notebook_bootstrap import dependency_find_links, install_air_gapped_packages

vol_path = dependency_find_links("312")
mlflow.set_registry_uri("databricks-uc")
from framework.team_runtime import candidate_model_uri
from framework.team_config import load_team_settings
_bootstrap_team = load_team_settings(team_name, bundle_root=os.path.dirname(sys.path[0])) if team_name else {}
bootstrap_model_uri = candidate_model_uri(dbutils, ".".join([
    _bootstrap_team.get("catalog") or catalog,
    _bootstrap_team.get("schema") or w_schema,
    _bootstrap_team.get("agent_name") or w_agent_name,
]))
deps_file = mlflow.pyfunc.get_model_dependencies(bootstrap_model_uri)
artifact_dir = os.path.dirname(deps_file)

for local_whl in sorted(glob.glob(os.path.join(artifact_dir, "artifacts", "**", "*.whl"), recursive=True)):
    print(f"Installing bundled wheel: {local_whl}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", local_whl, "--no-deps", "-q"])

external_deps = []
with open(deps_file) as requirements:
    for line in requirements:
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        # Bundled artifacts (e.g. the framework wheel) — already installed above.
        if requirement.startswith("./") or requirement.startswith("/"):
            print(f"Skipping bundled artifact requirement: {requirement}")
            continue
        external_deps.append(requirement)

if external_deps:
    install_air_gapped_packages(external_deps)
    print(f"Installed {len(external_deps)} dependencies for {bootstrap_model_uri}")

dbutils.library.restartPython()


# COMMAND ----------

# Re-read widgets after restart
# Minimal locator: make the shared startup helper importable before installation.
import sys
from pathlib import Path
_notebook = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
sys.path.insert(0, str(Path(_notebook if _notebook.startswith('/Workspace/') else '/Workspace' + _notebook).parents[3]))
from framework.notebook_bootstrap import initialize_notebook
notebook_runtime = initialize_notebook(dbutils)

import os

os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "2"
print(f"MLFLOW_GENAI_EVAL_MAX_WORKERS={os.environ['MLFLOW_GENAI_EVAL_MAX_WORKERS']}")

from framework.notebook_bootstrap import dependency_find_links, install_air_gapped_packages
from framework.notebook_bootstrap import (
    resolve_bundle_paths, resolve_mlflow_experiment_id, resolve_team_settings,
)

w_agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
w_schema = dbutils.widgets.get("schema")
w_audit_schema = dbutils.widgets.get("audit_schema")
environment = dbutils.widgets.get("environment")
eval_golden_table = dbutils.widgets.get("eval_golden_table")
eval_adversarial_table = dbutils.widgets.get("eval_adversarial_table")
eval_results_table = dbutils.widgets.get("eval_results_table")
team_name = dbutils.widgets.get("team_name").strip()
team_dir = dbutils.widgets.get("team_dir").strip() or team_name
mlflow_experiment_id = resolve_mlflow_experiment_id(
    dbutils, dbutils.widgets.get("mlflow_experiment_id"), upstream_task_key="register_model",
)

# Recompute wheel path after restart (air-gapped environment — volume is mandatory).
vol_path = dependency_find_links("312")

# Resolve bundle root for team_config helper
bundle_paths = resolve_bundle_paths(dbutils)
nb_path = bundle_paths["nb_path"]
nb_dir = bundle_paths["nb_dir"]
src_root = bundle_paths["src_root"]  # .../files/src
bundle_root = bundle_paths["bundle_root"]  # .../files
project_root = src_root  # back-compat alias used later for fixture path resolution

# Resolve team settings — team config wins when team_name is set; widgets fill gaps only.
resolved = resolve_team_settings(team_name, bundle_root, {
    "catalog": catalog, "agent_name": w_agent_name, "schema": w_schema, "audit_schema": w_audit_schema,
})
catalog = resolved["catalog"]
agent_name = resolved["agent_name"]
schema = resolved["schema"]
audit_schema = resolved["audit_schema"]
print(f"team_name={team_name!r}  resolved → catalog={catalog!r}, agent_name={agent_name!r} schema={schema!r} audit_schema={audit_schema!r}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0b. Start audit tracking

# COMMAND ----------

from framework.notebook_bootstrap import start_audit_pipeline

pipeline = start_audit_pipeline(
    catalog=catalog, audit_schema=audit_schema,
    pipeline_name="pre_deployment_eval", agent_name=agent_name, environment=environment,
    spark=spark, dbutils=dbutils, triggered_by="pipeline", depends_on="register_model",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Ensure evaluation datasets exist in UC

# COMMAND ----------

import pandas as pd
import sys

from framework.notebook_bootstrap import resolve_fixture_path


def resolve_fixture(filename):
    return resolve_fixture_path(nb_dir, bundle_root, team_dir, filename)

# Golden dataset → UC table (if not already there)
from framework.evaluation.evaluation_pipeline import (
    prepare_golden_dataset_for_spark,
    restore_golden_dataset_from_spark,
)
try:
    golden_pd = spark.table(f"{catalog}.{schema}.{eval_golden_table}").toPandas()
    golden_pd = restore_golden_dataset_from_spark(golden_pd)
    print(f"Golden dataset: {len(golden_pd)} rows from UC table")
except Exception:
    golden_df = pd.read_json(resolve_fixture("golden_dataset.json"))
    golden_spark_df = prepare_golden_dataset_for_spark(golden_df)
    spark.createDataFrame(golden_spark_df.to_dict("records")).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{eval_golden_table}")
    golden_pd = golden_df
    print(f"Golden dataset: {len(golden_pd)} rows loaded from JSON")

# Adversarial dataset → UC table
try:
    adversarial_pd = spark.table(f"{catalog}.{schema}.{eval_adversarial_table}").toPandas()
    print(f"Adversarial dataset: {len(adversarial_pd)} rows from UC table")
except Exception:
    adversarial_df = pd.read_json(resolve_fixture("adversarial_dataset.json"))
    spark.createDataFrame(adversarial_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{eval_adversarial_table}")
    adversarial_pd = adversarial_df
    print(f"Adversarial dataset: {len(adversarial_pd)} rows loaded from JSON")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load the agent locally (same as serving container)

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from framework.team_runtime import candidate_model_uri

# Load the registered model via mlflow.pyfunc.load_model()
# This tests the exact same code path the serving container uses
model_name = f"{catalog}.{schema}.{agent_name}"
client = MlflowClient()

# Get the latest version (just registered by RegisterModel step)
model_uri = candidate_model_uri(dbutils, model_name)
print(f"Loading candidate model: {model_uri}")

# Use a shared experiment ID when provided; otherwise keep default notebook context.
if mlflow_experiment_id:
    mlflow.set_experiment(experiment_id=mlflow_experiment_id)
    print(f"Using MLflow experiment_id from pipeline: {mlflow_experiment_id}")
else:
    print("Using default notebook MLflow experiment (no custom set_experiment)")

from framework.team_runtime import TeamRuntime
team_runtime = TeamRuntime(bundle_root, team_name, dbutils)
loaded_model = mlflow.pyfunc.load_model(model_uri)
if loaded_model is None:
    raise RuntimeError(f"Failed to load model {model_uri}: mlflow.pyfunc.load_model() returned None. Check model registry, dependencies, and MLflow logs above.")
print(f"Model loaded successfully via mlflow.pyfunc.load_model()")

# Quick sanity check
test_input = team_runtime.request(team_runtime.query('pre_deployment', 'What is Databricks?'))
test_output = loaded_model.predict(test_input)
if test_output is None:
    raise RuntimeError(f"Model predict() returned None. Model may not be a valid ResponsesAgent.")
print(f"Sanity check passed: {str(test_output)[:100]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run guardrail evaluation (adversarial dataset)

# COMMAND ----------

import yaml
from framework.guardrails.pre_llm import PreLLMGuardrails
from framework.evaluation.evaluation_pipeline import run_guardrail_evaluation
from framework.notebook_bootstrap import resolve_agent_config_path

cfg_path = resolve_agent_config_path(bundle_root, team_name)
with open(cfg_path) as f:
    agent_config = yaml.safe_load(f)

pre_guardrails = PreLLMGuardrails(agent_config.get("guardrails", {}).get("pre_llm", {}))

guardrail_results = run_guardrail_evaluation(
    agent=type("Agent", (), {"pre_llm_guardrails": pre_guardrails})(),
    adversarial_dataset=adversarial_pd,
)

print(f"Block accuracy:   {guardrail_results['block_accuracy']:.2%}")
print(f"Pass accuracy:    {guardrail_results['pass_accuracy']:.2%}")
print(f"Overall accuracy: {guardrail_results['overall_accuracy']:.2%}")
print(f"False positives:  {guardrail_results['false_positives']}")
print(f"False negatives:  {guardrail_results['false_negatives']}")

for d in guardrail_results["details"]:
    if not d["correct"]:
        print(f"  WRONG: [{d['attack_type']}] should_block={d['should_block']}, was_blocked={d['was_blocked']}: {d['request']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Run quality evaluation (golden dataset — local agent)

# COMMAND ----------

# Use the default notebook experiment (do NOT set a custom experiment path —
# custom paths require UC trace storage which is unavailable in this workspace).

@mlflow.trace(name="pre_deployment_predict", span_type="AGENT")
def predict_fn(**kwargs):
    """Run the local model inside MLflow evaluation to retain its complete trace."""
    query = kwargs.get("query")
    if query is None:
        inputs = kwargs.get("inputs", kwargs.get("input", kwargs))
        query = inputs.get("query", inputs) if isinstance(inputs, dict) else inputs

    from framework.inference import extract_response_text

    result = loaded_model.predict(
        team_runtime.request(str(query))
    )
    return team_runtime.evaluation_output(result)

# Prepare eval data — mlflow.genai.evaluate() is invoked by run_evaluation()
# below, which builds 'inputs'/'expectations' from golden_pd's
# request/expected_response/context/expected_output columns.

# Load scorers based on config.yaml evaluation settings
# Supports: "builtin" (default), "llm_judge", "domain", or "all"
# Groups are flattened into one evaluation so each row produces one agent trace.
sys.path.insert(0, project_root)
from agent_development.agent_evaluation.evaluation.scorer_loader import (
    load_scorer_groups, get_thresholds,
)
from framework.evaluation.evaluation_pipeline import run_evaluation

eval_config = agent_config.get("evaluation", {})
scorer_mode = eval_config.get("scorer_mode", "builtin")
eval_thresholds = get_thresholds(eval_config)

# Load scorer groups for parallel execution (mode="all" → 3 groups)
# Team isolation: load scorer YAMLs from team's scorers/ dir if team_name is set.
team_scorers_dir = os.path.join(bundle_root, "src", "teams", team_name, "scorers") if team_name else None
scorer_groups = load_scorer_groups(eval_config, team_scorers_dir=team_scorers_dir)
total_scorers = sum(len(s) for s in scorer_groups.values())
print(f"Scorer mode: {scorer_mode} → {total_scorers} scorers in {len(scorer_groups)} groups")
for group_name, scorers in scorer_groups.items():
    print(f"  {group_name}: {len(scorers)} scorers")

# Let MLflow invoke the local model so scorer assessments are attached to the
# agent's real predict trace instead of a synthetic trace with precomputed output.
eval_run = run_evaluation(
    eval_dataset=golden_pd,
    scorer_groups=scorer_groups,
    thresholds=eval_thresholds,
    predict_fn=predict_fn,
    row_tags={"agentops.trace_role": "prediction", "agentops.evaluation_phase": "pre_deployment"},
    use_default_thresholds=False,
)
eval_metrics = eval_run["metrics"]

print(f"\n=== Pre-Deployment Evaluation Metrics ===")
for k, v in sorted(eval_metrics.items()):
    print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Quality gate decision

# COMMAND ----------
from framework.evaluation.evaluation_pipeline import compute_quality_gate, print_quality_gate_report

GUARDRAIL_ACCURACY_THRESHOLD = eval_config.get("guardrail_accuracy_threshold", 0.80)
guardrail_accuracy = guardrail_results["overall_accuracy"]

gate_result = compute_quality_gate(eval_thresholds or {}, eval_metrics, guardrail_accuracy, GUARDRAIL_ACCURACY_THRESHOLD)
quality_gate_results = gate_result["quality_gate_results"]
quality_passed = gate_result["quality_passed"]
guardrail_passed = gate_result["guardrail_passed"]
gate_passed = gate_result["gate_passed"]

print_quality_gate_report(
    "PRE-DEPLOYMENT GATE", quality_gate_results, guardrail_accuracy, GUARDRAIL_ACCURACY_THRESHOLD,
    guardrail_passed, gate_passed, guardrail_pct_decimals=1, header_bottom_separator=True,
    failure_footer_message="\nERROR: Pre-deployment gate FAILED — blocking deployment",
)
# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save results to audit

# COMMAND ----------
from framework.evaluation.evaluation_pipeline import save_eval_results_to_table, raise_if_gate_failed

# Get per-row results from the traced evaluation run.
per_row = eval_run["per_row_df"]

print(f"per_row is None: {per_row is None}")
if per_row is not None:
    print(f"Shape: {per_row.shape}, Columns: {list(per_row.columns)}")

evaluation_id = save_eval_results_to_table(
    spark=spark,
    eval_result={
        "per_row_df": per_row,
        "passed": gate_passed,
        "metrics": eval_metrics,
    },
    catalog=catalog,
    audit_schema=audit_schema,
    agent_name=agent_name,
    environment=environment,
    results_table_name=f"pre_{eval_results_table}",
)
print(f"Pre-deployment results saved: evaluation_id={evaluation_id}")

# Fail the task if quality gate was not passed
raise_if_gate_failed(
    gate_passed, quality_gate_results, guardrail_accuracy, GUARDRAIL_ACCURACY_THRESHOLD, guardrail_passed,
    evaluation_id, failure_heading="Pre-deployment quality gate FAILED",
    closing_message="Deployment blocked until quality metrics meet thresholds. Review",
    guardrail_pct_decimals=1,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Complete audit tracking

# COMMAND ----------

step = pipeline.start_step("guardrail_evaluation", step_order=1, step_type="pre_eval", depends_on="vector_search_setup")
pipeline.end_step(step, status="COMPLETED", output_summary={
    "block_accuracy": guardrail_results["block_accuracy"],
    "pass_accuracy": guardrail_results["pass_accuracy"],
    "overall_accuracy": guardrail_results["overall_accuracy"],
    "false_positives": guardrail_results["false_positives"],
    "false_negatives": guardrail_results["false_negatives"],
})

# Build full quality summary with all metrics
quality_summary = {
    "scorer_mode": scorer_mode,
    "scorer_groups": {k: len(v) for k, v in scorer_groups.items()},
    "gate_passed": gate_passed,
    "all_metrics": {k: round(v, 3) if isinstance(v, float) else v for k, v in eval_metrics.items()},
}

step = pipeline.start_step("quality_evaluation", step_order=2, step_type="pre_eval", depends_on="guardrail_evaluation")
pipeline.end_step(step, status="COMPLETED", records_processed=len(golden_pd), output_summary=quality_summary)

pipeline.end(status="COMPLETED" if gate_passed else "FAILED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Output for pipeline

# COMMAND ----------

import json

quality_scores = {
    metric: float(gate["actual"])
    for metric, gate in quality_gate_results.items()
}

dbutils.notebook.exit(json.dumps({
    "gate_passed": bool(gate_passed),
    "quality_passed": bool(quality_passed),
    "guardrail_accuracy": float(guardrail_accuracy),
    "quality_scores": quality_scores,
    "evaluation_id": str(evaluation_id),
}))
