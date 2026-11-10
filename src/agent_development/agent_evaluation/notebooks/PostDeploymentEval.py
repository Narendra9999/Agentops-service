# Databricks notebook source
# MAGIC %md
# MAGIC # Agent Evaluation Pipeline
# MAGIC Loads evaluation datasets into UC tables, runs MLflow evaluation + guardrail tests,
# MAGIC saves per-row results, and outputs CI/CD gate decision.

# COMMAND ----------

# Resolve shared framework from the deployed bundle before dependency setup.
import os, sys
dbutils.widgets.text("dependency_source", "volume")
_dependency_source = dbutils.widgets.get("dependency_source").strip()
if _dependency_source not in {"volume", "environment"}:
    raise ValueError("dependency_source must be volume or environment")
os.environ["AGENTOPS_DEPENDENCY_SOURCE"] = _dependency_source
_bootstrap_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_bootstrap_path = _bootstrap_path if _bootstrap_path.startswith('/Workspace/') else '/Workspace' + _bootstrap_path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_bootstrap_path)))))


widget_defaults = {
    "agent_name": "databricks_docs_agent",
    "catalog": "classic_stable_cykcbe_catalog",
    "schema": "agentops",
    "audit_schema": "agentops_audit",
    "environment": "dev",
    "eval_golden_table": "eval_golden_dataset",
    "eval_adversarial_table": "eval_adversarial_dataset",
    "eval_results_table": "eval_results",
    "agent_endpoint": "agentops-docs-chatbot",
    "team_name": "",
    "team_dir": "",  # legacy override
    "mlflow_experiment_id": "",
    "skip_post_deployment_eval": "no",
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
w_agent_endpoint = dbutils.widgets.get("agent_endpoint")
team_name = dbutils.widgets.get("team_name").strip()
team_dir = dbutils.widgets.get("team_dir").strip() or team_name
mlflow_experiment_id = dbutils.widgets.get("mlflow_experiment_id").strip()
skip_post_deployment_eval = dbutils.widgets.get("skip_post_deployment_eval").strip().lower() in {"1", "true", "yes", "y"}

if skip_post_deployment_eval:
    print("Post-deployment evaluation skipped as requested (skip_post_deployment_eval=yes)")
    dbutils.notebook.exit("SKIPPED: skip_post_deployment_eval=yes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install dependencies + start audit tracking

# COMMAND ----------

# Ensure MLflow 3.x with genai module (needed for mlflow.genai.evaluate)
import os

from framework.notebook_bootstrap import install_air_gapped_packages

install_air_gapped_packages([
    "databricks-agents", "mlflow>=3.7.0", "databricks-langchain[memory]", "langchain",
    "langchain-community", "databricks-vectorsearch", "sqlglot", "tabulate", "numpy", "pandas",
])
dbutils.library.restartPython()

# COMMAND ----------

# Re-read widgets after restart
import os, sys
_runtime_nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_runtime_nb = _runtime_nb if _runtime_nb.startswith('/Workspace/') else '/Workspace' + _runtime_nb
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_runtime_nb)))))
import os

from framework.notebook_bootstrap import (
    resolve_bundle_paths, resolve_mlflow_experiment_id, resolve_team_settings, start_audit_pipeline,
)

w_agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
w_schema = dbutils.widgets.get("schema")
w_audit_schema = dbutils.widgets.get("audit_schema")
environment = dbutils.widgets.get("environment")
eval_golden_table = dbutils.widgets.get("eval_golden_table")
eval_adversarial_table = dbutils.widgets.get("eval_adversarial_table")
eval_results_table = dbutils.widgets.get("eval_results_table")
w_agent_endpoint = dbutils.widgets.get("agent_endpoint")
team_name = dbutils.widgets.get("team_name").strip()
team_dir = dbutils.widgets.get("team_dir").strip() or team_name
mlflow_experiment_id = resolve_mlflow_experiment_id(
    dbutils, dbutils.widgets.get("mlflow_experiment_id"), upstream_task_key="register_model",
)

# Resolve bundle root for team_config helper
bundle_paths = resolve_bundle_paths(dbutils)
nb_path = bundle_paths["nb_path"]
nb_dir = bundle_paths["nb_dir"]
bundle_root = bundle_paths["bundle_root"]
project_root = bundle_paths["src_root"]

# Resolve team settings — team config wins when team_name is set; widgets fill gaps only.
resolved = resolve_team_settings(team_name, bundle_root, {
    "catalog": catalog, "agent_name": w_agent_name, "schema": w_schema,
    "audit_schema": w_audit_schema, "agent_endpoint": w_agent_endpoint,
})
catalog = resolved["catalog"]
agent_name = resolved["agent_name"]
schema = resolved["schema"]
audit_schema = resolved["audit_schema"]
agent_endpoint = resolved["agent_endpoint"]
print(f"team_name={team_name!r}  resolved → catalog={catalog!r}, agent_name={agent_name!r} schema={schema!r} audit_schema={audit_schema!r} agent_endpoint={agent_endpoint!r}")

# COMMAND ----------

pipeline = start_audit_pipeline(
    catalog=catalog, audit_schema=audit_schema,
    pipeline_name="post_deployment_eval", agent_name=agent_name, environment=environment,
    spark=spark, dbutils=dbutils, triggered_by="cicd", depends_on="smoke_test",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load evaluation datasets into UC tables

# COMMAND ----------

import pandas as pd
import json, os

# Create audit schema and tables if they don't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{audit_schema}")
from framework.audit.audit_logger import get_audit_ddls
from framework.notebook_bootstrap import resolve_fixture_path

for table_name, ddl in get_audit_ddls(catalog, audit_schema).items():
    spark.sql(ddl)
    print(f"Ensured table: {catalog}.{audit_schema}.{table_name}")


def resolve_fixture(filename):
    return resolve_fixture_path(nb_dir, bundle_root, team_dir, filename)

# Golden dataset → UC table
golden_df = pd.read_json(resolve_fixture("golden_dataset.json"))
from framework.evaluation.evaluation_pipeline import (
    prepare_golden_dataset_for_spark,
    restore_golden_dataset_from_spark,
)
golden_spark_df = prepare_golden_dataset_for_spark(golden_df)
spark.createDataFrame(golden_spark_df.to_dict("records")).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{eval_golden_table}")
print(f"Golden: {len(golden_df)} rows → {catalog}.{schema}.{eval_golden_table}")

# Adversarial dataset → UC table
adversarial_df = pd.read_json(resolve_fixture("adversarial_dataset.json"))
spark.createDataFrame(adversarial_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{eval_adversarial_table}")
print(f"Adversarial: {len(adversarial_df)} rows → {catalog}.{schema}.{eval_adversarial_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Run Guardrail Evaluation

# COMMAND ----------

import yaml
from framework.guardrails.pre_llm import PreLLMGuardrails
from framework.evaluation.evaluation_pipeline import run_guardrail_evaluation
from framework.notebook_bootstrap import resolve_agent_config_path

cfg_path = resolve_agent_config_path(bundle_root, team_name)
with open(cfg_path) as f:
    agent_config = yaml.safe_load(f)

pre_guardrails = PreLLMGuardrails(agent_config.get("guardrails", {}).get("pre_llm", {}))
adversarial_pd = spark.table(f"{catalog}.{schema}.{eval_adversarial_table}").toPandas()

guardrail_results = run_guardrail_evaluation(
    agent=type("Agent", (), {"pre_llm_guardrails": pre_guardrails})(),
    adversarial_dataset=adversarial_pd,
)

print(f"Block accuracy:   {guardrail_results['block_accuracy']:.2%}")
print(f"Pass accuracy:    {guardrail_results['pass_accuracy']:.2%}")
print(f"Overall accuracy: {guardrail_results['overall_accuracy']:.2%}")
print(f"False positives:  {guardrail_results['false_positives']}")
print(f"False negatives:  {guardrail_results['false_negatives']}")

# Show incorrect results
for d in guardrail_results["details"]:
    if not d["correct"]:
        print(f"  WRONG: [{d['attack_type']}] should_block={d['should_block']}, was_blocked={d['was_blocked']}: {d['request']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run MLflow Quality Evaluation

# COMMAND ----------

from framework.evaluation.evaluation_pipeline import run_evaluation
import mlflow
import sys, os

if mlflow_experiment_id:
    mlflow.set_experiment(experiment_id=mlflow_experiment_id)
    print(f"Using MLflow experiment_id from pipeline: {mlflow_experiment_id}")
else:
    print("Using default notebook MLflow experiment (no custom set_experiment)")

sys.path.insert(0, project_root)

# Reuses agent_config already loaded for guardrail evaluation above.
from agent_development.agent_evaluation.evaluation.scorer_loader import (
    load_scorer_groups, get_thresholds,
)
eval_config = agent_config.get("evaluation", {})
# Team isolation: load scorer YAMLs from team's scorers/ dir if team_name is set.
team_scorers_dir = os.path.join(bundle_root, "src", "teams", team_name, "scorers") if team_name else None
scorer_groups = load_scorer_groups(eval_config, team_scorers_dir=team_scorers_dir)
eval_thresholds = get_thresholds(eval_config)
scorer_mode = eval_config.get("scorer_mode", "builtin")
total_scorers = sum(len(s) for s in scorer_groups.values())
print(f"Scorer mode: {scorer_mode} → {total_scorers} scorers in {len(scorer_groups)} groups")
for group_name, group_scorers in scorer_groups.items():
    print(f"  {group_name}: {len(group_scorers)} scorers")

golden_pd = spark.table(f"{catalog}.{schema}.{eval_golden_table}").toPandas()
golden_pd = restore_golden_dataset_from_spark(golden_pd)

# Set env vars for endpoint query function
os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
ws_url = spark.conf.get("spark.databricks.workspaceUrl", "")
if not ws_url.startswith("http"):
    ws_url = f"https://{ws_url}"
os.environ["DATABRICKS_HOST"] = ws_url

# Find the deployed endpoint
from databricks.sdk import WorkspaceClient

workspace_client = WorkspaceClient()
endpoint_name = None
try:
    endpoint = workspace_client.serving_endpoints.get(agent_endpoint)
    if endpoint.state and str(endpoint.state.ready).endswith("READY"):
        endpoint_name = agent_endpoint
except Exception:
    pass

if not endpoint_name:
    match_prefix = f"agents_{catalog}-{schema}"
    for endpoint in workspace_client.serving_endpoints.list():
        if endpoint.name.startswith(match_prefix) and endpoint.state and str(endpoint.state.ready).endswith("READY"):
            endpoint_name = endpoint.name
            break
print(f"Evaluating endpoint: {endpoint_name}")

from framework.team_runtime import TeamRuntime
team_runtime = TeamRuntime(bundle_root, team_name, dbutils)

@mlflow.trace(name="post_deployment_predict", span_type="AGENT")
def predict_fn(query, **kwargs):
    from databricks.sdk.core import Config
    client = WorkspaceClient(config=Config(http_timeout_seconds=240))
    result = client.api_client.do("POST", "/serving-endpoints/" + endpoint_name + "/invocations",
                                body=team_runtime.request(str(query)))
    return team_runtime.evaluation_output(result)

# Run one traced endpoint evaluation so MLflow can import the complete serving trace.
# Scorer groups are flattened to avoid invoking the endpoint more than once per row.
eval_result = run_evaluation(
    eval_dataset=golden_pd,
    scorer_groups=scorer_groups,
    thresholds=eval_thresholds,
    predict_fn=predict_fn,
    use_default_thresholds=False,
)

print(f"\n=== Evaluation Metrics ===")
for k, v in sorted(eval_result.get("metrics", {}).items()):
    print(f"  {k}: {v}")

# Show parallel execution timing if available
if eval_result.get("group_results"):
    print(f"\n=== Parallel Execution ===")
    for group_name, group_result in eval_result["group_results"].items():
        duration = group_result.get("duration_ms", 0)
        scorers = group_result.get("scorers", [])
        print(f"  {group_name}: {duration}ms ({len(scorers)} scorers)")

print(f"\n=== Quality Gate ===")
print(f"Overall: {'PASSED' if eval_result['passed'] else 'FAILED'}")
for metric, gate in eval_result.get("gate_results", {}).items():
    status = "PASS" if gate["passed"] else "FAIL"
    print(f"  [{status}] {metric}: {gate['actual']:.3f} (threshold: {gate['threshold']})")


# Use the configured quality gate metrics only; no legacy mean-based fallback.
quality_gate_results = eval_result.get("gate_results", {})
quality_passed = bool(eval_result.get("passed", True))
GUARDRAIL_ACCURACY_THRESHOLD = eval_config.get("guardrail_accuracy_threshold", 0.95)
guardrail_accuracy = guardrail_results["overall_accuracy"]
guardrail_passed = guardrail_accuracy >= GUARDRAIL_ACCURACY_THRESHOLD
gate_passed = quality_passed and guardrail_passed

def metric_to_scorer_name(metric_name: str) -> str:
    """Map an evaluation metric name (e.g. scorer/mean) back to the scorer name."""
    return metric_name.rsplit("/", 1)[0] if "/" in metric_name else metric_name

def metric_alias(metric_name: str) -> str:
    return metric_name.replace("/", "_")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Save per-row results to UC audit table

# COMMAND ----------

from framework.evaluation.evaluation_pipeline import save_eval_results_to_table

evaluation_id = save_eval_results_to_table(
    spark=spark,
    eval_result=eval_result,
    catalog=catalog,
    audit_schema=audit_schema,
    agent_name=agent_name,
    environment=environment,
    results_table_name=eval_results_table,
)

print(f"Per-row results saved: evaluation_id={evaluation_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Preview per-row scores

# COMMAND ----------

configured_metric_names = sorted(quality_gate_results.keys())
preview_selects = ["request"]
for metric_name in configured_metric_names:
    scorer_name = metric_to_scorer_name(metric_name)
    alias = metric_alias(metric_name)
    preview_selects.append(f"round(scores['{scorer_name}'], 3) as {alias}")
preview_selects.append("metrics_json")
preview_selects.append("overall_passed")

display(spark.sql(f"""
    SELECT {', '.join(preview_selects)}
    FROM {catalog}.{audit_schema}.{eval_results_table}
    WHERE evaluation_id = '{evaluation_id}'
    ORDER BY row_index
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Complete audit tracking

# COMMAND ----------

step = pipeline.start_step("guardrail_evaluation", step_order=1, step_type="post_eval", depends_on="smoke_test")
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
    "quality_passed": quality_passed,
    "guardrail_passed": guardrail_passed,
    "guardrail_threshold": GUARDRAIL_ACCURACY_THRESHOLD,
    "configured_metrics": {
        metric: {
            "actual": round(gate["actual"], 3) if isinstance(gate.get("actual"), float) else gate.get("actual"),
            "threshold": gate["threshold"],
            "passed": gate["passed"],
        }
        for metric, gate in quality_gate_results.items()
    },
    "all_metrics": {k: round(v, 3) if isinstance(v, float) else v for k, v in eval_result.get("metrics", {}).items()},
    "gate_results": quality_gate_results,
}
if eval_result.get("group_results"):
    quality_summary["parallel_timing_ms"] = {
        group_name: group_result.get("duration_ms", 0)
        for group_name, group_result in eval_result["group_results"].items()
    }

step = pipeline.start_step("quality_evaluation", step_order=2, step_type="post_eval", depends_on="guardrail_evaluation")
pipeline.end_step(step, status="COMPLETED", records_processed=len(golden_pd), output_summary=quality_summary)

step = pipeline.start_step("save_eval_results", step_order=3, step_type="post_eval")
pipeline.end_step(step, status="COMPLETED", records_processed=len(golden_pd), output_summary={
    "evaluation_id": evaluation_id,
})

pipeline.end(status="COMPLETED" if gate_passed else "FAILED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. CI/CD Gate Output

# COMMAND ----------

from framework.evaluation.evaluation_pipeline import print_quality_gate_report, raise_if_gate_failed

print_quality_gate_report(
    "EVALUATION GATE", quality_gate_results, guardrail_accuracy, GUARDRAIL_ACCURACY_THRESHOLD,
    guardrail_passed, gate_passed, guardrail_pct_decimals=2, footer_separator=True,
)

# Fail the task if quality gate was not passed
raise_if_gate_failed(
    gate_passed, quality_gate_results, guardrail_accuracy, GUARDRAIL_ACCURACY_THRESHOLD, guardrail_passed,
    evaluation_id, failure_heading="Post-deployment quality gate FAILED",
    closing_message="Endpoint remains live but review recommended.",
    guardrail_pct_decimals=2, list_failed_metrics=True,
)

dbutils.notebook.exit(json.dumps({
    "passed": bool(gate_passed),
    "quality_passed": bool(quality_passed),
    "quality_scores": {metric: float(gate["actual"]) for metric, gate in quality_gate_results.items()},
    "guardrail_accuracy": float(guardrail_accuracy),
    "evaluation_id": str(evaluation_id),
}))
