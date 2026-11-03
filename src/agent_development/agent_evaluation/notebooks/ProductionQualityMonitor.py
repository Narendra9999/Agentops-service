# Databricks notebook source
# MAGIC %md
# MAGIC # Production Quality Monitor
# MAGIC Scores live traffic — read from the Model Serving **inference table**
# MAGIC (resolved from the serving endpoint's actual AI Gateway config, enabled
# MAGIC in DeployAgent.py) — with the same scorers used offline (builtin +
# MAGIC llm_judge + domain, config-driven). Runs scorers locally via
# MAGIC `mlflow.genai.evaluate()`, so none of the "remote execution"
# MAGIC serialization restrictions that apply to MLflow's native
# MAGIC production-monitoring scorers apply here. Results are written to a
# MAGIC governed Delta table for dashboards.
# MAGIC
# MAGIC **Incremental by default**: on the very first run for an agent (no prior
# MAGIC rows in the results table), it scores the *entire* payload table's
# MAGIC history. Every run after that only scores days since the last one
# MAGIC already recorded — no manual date bookkeeping needed. `run_date` or
# MAGIC `backfill_start_date`/`backfill_end_date` widgets override this for
# MAGIC ad-hoc single-day or ranged re-runs.
# MAGIC
# MAGIC Runs as its own scheduled job (see resources/production-monitoring-resource.yml),
# MAGIC independent of the deploy pipeline — it reads whatever traffic has
# MAGIC accumulated on the endpoint, regardless of when/whether a deploy ran.

# COMMAND ----------

widget_defaults = {
    "agent_name": "databricks_docs_agent",
    "catalog": "classic_stable_cykcbe_catalog",
    "schema": "agentops",
    "audit_schema": "agentops_audit",
    "environment": "dev",
    "team_name": "",
    "team_dir": "",
    "agent_endpoint": "",
    "inference_table": "",
    "results_table": "production_quality_scores",
    "sample_fraction": "0.5",
    "run_date": "",
    "backfill_start_date": "",
    "backfill_end_date": "",
    "skip_production_quality_monitor": "no",
}
for name, default in widget_defaults.items():
    dbutils.widgets.text(name, default)

skip_production_quality_monitor = dbutils.widgets.get("skip_production_quality_monitor").strip().lower() in {"1", "true", "yes", "y"}

if skip_production_quality_monitor:
    print("Production quality monitor skipped as requested (skip_production_quality_monitor=yes)")
    dbutils.notebook.exit("SKIPPED: skip_production_quality_monitor=yes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install dependencies

# COMMAND ----------

from framework.notebook_bootstrap import install_air_gapped_packages

install_air_gapped_packages([
    "databricks-agents", "mlflow>=3.7.0", "databricks-langchain[memory]", "langchain",
    "langchain-community", "databricks-vectorsearch", "sqlglot", "tabulate", "numpy", "pandas",
])
dbutils.library.restartPython()

# COMMAND ----------

# Re-read widgets after restart (restartPython clears Python state, not dbutils)
import os
import sys

from framework.notebook_bootstrap import (
    resolve_agent_config_path, resolve_bundle_paths, resolve_team_settings, start_audit_pipeline,
)

_w_agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
_w_schema = dbutils.widgets.get("schema")
_w_audit_schema = dbutils.widgets.get("audit_schema")
environment = dbutils.widgets.get("environment")
team_name = dbutils.widgets.get("team_name").strip()
team_dir = dbutils.widgets.get("team_dir").strip() or team_name
_w_agent_endpoint = dbutils.widgets.get("agent_endpoint").strip()
_w_inference_table = dbutils.widgets.get("inference_table").strip()
_w_results_table = dbutils.widgets.get("results_table").strip()
_w_sample_fraction = float(dbutils.widgets.get("sample_fraction") or 0.5)
run_date = dbutils.widgets.get("run_date").strip()
backfill_start_date = dbutils.widgets.get("backfill_start_date").strip()
backfill_end_date = dbutils.widgets.get("backfill_end_date").strip()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Resolve team settings + scorer config

# COMMAND ----------

bundle_paths = resolve_bundle_paths(dbutils)
_bundle_root = bundle_paths["bundle_root"]
_project_root = bundle_paths["src_root"]
sys.path.insert(0, _project_root)

resolved = resolve_team_settings(team_name, _bundle_root, {
    "catalog": catalog, "agent_name": _w_agent_name, "schema": _w_schema,
    "audit_schema": _w_audit_schema, "agent_endpoint": _w_agent_endpoint,
    "monitoring_results_table": _w_results_table or "production_quality_scores",
})
catalog = resolved["catalog"]
agent_name = resolved["agent_name"]
schema = resolved["schema"]
audit_schema = resolved["audit_schema"]
agent_endpoint = resolved["agent_endpoint"] or agent_name
results_table = resolved["monitoring_results_table"]
sample_fraction = _w_sample_fraction
print(f"team_name={team_name!r} resolved → agent_name={agent_name!r} schema={schema!r} "
      f"agent_endpoint={agent_endpoint!r} results_table={results_table!r} sample_fraction={sample_fraction!r}")

from databricks.sdk import WorkspaceClient


def _resolve_inference_table(endpoint_name: str):
    """Query the serving endpoint's real AI Gateway config for its inference
    table — agents.deploy()/DeployAgent.py can enable it under a different
    table_name_prefix than the `{endpoint}_payload` default guess."""
    try:
        endpoint = WorkspaceClient().api_client.do("GET", f"/api/2.0/serving-endpoints/{endpoint_name}")
        itc = (endpoint.get("ai_gateway") or {}).get("inference_table_config")
        if not itc or not itc.get("enabled", False):
            return None
        prefix = itc.get("table_name_prefix") or endpoint_name
        itc_catalog = itc.get("catalog_name") or catalog
        itc_schema = itc.get("schema_name") or schema
        return f"{itc_catalog}.{itc_schema}.`{prefix}_payload`"
    except Exception as exc:
        print(f"WARNING: could not query endpoint '{endpoint_name}' for inference table config: {exc}")
        return None


if _w_inference_table:
    inference_table = _w_inference_table
    print(f"Using explicit inference_table override: {inference_table}")
else:
    inference_table = _resolve_inference_table(agent_endpoint)
    if inference_table:
        print(f"Resolved inference table from endpoint config: {inference_table}")
    else:
        inference_table = f"{catalog}.{schema}.`{agent_endpoint}_payload`"
        print(f"WARNING: could not confirm inference table from endpoint '{agent_endpoint}' "
              f"(AI Gateway inference tables may be disabled) — falling back to default guess: {inference_table}")

pipeline = start_audit_pipeline(
    catalog=catalog, audit_schema=audit_schema,
    pipeline_name="production_quality_monitor", agent_name=agent_name, environment=environment,
    triggered_by="scheduled", depends_on="", spark=spark, dbutils=dbutils,
)
_step = pipeline.start_step("score_production_payloads", step_order=1, step_type="monitoring", depends_on="")

# Load evaluation config — same team/shared resolution used by pre/post-deployment eval.
import yaml
_cfg_path = resolve_agent_config_path(_bundle_root, team_name)
print(f"Loading evaluation config: {_cfg_path}")
with open(_cfg_path) as f:
    _agent_config = yaml.safe_load(f)
eval_config = _agent_config.get("evaluation", {})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Bind the agent's MLflow experiment (for trace/metric linkage)

# COMMAND ----------

import mlflow

mlflow.set_tracking_uri("databricks")
try:
    # Same naming convention RegisterModel.py uses — all jobs in this bundle
    # run as the same service principal (see databricks.yml run_as), so this
    # resolves to the identical experiment without needing cross-job task values.
    _user = spark.sql("SELECT current_user()").first()[0]
    _experiment_name = f"/Users/{_user}/{agent_name}"
    _exp = mlflow.get_experiment_by_name(_experiment_name)
    if _exp:
        mlflow.set_experiment(experiment_id=_exp.experiment_id)
        print(f"Using MLflow experiment: {_experiment_name} ({_exp.experiment_id})")
    else:
        print(f"Experiment '{_experiment_name}' not found — using default notebook experiment.")
except Exception as e:
    print(f"WARNING: could not resolve agent experiment, using default: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load scorers (config-driven — builtin + llm_judge + domain)

# COMMAND ----------

from agent_development.agent_evaluation.evaluation.scorer_loader import (
    load_scorer_groups, run_parallel_evaluation,
)

_team_scorers_dir = os.path.join(_bundle_root, "src", "teams", team_name, "scorers") if team_name else None
scorer_groups = load_scorer_groups(eval_config, team_scorers_dir=_team_scorers_dir)
total_scorers = sum(len(s) for s in scorer_groups.values())
print(f"Loaded {total_scorers} scorers in {len(scorer_groups)} groups: "
      f"{{{', '.join(f'{k}: {len(v)}' for k, v in scorer_groups.items())}}}")

if total_scorers == 0:
    pipeline.end_step(_step, status="FAILED", output_summary={"error": "no scorers configured"})
    pipeline.end(status="FAILED")
    raise RuntimeError("No scorers configured (evaluation.scorer_mode / builtin_scorers / domain_scorers) — nothing to monitor.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Determine which day(s) to score
# MAGIC Priority: explicit `backfill_start_date`/`backfill_end_date` range >
# MAGIC explicit `run_date` > incremental (resume from the day after the last
# MAGIC one already recorded in the results table; first-ever run scores the
# MAGIC entire payload table's history).

# COMMAND ----------

from datetime import datetime, timedelta, timezone

_yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
_results_fqn = f"{catalog}.{schema}.{results_table}"

if backfill_start_date:
    _start = datetime.strptime(backfill_start_date, "%Y-%m-%d").date()
    _end = datetime.strptime(backfill_end_date, "%Y-%m-%d").date() if backfill_end_date else _yesterday
    if _end < _start:
        raise ValueError(f"backfill_end_date ({_end}) is before backfill_start_date ({_start})")
    dates_to_score = [_start + timedelta(days=i) for i in range((_end - _start).days + 1)]
    print(f"Explicit backfill range: {len(dates_to_score)} day(s) from {_start} to {_end}")
elif run_date:
    dates_to_score = [datetime.strptime(run_date, "%Y-%m-%d").date()]
    print(f"Explicit single-day run: {dates_to_score[0]}")
else:
    _last_scored = None
    try:
        _watermark_row = spark.sql(
            f"SELECT MAX(monitor_date) AS last_date FROM {_results_fqn} "
            f"WHERE agent_name = '{agent_name}' AND team_name = '{team_name}'"
        ).first()
        if _watermark_row and _watermark_row["last_date"]:
            _last_scored = datetime.strptime(str(_watermark_row["last_date"]), "%Y-%m-%d").date()
    except Exception:
        _last_scored = None  # results table doesn't exist yet — day-0 bootstrap

    if _last_scored:
        _start = _last_scored + timedelta(days=1)
        if _start > _yesterday:
            dates_to_score = []
            print(f"Already up to date through {_last_scored} — nothing new to score.")
        else:
            dates_to_score = [_start + timedelta(days=i) for i in range((_yesterday - _start).days + 1)]
            print(f"Incremental run: resuming from {_start} through {_yesterday} "
                  f"({len(dates_to_score)} day(s)); last scored = {_last_scored}")
    else:
        # Day-0 bootstrap — score every distinct request_date present in the
        # inference table (full history), up to yesterday.
        _date_rows = (
            spark.table(inference_table)
            .where(f"request_date <= date('{_yesterday.isoformat()}')")
            .select("request_date").distinct().orderBy("request_date").collect()
        )
        dates_to_score = [r["request_date"] for r in _date_rows]
        print(f"Day-0 bootstrap: no prior results found — scoring entire payload "
              f"table history ({len(dates_to_score)} day(s) with traffic).")

if not dates_to_score:
    pipeline.end_step(_step, status="COMPLETED", output_summary={"rows_scored": 0, "days_requested": 0})
    pipeline.end(status="COMPLETED")
    dbutils.notebook.exit("SKIPPED: already up to date, nothing new to score")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Score each day

# COMMAND ----------

import json
from framework.inference import extract_request_text, extract_response_text


def _score_one_day(monitor_date) -> dict:
    """Score inference-table payloads for a single date. Returns a result
    summary for SKIPPED (no data) or COMPLETED; raises on genuine failures
    (table read, evaluate, write) so the caller can record/report them."""
    payload_df = (
        spark.table(inference_table)
        .where(f"request_date = date('{monitor_date.isoformat()}')")
        .where("status_code = '200'")
        .sample(withReplacement=False, fraction=sample_fraction, seed=42)
    )
    payload_pdf = payload_df.toPandas()

    if payload_pdf.empty:
        print(f"[{monitor_date}] No payloads found — nothing to score.")
        return {"date": monitor_date, "status": "SKIPPED", "rows_scored": 0}

    print(f"[{monitor_date}] Sampled {len(payload_pdf)} payloads (fraction={sample_fraction})")

    eval_data = []
    for _, row in payload_pdf.iterrows():
        try:
            request_json = json.loads(row["request"]) if row["request"] else {}
            response_json = json.loads(row["response"]) if row["response"] else {}
        except (json.JSONDecodeError, TypeError):
            continue

        if not isinstance(response_json, dict) or not any(
            isinstance(response_json.get(key), list) and response_json[key]
            for key in ("output", "messages", "choices")
        ):
            continue

        query = extract_request_text(request_json)
        response_text = extract_response_text(response_json)
        if not query or not response_text:
            continue

        eval_data.append({
            "request_id": str(row.get("databricks_request_id", "")),
            "inputs": {"query": query},
            "outputs": response_text,
            "expectations": {},  # No ground truth for live traffic — builtin scorers
                                  # that need it (e.g. Correctness) score as "not applicable".
            "tags": {"agentops.trace_role": "production_monitor", "agentops.monitor_date": str(monitor_date)},
        })

    if not eval_data:
        print(f"[{monitor_date}] All sampled payloads failed to parse — nothing to score.")
        return {"date": monitor_date, "status": "SKIPPED", "rows_scored": 0}

    print(f"[{monitor_date}] Parsed {len(eval_data)}/{len(payload_pdf)} payloads into scorable rows")

    eval_result = run_parallel_evaluation(eval_data, scorer_groups)

    print(f"[{monitor_date}] === Metrics ===")
    for k, v in sorted(eval_result.get("metrics", {}).items()):
        print(f"  {k}: {v}")

    scores_pdf = eval_result["tables"]["eval_results"].copy()
    if "request_id" not in scores_pdf.columns:
        scores_pdf["request_id"] = [item["request_id"] for item in eval_data]

    scores_pdf["monitor_date"] = str(monitor_date)
    scores_pdf["monitor_run_ts_utc"] = datetime.now(timezone.utc).isoformat()
    scores_pdf["sample_fraction"] = sample_fraction
    scores_pdf["team_name"] = team_name
    scores_pdf["agent_name"] = agent_name
    scores_pdf["environment"] = environment

    # Convert nested feedback/rationale objects into JSON strings for stable Delta writes.
    for column in scores_pdf.columns:
        if scores_pdf[column].dtype == "object":
            scores_pdf[column] = scores_pdf[column].map(
                lambda value: json.dumps(value, default=str)
                if isinstance(value, (dict, list, tuple))
                else value
            )

    scores_sdf = spark.createDataFrame(scores_pdf)
    (
        scores_sdf.write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(_results_fqn)
    )
    print(f"[{monitor_date}] Scored {len(eval_data)} payloads using {total_scorers} scorers; "
          f"wrote results to {_results_fqn}")
    return {"date": monitor_date, "status": "COMPLETED", "rows_scored": len(eval_data)}


day_results = []
for _d in dates_to_score:
    print(f"\n=== Scoring {_d} ===")
    try:
        day_results.append(_score_one_day(_d))
    except Exception as e:
        print(f"[{_d}] FAILED: {e}")
        day_results.append({"date": _d, "status": "FAILED", "rows_scored": 0, "error": str(e)[:500]})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Finalize audit tracking

# COMMAND ----------

total_rows = sum(r["rows_scored"] for r in day_results)
failed_days = [r for r in day_results if r["status"] == "FAILED"]
completed_days = [r for r in day_results if r["status"] == "COMPLETED"]
skipped_days = [r for r in day_results if r["status"] == "SKIPPED"]

summary = {
    "days_requested": len(dates_to_score),
    "days_completed": len(completed_days),
    "days_skipped_no_data": len(skipped_days),
    "days_failed": len(failed_days),
    "total_rows_scored": total_rows,
    "scorer_count": total_scorers,
    "results_table": _results_fqn,
}
print(f"\n=== Summary === {summary}")

if failed_days:
    pipeline.end_step(_step, status="FAILED", output_summary={**summary, "failed_dates": [str(r["date"]) for r in failed_days]})
    pipeline.end(status="FAILED")
    raise RuntimeError(
        f"Production quality monitor failed for {len(failed_days)}/{len(dates_to_score)} day(s): "
        f"{[str(r['date']) for r in failed_days]}"
    )

pipeline.end_step(_step, status="COMPLETED", output_summary=summary)
pipeline.end(status="COMPLETED")

