"""
AgentOPS Framework — Evaluation Pipeline
Uses mlflow.evaluate() with model_type="question-answering" for built-in metrics
plus LLM-as-judge custom metrics (accuracy, helpfulness, professionalism).
Saves per-row results to UC audit table.
Used as CI/CD quality gate before environment promotion.
"""

import mlflow
import pandas as pd
import logging
import math
import json
import uuid
import requests
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = {
    # MLflow 3.3.2 uses {name}/mean, MLflow 3.10+ uses {name}/v1/mean
    "accuracy/mean": 3.5,
    "helpfulness/mean": 3.5,
    "professionalism/mean": 3.5,
    "docs_relevance/mean": 3.5,
    "answer_completeness/mean": 3.5,
}

GOLDEN_DATASET_REQUIRED_COLUMNS = {"request", "expected_response"}
GOLDEN_DATASET_OPTIONAL_COLUMNS = {"context", "expected_output"}


def validate_golden_dataset(dataset: pd.DataFrame) -> None:
    """Validate the canonical AgentOPS golden dataset schema."""
    if dataset.empty:
        raise ValueError("Golden dataset must contain at least one row.")

    columns = set(dataset.columns)
    missing = GOLDEN_DATASET_REQUIRED_COLUMNS - columns
    unknown = columns - GOLDEN_DATASET_REQUIRED_COLUMNS - GOLDEN_DATASET_OPTIONAL_COLUMNS
    errors = []
    if missing:
        errors.append(f"missing required columns: {sorted(missing)}")
    if unknown:
        errors.append(f"unsupported columns: {sorted(unknown)}")

    for column in sorted(GOLDEN_DATASET_REQUIRED_COLUMNS & columns):
        invalid_rows = dataset.index[
            dataset[column].isna() | dataset[column].astype(str).str.strip().eq("")
        ].tolist()
        if invalid_rows:
            errors.append(f"'{column}' is empty at row indexes: {invalid_rows[:10]}")

    if "expected_output" in columns:
        invalid_rows = dataset.index[dataset["expected_output"].isna()].tolist()
        if invalid_rows:
            errors.append(f"'expected_output' is empty at row indexes: {invalid_rows[:10]}")

    if errors:
        allowed = sorted(GOLDEN_DATASET_REQUIRED_COLUMNS | GOLDEN_DATASET_OPTIONAL_COLUMNS)
        raise ValueError(
            "Invalid golden dataset schema: " + "; ".join(errors) + ". "
            f"Allowed columns are: {allowed}."
        )


def prepare_golden_dataset_for_spark(dataset: pd.DataFrame) -> pd.DataFrame:
    """Normalize golden dataset columns that require one stable Spark type."""
    validate_golden_dataset(dataset)
    normalized = dataset.copy()
    if "expected_output" not in normalized.columns:
        return normalized

    def serialize(value):
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass
        return json.dumps(value, default=str)

    normalized["expected_output"] = normalized["expected_output"].map(serialize)
    return normalized


def restore_golden_dataset_from_spark(dataset: pd.DataFrame) -> pd.DataFrame:
    """Restore JSON-encoded expected outputs loaded from a Spark table."""
    restored = dataset.copy()
    if "expected_output" not in restored.columns:
        return restored

    def deserialize(value):
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    restored["expected_output"] = restored["expected_output"].map(deserialize)
    validate_golden_dataset(restored)
    return restored


def _build_expectations(dataset: pd.DataFrame) -> list[dict]:
    """Build typed MLflow expectations from supported golden dataset columns."""
    expectation_columns = ["expected_response"]
    if "expected_output" in dataset.columns:
        expectation_columns.append("expected_output")
    return [
        {column: row[column] for column in expectation_columns}
        for _, row in dataset.iterrows()
    ]


def _build_inputs(dataset: pd.DataFrame) -> list[dict]:
    """Build MLflow inputs while preserving optional dataset context."""
    return [
        {
            "query": row["request"],
            **({"context": row["context"]} if "context" in dataset.columns else {}),
        }
        for _, row in dataset.iterrows()
    ]


def _build_endpoint_inputs(dataset: pd.DataFrame) -> list[dict]:
    """Build ResponsesAgent inputs accepted by a Databricks serving endpoint."""
    return [
        {"input": [{"role": "user", "content": row["request"]}]}
        for _, row in dataset.iterrows()
    ]


def load_evaluation_dataset(path_or_table: str, spark=None) -> pd.DataFrame:
    """Load evaluation dataset from JSON file or UC table."""
    if path_or_table.endswith(".json"):
        dataset = pd.read_json(path_or_table)
        validate_golden_dataset(dataset)
        return dataset
    if spark:
        dataset = spark.table(path_or_table).toPandas()
        validate_golden_dataset(dataset)
        return dataset
    from databricks.connect import DatabricksSession
    spark = DatabricksSession.builder.getOrCreate()
    dataset = spark.table(path_or_table).toPandas()
    validate_golden_dataset(dataset)
    return dataset


def _build_query_fn(endpoint_name: str):
    """Build a function that calls the serving endpoint for each eval row."""
    from framework.inference import extract_response_text

    def query_iteration(inputs_df):
        token = os.environ.get("DATABRICKS_TOKEN", "")
        host = os.environ.get("DATABRICKS_HOST", "")
        if not host.startswith("http"):
            host = f"https://{host}"
        url = f"{host}/serving-endpoints/{endpoint_name}/invocations"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        answers = []
        for _, row in inputs_df.iterrows():
            query = row["inputs"]
            try:
                resp = requests.post(url, headers=headers, json={
                    "input": [{"role": "user", "content": query}]
                })
                resp.raise_for_status()
                data = resp.json()
                answers.append(extract_response_text(data))
            except Exception as e:
                logger.error(f"Endpoint query failed: {e}")
                answers.append(f"Error: {e}")
        return answers

    return query_iteration


def save_eval_results_to_table(
    spark,
    eval_result: dict,
    catalog: str,
    audit_schema: str,
    agent_name: str,
    agent_version: str = "1.0.0",
    environment: str = "dev",
    execution_id: str = "",
    results_table_name: str = "eval_results",
) -> str:
    """
    Save per-row evaluation results to the UC eval_results audit table.

    Uses the eval_results_table from mlflow.evaluate() which contains
    per-row scores from both built-in and LLM-as-judge metrics.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        BooleanType,
        DoubleType,
        LongType,
        MapType,
        StringType,
        StructField,
        StructType,
    )

    evaluation_id = str(uuid.uuid4())

    per_row_df = eval_result.get("per_row_df")
    if per_row_df is None or per_row_df.empty:
        logger.warning("No per-row results to save")
        return evaluation_id

    # Log available columns for debugging
    if per_row_df is not None:
        score_cols = [c for c in per_row_df.columns if "/score" in c or "/v1" in c]
        logger.info(f"Available score columns: {score_cols}")

    def _coerce_score(val):
        """Coerce a scorer value to float. Numeric → float; "yes"/"true"/"pass" → 1.0;
        "no"/"false"/"fail" → 0.0. Returns None if unconvertible."""
        if val is None:
            return None
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
        s = str(val).strip().lower()
        if s in ("yes", "true", "pass", "passed"):
            return 1.0
        if s in ("no", "false", "fail", "failed"):
            return 0.0
        return None

    def _extract_all_scores(row_dict):
        """Build {scorer_name: score} for every scorer that ran on this row.

        Sources, in priority order:
        1. MLflow 3.10+ per-row columns: '<scorer>/value' or '<scorer>/v1/score'
        2. MLflow 3.3.2 assessments list: each entry has assessment_name + feedback.value

        Pass/fail string values (yes/no, true/false, pass/fail) are coerced to 1.0/0.0
        so Guidelines and other boolean scorers also appear in the scores map.
        """
        scores: dict[str, float] = {}
        for col, val in row_dict.items():
            if val is None or str(val) in ("None", "nan", ""):
                continue
            scorer_name = None
            if isinstance(col, str):
                if col.endswith("/value"):
                    scorer_name = col[: -len("/value")]
                elif col.endswith("/v1/score"):
                    scorer_name = col[: -len("/v1/score")]
            if scorer_name is None or scorer_name in scores:
                continue
            coerced = _coerce_score(val)
            if coerced is not None:
                scores[scorer_name] = coerced
        # Assessments fallback (MLflow 3.3.2)
        for a in (row_dict.get("assessments") or []):
            if not isinstance(a, dict):
                continue
            name = a.get("assessment_name")
            feedback = a.get("feedback")
            if not name or name in scores:
                continue
            if isinstance(feedback, dict):
                coerced = _coerce_score(feedback.get("value"))
                if coerced is not None:
                    scores[name] = coerced
        return scores

    eval_rows = []
    for position, (i, row) in enumerate(per_row_df.iterrows()):
        row_dict = row.to_dict()
        scores_map = _extract_all_scores(row_dict)
        inputs = row_dict.get("inputs", row_dict.get("request", ""))
        expectations = row_dict.get("expectations", row_dict.get("targets", {}))
        if isinstance(expectations, dict):
            expected_response = expectations.get("expected_response", "")
            expected_output = expectations.get("expected_output")
        else:
            expected_response = expectations
            expected_output = None
        eval_rows.append({
            "evaluation_id": evaluation_id,
            "execution_id": execution_id,
            "row_index": int(i) if isinstance(i, int) or str(i).isdigit() else position,
            "request": (inputs.get("query", str(inputs)) if isinstance(inputs, dict) else str(inputs))[:4000],
            "response": str(row_dict.get("response", row_dict.get("outputs", "")))[:4000],
            "expected_response": str(expected_response)[:4000],
            "expected_output": (
                json.dumps(expected_output, default=str)[:4000]
                if expected_output is not None
                else None
            ),
            "context": str(inputs.get("context", ""))[:4000] if isinstance(inputs, dict) else "",
            # Captures EVERY scorer that ran — team-defined and built-in alike.
            "scores": scores_map,
            # Run-level metric summary, stored as JSON so the metric count can vary.
            "metrics_json": json.dumps(eval_result.get("metrics", {}), default=str),
            "overall_passed": bool(eval_result.get("passed", False)),
            "agent_name": agent_name,
            "agent_version": agent_version,
            "environment": environment,
        })

    table_name = f"{catalog}.{audit_schema}.{results_table_name}"
    result_schema = StructType([
        StructField("evaluation_id", StringType(), False),
        StructField("execution_id", StringType(), True),
        StructField("row_index", LongType(), True),
        StructField("request", StringType(), True),
        StructField("response", StringType(), True),
        StructField("expected_response", StringType(), True),
        StructField("expected_output", StringType(), True),
        StructField("context", StringType(), True),
        StructField("scores", MapType(StringType(), DoubleType()), True),
        StructField("metrics_json", StringType(), True),
        StructField("overall_passed", BooleanType(), True),
        StructField("agent_name", StringType(), True),
        StructField("agent_version", StringType(), True),
        StructField("environment", StringType(), True),
    ])
    df = spark.createDataFrame(eval_rows, schema=result_schema)
    df = df.withColumn("evaluated_at", F.current_timestamp())
    df.write.mode("append").option("mergeSchema", "true").saveAsTable(table_name)

    logger.info(f"Saved {len(eval_rows)} eval results to {table_name} (evaluation_id={evaluation_id})")
    return evaluation_id


@mlflow.trace(span_type="EVALUATION")
def run_evaluation(
    eval_dataset: pd.DataFrame,
    scorers: list = None,
    scorer_groups: dict = None,
    thresholds: dict = None,
    model_endpoint: str = None,
    predict_fn=None,
    row_tags: dict = None,
    use_default_thresholds: bool = True,
) -> dict:
    """
    Run the standardized evaluation pipeline using mlflow.genai.evaluate().

    Supports three modes:
      - model_endpoint: traces a live serving endpoint (one evaluate() call)
      - predict_fn: traces a local callable, e.g. a loaded pyfunc model's
        predict (one evaluate() call) — mutually exclusive with model_endpoint
      - scorers / scorer_groups with neither of the above: evaluates
        precomputed 'outputs', sequential or parallel per group

    Args:
        eval_dataset: DataFrame with 'request', 'expected_response', and optional
            'expected_output' and 'context' columns
        scorers: Flat list of scorer functions (sequential mode)
        scorer_groups: Dict of {group_name: [scorers]} from load_scorer_groups() (parallel mode)
        thresholds: Override default thresholds
        model_endpoint: Serving endpoint name to evaluate (for predict_fn)
        predict_fn: Local traced callable to evaluate instead of a serving
            endpoint (e.g. a loaded pyfunc model's predict). Mutually
            exclusive with model_endpoint.
        row_tags: Optional dict of MLflow trace tags applied to every eval
            row (e.g. {"agentops.trace_role": "prediction"}).
        use_default_thresholds: When True (default), merge DEFAULT_THRESHOLDS
            under the provided thresholds. Set False to gate on exactly the
            provided thresholds and nothing else.

    Returns:
        dict with pass/fail, metrics, gate results, per_row_df, and group timing
    """
    if model_endpoint and predict_fn is not None:
        raise ValueError("Provide either model_endpoint or predict_fn, not both.")

    validate_golden_dataset(eval_dataset)
    if use_default_thresholds:
        thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    else:
        thresholds = dict(thresholds or {})

    # Prepare dataset for mlflow.genai.evaluate()
    eval_df = eval_dataset.copy()
    if "inputs" not in eval_df.columns and "request" in eval_df.columns:
        eval_df["inputs"] = _build_inputs(eval_df)
    if "expectations" not in eval_df.columns:
        eval_df["expectations"] = _build_expectations(eval_df)
    if row_tags:
        eval_df["tags"] = [dict(row_tags) for _ in range(len(eval_df))]
    traced_columns = ["inputs", "expectations"] + (["tags"] if row_tags else [])

    if model_endpoint:
        if "outputs" in eval_df.columns:
            raise ValueError("Endpoint evaluation cannot use precomputed 'outputs'.")
        logger.info(f"Evaluating traced endpoint: {model_endpoint}")
        eval_df["inputs"] = _build_endpoint_inputs(eval_df)
        predict_fn = mlflow.genai.to_predict_fn(f"endpoints:/{model_endpoint}")
        eval_data = eval_df[traced_columns]
    elif predict_fn is not None:
        if "outputs" in eval_df.columns:
            raise ValueError("predict_fn evaluation cannot use precomputed 'outputs'.")
        logger.info("Evaluating local predict_fn (traced)")
        eval_data = eval_df[traced_columns]
    else:
        eval_data = eval_df[["inputs", "outputs", "expectations"]]

    # Run evaluation — parallel if scorer_groups provided, else sequential
    group_results = {}
    if model_endpoint or predict_fn is not None:
        # MLflow requests the full trace and evaluates its complete span tree.
        # Keep one evaluate call so the model/endpoint runs only once per row —
        # this rules out true parallel per-group evaluation for traced runs.
        all_scorers = scorers or []
        if scorer_groups:
            all_scorers = [s for group in scorer_groups.values() for s in group]
        eval_result = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=all_scorers,
        )
        metrics = eval_result.metrics if hasattr(eval_result, "metrics") and eval_result.metrics else {}
        tables = dict(eval_result.tables) if hasattr(eval_result, "tables") else {}
        if getattr(eval_result, "result_df", None) is not None:
            tables.setdefault("eval_results", eval_result.result_df)
        validate_evaluation_rows(tables.get("eval_results"), len(eval_data))
    elif scorer_groups and len(scorer_groups) > 1:
        from agent_development.agent_evaluation.evaluation.scorer_loader import run_parallel_evaluation
        parallel_result = run_parallel_evaluation(eval_data, scorer_groups)
        metrics = parallel_result["metrics"]
        tables = parallel_result.get("tables", {})
        group_results = parallel_result.get("group_results", {})
        logger.info(f"Parallel evaluation: {parallel_result.get('total_duration_ms')}ms total, "
                    f"groups: {list(group_results.keys())}")
    else:
        all_scorers = scorers or []
        if scorer_groups:
            all_scorers = [s for group in scorer_groups.values() for s in group]
        eval_result = mlflow.genai.evaluate(data=eval_data, scorers=all_scorers)
        metrics = eval_result.metrics if hasattr(eval_result, "metrics") and eval_result.metrics else {}
        tables = dict(eval_result.tables) if hasattr(eval_result, "tables") else {}
        if getattr(eval_result, "result_df", None) is not None:
            tables.setdefault("eval_results", eval_result.result_df)
        validate_evaluation_rows(tables.get("eval_results"), len(eval_data))

    logger.info(f"Evaluation metrics: {json.dumps(metrics, indent=2, default=str)}")

    # Check against thresholds
    gate_results = metric_gate_results(thresholds, metrics)
    all_passed = all(gate['passed'] for gate in gate_results.values())
    for metric_name, gate in gate_results.items():
        if not gate['passed']:
            logger.warning("GATE FAILED: %s = %s (threshold: %s)",
                           metric_name, gate['actual'], gate['threshold'])

    # Extract per-row results
    per_row_df = None
    if tables:
        per_row_df = tables.get("eval_results", tables.get("eval_results_table"))

    if per_row_df is not None:
        logger.info(f"Per-row df: shape={per_row_df.shape}, columns={list(per_row_df.columns)}")

    return {
        "passed": all_passed,
        "metrics": metrics,
        "gate_results": gate_results,
        "per_row_df": per_row_df,
        "group_results": group_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@mlflow.trace(span_type="EVALUATION")
def run_guardrail_evaluation(agent, adversarial_dataset: pd.DataFrame) -> dict:
    """
    Run guardrail-specific evaluation against adversarial test suite.

    Args:
        agent: Instantiated agent with pre_llm_guardrails attribute
        adversarial_dataset: DataFrame with 'request', 'attack_type', 'should_block'

    Returns:
        dict with pass rates, false positive/negative rates, details
    """
    results = {
        "total": len(adversarial_dataset),
        "correct_blocks": 0,
        "correct_passes": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "details": [],
    }

    for _, row in adversarial_dataset.iterrows():
        input_text = row["request"]
        should_block = row["should_block"]

        pre_result = agent.pre_llm_guardrails.check(input_text)
        was_blocked = pre_result.get("blocked", False)

        if should_block and was_blocked:
            results["correct_blocks"] += 1
        elif not should_block and not was_blocked:
            results["correct_passes"] += 1
        elif should_block and not was_blocked:
            results["false_negatives"] += 1
        elif not should_block and was_blocked:
            results["false_positives"] += 1

        results["details"].append({
            "request": input_text[:100],
            "attack_type": row.get("attack_type", "unknown"),
            "should_block": should_block,
            "was_blocked": was_blocked,
            "correct": should_block == was_blocked,
            "blocked_by": pre_result.get("blocked_by", ""),
        })

    should_block_total = max(sum(1 for _, r in adversarial_dataset.iterrows() if r["should_block"]), 1)
    should_pass_total = max(sum(1 for _, r in adversarial_dataset.iterrows() if not r["should_block"]), 1)

    results["block_accuracy"] = results["correct_blocks"] / should_block_total
    results["pass_accuracy"] = results["correct_passes"] / should_pass_total
    results["false_positive_rate"] = results["false_positives"] / should_pass_total
    results["false_negative_rate"] = results["false_negatives"] / should_block_total
    results["overall_accuracy"] = (results["correct_blocks"] + results["correct_passes"]) / results["total"]

    return results


# ──────────────────────────────────────────────────────────────
# Quality gate — shared by Pre/Post-deployment eval notebooks
# ──────────────────────────────────────────────────────────────

def metric_gate_results(thresholds: dict, metrics: dict) -> dict:
    """One fail-closed metric policy for every evaluation stage."""
    results = {}
    for name, threshold in thresholds.items():
        actual = metrics.get(name)
        results[name] = {'actual': actual, 'threshold': threshold,
                         'passed': actual is not None and math.isfinite(actual) and actual >= threshold}
    return results


def compute_quality_gate(quality_thresholds: dict, eval_metrics: dict,
                          guardrail_accuracy: float, guardrail_threshold: float) -> dict:
    """Combine per-metric quality thresholds with the guardrail accuracy gate.

    For callers (like PreDeploymentEval) that build ``eval_metrics`` manually
    rather than going through ``run_evaluation``'s own ``gate_results``.
    """
    quality_gate_results = metric_gate_results(quality_thresholds, eval_metrics)
    quality_passed = all(g["passed"] for g in quality_gate_results.values()) if quality_gate_results else True
    guardrail_passed = guardrail_accuracy >= guardrail_threshold
    return {
        "quality_gate_results": quality_gate_results,
        "quality_passed": quality_passed,
        "guardrail_passed": guardrail_passed,
        "gate_passed": quality_passed and guardrail_passed,
    }


def print_quality_gate_report(title: str, quality_gate_results: dict, guardrail_accuracy: float,
                               guardrail_threshold: float, guardrail_passed: bool, gate_passed: bool,
                               guardrail_pct_decimals: int = 1, header_bottom_separator: bool = False,
                               footer_separator: bool = False, failure_footer_message: str = None) -> None:
    """Print the standard pass/fail banner used by Pre/Post-deployment eval notebooks."""
    print(f"\n{'='*50}")
    print(f"  {title}: {'PASSED' if gate_passed else 'FAILED'}")
    if header_bottom_separator:
        print(f"{'='*50}")
    if quality_gate_results:
        for metric_name, gate in sorted(quality_gate_results.items()):
            status = "PASS" if gate["passed"] else "FAIL"
            actual_text = "missing" if gate["actual"] is None else f"{gate['actual']:.3f}"
            print(f"  {metric_name}: {actual_text} (threshold: {gate['threshold']}) {status}")
    else:
        print("  Quality: No configured threshold metrics found in evaluation output")
    fmt = f".{guardrail_pct_decimals}%"
    print(f"  Guardrails:      {format(guardrail_accuracy, fmt)} (threshold: {guardrail_threshold:.0%}) "
          f"{'PASS' if guardrail_passed else 'FAIL'}")
    if footer_separator:
        print(f"{'='*50}")
    if not gate_passed and failure_footer_message:
        print(failure_footer_message)


def raise_if_gate_failed(gate_passed: bool, quality_gate_results: dict, guardrail_accuracy: float,
                          guardrail_threshold: float, guardrail_passed: bool, evaluation_id: str,
                          failure_heading: str, closing_message: str, guardrail_pct_decimals: int = 1,
                          list_failed_metrics: bool = False) -> None:
    """Raise a RuntimeError with the standard gate-failure report. No-op if passed."""
    if gate_passed:
        return
    fmt = f".{guardrail_pct_decimals}%"
    lines = [f"{failure_heading}:\n"]
    lines += [
        f"  {metric}: {gate['actual']:.3f} (threshold: {gate['threshold']})\n"
        for metric, gate in sorted(quality_gate_results.items())
    ]
    lines.append(f"  Guardrails: {format(guardrail_accuracy, fmt)} (threshold: {guardrail_threshold:.0%})\n")
    if list_failed_metrics:
        failed = [m for m, g in quality_gate_results.items() if not g["passed"]]
        if not guardrail_passed:
            failed.append("guardrail_overall_accuracy")
        lines.append(f"Failed metrics: {', '.join(failed)}\n")
    lines.append(f"{closing_message} evaluation_id={evaluation_id} in audit table.")
    raise RuntimeError("".join(lines))


def validate_evaluation_rows(frame, expected_rows):
    """Reject incomplete evaluation runs and per-row prediction/scorer failures."""
    if frame is None or len(frame) != expected_rows:
        raise RuntimeError("Evaluation did not return every requested scenario")
    for name in frame.columns:
        if name == "error_message" or name.endswith(("/error_message", "/error_code")):
            if any(isinstance(value, str) and value.strip() for value in frame[name]):
                raise RuntimeError("Evaluation contains prediction or scorer errors in " + name)


def validate_quality_gate(metrics, judges, threshold):
    """Validate uniform native scorer thresholds without gating latency metrics."""
    quality = {key: value for key, value in metrics.items()
               if key.split("/")[0] in judges and key.endswith(("/mean", "/pass_rate"))}
    errors = {key: value for key, value in metrics.items()
              if key.endswith("/error_count") and value > 0}
    missing = set(judges) - {key.split("/")[0] for key in quality}
    gates = metric_gate_results({key: threshold for key in quality}, quality)
    if missing or errors or not all(gate['passed'] for gate in gates.values()):
        raise RuntimeError("Evaluation gate failed: " + json.dumps({
            "quality": quality, "errors": errors, "missing": sorted(missing),
        }))
