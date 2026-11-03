"""
AgentOPS Framework — Post-Deployment Trace Monitoring
Collects operational metrics, guardrail stats, and drift from production traces.
"""

import mlflow
from databricks.sdk import WorkspaceClient
import logging
import inspect
import json
import ast
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class TraceMonitor:
    """Monitors deployed agent traces for quality, drift, and guardrail performance."""

    def __init__(self, agent_name: str, environment: str, catalog: str = None,
                 schema: str = None, endpoint_name: str = None, spark=None):
        self.agent_name = agent_name
        self.environment = environment
        self.catalog = catalog
        self.schema = schema
        self.endpoint_name = endpoint_name
        self.spark = spark
        self.w = WorkspaceClient()

    def _get_warehouse_id(self) -> str:
        warehouses = list(self.w.warehouses.list())
        return warehouses[0].id if warehouses else None

    def collect_operational_metrics(self, hours: int = 1) -> dict:
        """Collect latency, error rates, token usage from inference table."""
        metrics = {}

        if not self.endpoint_name or not self.catalog or not self.schema:
            # Endpoint table context missing; still try MLflow trace tags for usage/cost.
            return self.collect_token_cost_metrics_from_traces(hours=hours)

        try:
            # Inference table is auto-created by model serving auto-capture
            payload_table = f"{self.catalog}.{self.schema}.`{self.endpoint_name}_payload`"
            query = f"""
                SELECT
                    count(*) as total_requests,
                    round(avg(response_duration_ms), 2) as avg_latency_ms,
                    round(percentile(response_duration_ms, 0.50), 2) as p50_latency_ms,
                    round(percentile(response_duration_ms, 0.95), 2) as p95_latency_ms,
                    round(percentile(response_duration_ms, 0.99), 2) as p99_latency_ms,
                    sum(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) as error_count,
                    round(avg(total_tokens_count), 0) as avg_tokens_per_request
                FROM {payload_table}
                WHERE timestamp_ms > unix_timestamp(current_timestamp() - INTERVAL {hours} HOURS) * 1000
            """

            if self.spark is not None:
                row = self.spark.sql(query).first()
                if row is not None:
                    metrics = row.asDict(recursive=True)
            else:
                wh_id = self._get_warehouse_id()
                result = self.w.statement_execution.execute_statement(
                    warehouse_id=wh_id,
                    statement=query,
                )
                if result and result.result and result.result.data_array:
                    columns = [col.name for col in result.manifest.schema.columns]
                    row = result.result.data_array[0]
                    metrics = dict(zip(columns, row))
        except Exception as e:
            logger.warning(f"Failed to collect operational metrics: {e}")

        # Merge MLflow trace-tag usage/cost metrics to avoid schema coupling to payload tables.
        tag_metrics = self.collect_token_cost_metrics_from_traces(hours=hours)
        if tag_metrics:
            metrics.update(tag_metrics)
            # Backfill average tokens if table-based field is absent.
            if "avg_tokens_per_request" not in metrics and tag_metrics.get("trace_avg_tokens_per_request") is not None:
                metrics["avg_tokens_per_request"] = tag_metrics["trace_avg_tokens_per_request"]

        return metrics

    def collect_token_cost_metrics_from_traces(self, hours: int = 1, max_results: int = 2000) -> dict:
        """Aggregate token/cost metrics from MLflow trace tags.

        Reads framework-emitted tag keys:
        - agentops.tokens.prompt
        - agentops.tokens.completion
        - agentops.tokens.total
        - agentops.tokens.reasoning
        - agentops.tokens.cached
        - agentops.cost_usd
        - agentops.llm_requests
        """
        try:
            cutoff = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp() * 1000)
            traces = self._search_traces_compat(cutoff=cutoff, max_results=max_results)

            total_traces = 0
            traces_with_usage = 0
            sum_prompt = 0
            sum_completion = 0
            sum_total = 0
            sum_reasoning = 0
            sum_cached = 0
            sum_cost = 0.0
            sum_llm_requests = 0

            for trace in self._iter_traces(traces):
                total_traces += 1
                tags = self._extract_tags(trace)

                prompt = self._to_int(tags.get("agentops.tokens.prompt"))
                completion = self._to_int(tags.get("agentops.tokens.completion"))
                total = self._to_int(tags.get("agentops.tokens.total"))
                reasoning = self._to_int(tags.get("agentops.tokens.reasoning"))
                cached = self._to_int(tags.get("agentops.tokens.cached"))
                cost = self._to_float(tags.get("agentops.cost_usd"))
                llm_requests = self._to_int(tags.get("agentops.llm_requests"), default=1)

                has_usage = any([prompt > 0, completion > 0, total > 0, reasoning > 0, cached > 0, cost > 0])
                if not has_usage:
                    continue

                traces_with_usage += 1
                sum_prompt += prompt
                sum_completion += completion
                sum_total += total
                sum_reasoning += reasoning
                sum_cached += cached
                sum_cost += cost
                sum_llm_requests += max(llm_requests, 0)

            if traces_with_usage == 0:
                return {
                    "trace_total_requests": total_traces,
                    "trace_requests_with_usage": 0,
                }

            avg_tokens = round(sum_total / traces_with_usage, 2)
            avg_cost = round(sum_cost / traces_with_usage, 6)

            return {
                "trace_total_requests": total_traces,
                "trace_requests_with_usage": traces_with_usage,
                "trace_prompt_tokens": sum_prompt,
                "trace_completion_tokens": sum_completion,
                "trace_total_tokens": sum_total,
                "trace_reasoning_tokens": sum_reasoning,
                "trace_cached_tokens": sum_cached,
                "trace_total_cost_usd": round(sum_cost, 6),
                "trace_llm_requests": sum_llm_requests,
                "trace_avg_tokens_per_request": avg_tokens,
                "trace_avg_cost_per_request_usd": avg_cost,
            }
        except Exception as e:
            logger.warning(f"Failed to collect token/cost metrics from trace tags: {e}")
            return {}

    def collect_guardrail_metrics(self, hours: int = 24) -> dict:
        """Collect guardrail block rates from MLflow traces."""
        try:
            cutoff = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp() * 1000)
            traces = self._search_traces_compat(cutoff=cutoff, max_results=1000)

            metrics = {
                "total_requests": len(traces) if traces is not None else 0,
                "pre_llm_blocks": 0,
                "post_llm_blocks": 0,
                "block_reasons": {},
            }

            if traces is not None:
                for trace in self._iter_traces(traces):
                    tags = self._extract_tags(trace)
                    if tags.get("agentops.guardrail.pre_llm.blocked") == "true":
                        metrics["pre_llm_blocks"] += 1
                        reason = tags.get("agentops.guardrail.pre_llm.blocked_by", "unknown")
                        metrics["block_reasons"][reason] = metrics["block_reasons"].get(reason, 0) + 1
                    if tags.get("agentops.guardrail.post_llm.blocked") == "true":
                        metrics["post_llm_blocks"] += 1

            if metrics["total_requests"] > 0:
                metrics["pre_llm_block_rate"] = metrics["pre_llm_blocks"] / metrics["total_requests"]
                metrics["post_llm_block_rate"] = metrics["post_llm_blocks"] / metrics["total_requests"]

            return metrics
        except Exception as e:
            logger.warning(f"Failed to collect guardrail metrics: {e}")
            return {}

    def _search_traces_compat(self, cutoff: int, max_results: int = 1000):
        """Search traces across MLflow versions with differing argument signatures."""
        sig = inspect.signature(mlflow.search_traces)
        params = sig.parameters

        kwargs = {
            "filter_string": f"timestamp > {cutoff}",
            "max_results": max_results,
        }

        exp_name = f"/agentops/{self.agent_name}"

        if "experiment_names" in params:
            kwargs["experiment_names"] = [exp_name]
            return mlflow.search_traces(**kwargs)

        if "experiment_ids" in params:
            exp_id = self._resolve_experiment_id(exp_name)
            if exp_id:
                kwargs["experiment_ids"] = [exp_id]
            return mlflow.search_traces(**kwargs)

        # Oldest fallback: call without experiment scoping if unsupported.
        return mlflow.search_traces(**kwargs)

    @staticmethod
    def _iter_traces(traces):
        """Iterate traces regardless of MLflow return type (DataFrame/list)."""
        if hasattr(traces, "itertuples"):
            return traces.itertuples()
        if isinstance(traces, list):
            return traces
        return []

    @staticmethod
    def _extract_tags(trace) -> dict:
        """Extract tag dict from trace row/object across return shapes."""
        tags = trace.get("tags", {}) if isinstance(trace, dict) else getattr(trace, "tags", {})
        if isinstance(tags, dict):
            return tags
        if isinstance(tags, str):
            # Some MLflow return shapes serialize tags as JSON/py-literal strings.
            try:
                parsed = json.loads(tags)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(tags)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return {}

    @staticmethod
    def _to_int(value, default: int = 0) -> int:
        try:
            if value is None or value == "":
                return default
            return int(float(value))
        except Exception:
            return default

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _resolve_experiment_id(exp_name: str) -> str:
        """Resolve experiment id for a name if available."""
        try:
            exp = mlflow.get_experiment_by_name(exp_name)
            return exp.experiment_id if exp else ""
        except Exception:
            return ""

    def check_quality_drift(self, window_days: int = 7) -> dict:
        """
        Compare recent trace quality vs baseline to detect drift.
        Checks: latency drift, token usage drift, error rate changes.
        """
        if not self.endpoint_name or not self.catalog or not self.schema:
            return {}

        try:
            payload_table = f"{self.catalog}.{self.schema}.`{self.endpoint_name}_payload`"
            query = f"""
                WITH recent AS (
                    SELECT avg(response_duration_ms) as avg_latency,
                           avg(total_tokens_count) as avg_tokens
                    FROM {payload_table}
                    WHERE timestamp_ms > unix_timestamp(current_timestamp() - INTERVAL 1 DAY) * 1000
                ),
                baseline AS (
                    SELECT avg(response_duration_ms) as avg_latency,
                           avg(total_tokens_count) as avg_tokens
                    FROM {payload_table}
                    WHERE timestamp_ms > unix_timestamp(current_timestamp() - INTERVAL {window_days} DAYS) * 1000
                      AND timestamp_ms <= unix_timestamp(current_timestamp() - INTERVAL 1 DAY) * 1000
                )
                SELECT
                    recent.avg_latency as recent_latency,
                    baseline.avg_latency as baseline_latency,
                    abs(recent.avg_latency - baseline.avg_latency) / nullif(baseline.avg_latency, 0) as latency_drift_pct,
                    recent.avg_tokens as recent_tokens,
                    baseline.avg_tokens as baseline_tokens,
                    abs(recent.avg_tokens - baseline.avg_tokens) / nullif(baseline.avg_tokens, 0) as token_drift_pct
                FROM recent, baseline
            """

            if self.spark is not None:
                row = self.spark.sql(query).first()
                if row is not None:
                    drift = row.asDict(recursive=True)
                else:
                    drift = {}
            else:
                wh_id = self._get_warehouse_id()
                result = self.w.statement_execution.execute_statement(
                    warehouse_id=wh_id,
                    statement=query,
                )
                if result and result.result and result.result.data_array:
                    columns = [col.name for col in result.manifest.schema.columns]
                    row = result.result.data_array[0]
                    drift = dict(zip(columns, row))
                else:
                    drift = {}

            if drift:
                drift["latency_alert"] = float(drift.get("latency_drift_pct", 0) or 0) > 0.50
                drift["token_alert"] = float(drift.get("token_drift_pct", 0) or 0) > 0.25
                return drift
        except Exception as e:
            logger.warning(f"Failed to check drift: {e}")
        return {}

    def run_monitoring_cycle(self) -> dict:
        """Run a complete monitoring cycle."""
        logger.info(f"Monitoring: {self.agent_name} in {self.environment}")
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_name": self.agent_name,
            "environment": self.environment,
            "operational_metrics": self.collect_operational_metrics(hours=1),
            "guardrail_metrics": self.collect_guardrail_metrics(hours=24),
            "drift_check": self.check_quality_drift(window_days=7),
        }
