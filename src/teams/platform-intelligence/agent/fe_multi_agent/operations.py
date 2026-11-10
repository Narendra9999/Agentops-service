"""Deployment operations, invoked deliberately after configuration/preflight."""
import json
from pathlib import Path


def configure_gateway(workspace, endpoint, catalog, schema):
    from databricks.sdk.service.serving import (
        AiGatewayGuardrails, AiGatewayGuardrailParameters, AiGatewayGuardrailPiiBehavior,
        AiGatewayGuardrailPiiBehaviorBehavior, AiGatewayInferenceTableConfig, AiGatewayRateLimit,
        AiGatewayRateLimitKey, AiGatewayRateLimitRenewalPeriod, AiGatewayUsageTrackingConfig,
    )
    guard = AiGatewayGuardrailParameters(pii=AiGatewayGuardrailPiiBehavior(
        behavior=AiGatewayGuardrailPiiBehaviorBehavior.BLOCK))
    workspace.serving_endpoints.put_ai_gateway(
        name=endpoint, guardrails=AiGatewayGuardrails(input=guard, output=guard),
        rate_limits=[AiGatewayRateLimit(calls=60, key=AiGatewayRateLimitKey.ENDPOINT,
                                       renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE)],
        inference_table_config=AiGatewayInferenceTableConfig(enabled=True, catalog_name=catalog,
                                                            schema_name=schema, table_name_prefix="fe_multi"),
        usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
    )
    # Verify effective config. Unsupported endpoint/region/schema must fail visibly.
    effective = workspace.serving_endpoints.get(endpoint).ai_gateway
    if not effective or not effective.guardrails or not effective.inference_table_config.enabled:
        raise RuntimeError("Gateway settings were not persisted")
    return effective.as_dict()


def configure_inference_logging(workspace, endpoint, catalog, schema):
    """Explicit logging-only setup for agent endpoints lacking Gateway controls.

    This does not enforce PII guardrails or rate limits. Keep the full requested
    configuration above strict so callers cannot mistake logging for enforcement.
    """
    from databricks.sdk.service.serving import AiGatewayInferenceTableConfig
    workspace.serving_endpoints.put_ai_gateway(
        name=endpoint, inference_table_config=AiGatewayInferenceTableConfig(
            enabled=True, catalog_name=catalog, schema_name=schema,
            table_name_prefix="fe_multi"))
    effective = workspace.serving_endpoints.get(endpoint).ai_gateway
    if not effective or not effective.inference_table_config or not effective.inference_table_config.enabled:
        raise RuntimeError("Inference logging was not enabled")
    return effective.as_dict()


def setup_review(experiment_id, endpoint, assigned_users):
    import mlflow
    from mlflow.genai import get_review_app, create_labeling_session, label_schemas
    mlflow.set_experiment(experiment_id=experiment_id)
    label_schemas.create_label_schema(name="team_quality", type="feedback", title="Agent team quality",
        input=label_schemas.InputNumeric(min_value=1, max_value=5), enable_comment=True,
        instruction="Assess routing, source-supported platform answers, and grounded cost estimates.", overwrite=True)
    review = get_review_app(experiment_id=experiment_id).add_agent(
        agent_name="platform_engineering", model_serving_endpoint=endpoint, overwrite=True)
    session = create_labeling_session(name="platform-engineering-review", assigned_users=assigned_users,
                                     label_schemas=["team_quality"])
    traces = mlflow.search_traces(experiment_ids=[experiment_id], return_type="list", max_results=100)
    if traces:
        session = session.add_traces(traces=traces)
    return {"labeling_url": session.url}


def daily_monitor(experiment_id, judge_model):
    """Score captured production traces; never pass predict_fn or call the application."""
    import mlflow
    import pandas as pd
    import hashlib
    from datetime import datetime, timedelta, timezone
    from mlflow.genai.scorers import Safety, RelevanceToQuery
    mlflow.set_experiment(experiment_id=experiment_id)
    cutoff = int((datetime.now(timezone.utc) - timedelta(days=1)).timestamp() * 1000)
    traces = mlflow.search_traces(experiment_ids=[experiment_id], return_type="list",
        filter_string=f"timestamp_ms > {cutoff} AND tags.`agentops.system_id` = 'fe-platform-engineering'",
        max_results=1000)
    marker = "fe.monitor." + hashlib.sha256((judge_model + ":v1").encode()).hexdigest()[:12]
    selected = [t for t in traces if not t.info.tags.get("fe.eval_phase")
                and not t.info.tags.get("mlflow.eval.requestId")
                and t.info.tags.get(marker) != "complete"]
    if not selected:
        return {"status": "no_data"}
    # Bounded demo batch; report the cap rather than claiming exhaustive coverage.
    model = f"databricks:/{judge_model}"
    with mlflow.start_run(run_name="daily-platform-agent-monitor"):
        mlflow.log_params({"window_hours": 24, "trace_cap": 1000, "selected_count": len(selected)})
        result = mlflow.genai.evaluate(data=pd.DataFrame({"trace": selected}),
                                      scorers=[Safety(model=model), RelevanceToQuery(model=model)])
        # Only mark completed batch evaluations. Single-concurrency daily job avoids races;
        # a crash between scoring and markers can still duplicate assessments (documented).
        for trace in selected:
            mlflow.set_trace_tag(trace_id=trace.info.trace_id, key=marker, value="complete")
        return {"status": "scored", "run_id": result.run_id, "metrics": result.metrics,
                "cap_reached": len(traces) == 1000}
