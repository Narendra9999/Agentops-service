"""Explicit model registration/deployment. Does not change the AgentOPS root bundle."""
from dataclasses import asdict
import json
from pathlib import Path


def register_model(settings, model_name, project_root, runtime_file, dependency_lock):
    import mlflow
    from mlflow.models.resources import DatabricksServingEndpoint, DatabricksGenieSpace, DatabricksVectorSearchIndex
    root = Path(project_root).resolve()
    lock = Path(dependency_lock)
    if not lock.is_file():
        raise ValueError("Resolve and validate requirements.lock before model registration")
    settings.validate()
    Path(runtime_file).write_text(json.dumps(asdict(settings), indent=2))
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment_id=settings.experiment_id)
    requirements = [line for line in lock.read_text().splitlines()
                    if line and not line.startswith(("#", " ", "--"))]
    # Resource declarations reference existing workers; no worker is registered.
    resources = ([DatabricksVectorSearchIndex(index_name=settings.docs_index)] if settings.platform_mode=='in_process' else [DatabricksServingEndpoint(endpoint_name=settings.platform_endpoint)])
    if settings.cost_endpoint:
        resources.append(DatabricksServingEndpoint(endpoint_name=settings.cost_endpoint))
    else:
        resources.append(DatabricksGenieSpace(genie_space_id=settings.genie_space_id))
    if settings.model_transport == "model_serving":
        resources.append(DatabricksServingEndpoint(endpoint_name=settings.supervisor_model))
        if settings.platform_mode=='in_process':resources.append(DatabricksServingEndpoint(endpoint_name=settings.platform_model))
    # Model services are UC securables, not serving-endpoint resources. Gateway
    # EXECUTE grants are provisioned explicitly for the dedicated OAuth identity.
    package = Path(__file__).resolve().parent
    code_paths = [str(package)]
    # Usage collection is independent of optional observation spans.
    framework = root / "framework"
    if not framework.is_dir():
        framework = root.parent.parent / "src/framework"
    if not framework.is_dir():
        raise ValueError("Usage instrumentation requires the framework source")
    code_paths.append(str(framework))
    example={"input":[{"role":"user","content":"Explain MLflow tracing with a documentation citation."}]}
    if settings.memory_enabled and settings.unsigned_context_policy != 'stateless':
        example = None  # Never publish expiring signed context as a reusable example.
    with mlflow.start_run(run_name="register-platform-agent-team"), mlflow.start_span(name="model.registration"):
        mlflow.update_current_trace(tags={"fe.eval_phase": "packaging"})
        info = mlflow.pyfunc.log_model(
            name="agent", python_model=str(root / "model.py"), code_paths=code_paths,
            artifacts={"runtime": str(runtime_file), "prompts": settings.prompt_manifest},
            pip_requirements=requirements, resources=resources,
            input_example=example,
            registered_model_name=model_name,
        )
    return {"model_uri": info.model_uri, "version": str(info.registered_model_version)}


def deploy_endpoint(workspace, endpoint_name, model_name, version, serving_environment):
    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
    serving_environment = {"ENABLE_MLFLOW_TRACING": "true",
                           "MLFLOW_ENABLE_TRACE_IN_SERVING": "true", **serving_environment}
    # Autoscaling Lakebase auth passthrough must be verified for the selected FEVM.
    # Until verified, use a dedicated least-privilege OAuth SP via secret references.
    required = {"DATABRICKS_HOST", "DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET"}
    if not required.issubset(serving_environment):
        raise ValueError("Supply a dedicated serving OAuth identity; worker access must be provisioned separately")
    secret = serving_environment["DATABRICKS_CLIENT_SECRET"]
    if not secret.startswith("{{secrets/") or not secret.endswith("}}"):
        raise ValueError("Use a Databricks secret reference, never a literal OAuth secret")
    entity = ServedEntityInput(entity_name=model_name, entity_version=str(version), workload_size="Small",
                               scale_to_zero_enabled=True, environment_vars=serving_environment)
    # Unique demo endpoint creation; existing endpoints are never overwritten automatically.
    return workspace.serving_endpoints.create_and_wait(
        name=endpoint_name, config=EndpointCoreConfigInput(name=endpoint_name, served_entities=[entity]))
