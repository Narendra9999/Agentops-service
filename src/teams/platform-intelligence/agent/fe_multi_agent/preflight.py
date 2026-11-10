"""Read-only readiness checks; report unsupported capabilities before provisioning."""
import importlib.metadata as metadata
import inspect

REQUESTED = {"langgraph": "0.6.11", "langgraph-prebuilt": "0.6.5",
             "langgraph-sdk": "0.2.12", "langchain-core": "0.3.81",
             "databricks-langchain": "0.8.0", "mlflow": "3.11.1", "fastapi": "0.123.0"}


def check(settings, workspace):
    settings.validate()
    report = {"packages": {}, "resources": {}, "ready": True}
    for name, version in REQUESTED.items():
        installed = metadata.version(name)
        report["packages"][name] = {"requested": version, "installed": installed}
        if installed != version:
            report["ready"] = False
    # Verify APIs against the installed pins, not against latest online examples.
    from langgraph.prebuilt import create_react_agent
    if "prompt" not in inspect.signature(create_react_agent).parameters:
        raise RuntimeError("Installed create_react_agent lacks prompt; validate the native worker adapter")
    for endpoint in [settings.platform_endpoint] + ([settings.cost_endpoint] if settings.cost_endpoint else []):
        obj = workspace.serving_endpoints.get(endpoint)
        report["resources"][endpoint] = obj.state.as_dict()
    if not settings.cost_endpoint:
        space = workspace.genie.get_space(settings.genie_space_id)
        report["resources"]["genie"] = {"space_id": settings.genie_space_id, "title": space.title}
    report["supervisor_inference_access"] = "Validate by invoking the supervisor with the serving identity"
    return report
