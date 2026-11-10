# Databricks notebook source
# Runs only after preflight and paired evaluation review. A unique endpoint is created.
import json
import sys
from pathlib import Path

for key in ("project_root", "runtime_file", "model_name", "endpoint_name", "serving_environment_file",
            "catalog", "schema", "reviewers"):
    dbutils.widgets.text(key, "")
root = Path(dbutils.widgets.get("project_root"))
sys.path.insert(0, str(root))
from databricks.sdk import WorkspaceClient
from fe_multi_agent.config import Settings
from fe_multi_agent.deployment import register_model, deploy_endpoint
from fe_multi_agent.operations import configure_gateway, setup_review

settings = Settings(**json.loads(Path(dbutils.widgets.get("runtime_file")).read_text()))
environment = json.loads(Path(dbutils.widgets.get("serving_environment_file")).read_text())
# This file contains OAuth secret REFERENCES only, never literal credentials.
workspace = WorkspaceClient()
registered = register_model(settings, dbutils.widgets.get("model_name"), root,
                            root / "runtime.json", root / "requirements.lock")
endpoint = dbutils.widgets.get("endpoint_name")
deploy_endpoint(workspace, endpoint, dbutils.widgets.get("model_name"), registered["version"], environment)
gateway = configure_gateway(workspace, endpoint, dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
review = setup_review(settings.experiment_id, endpoint,
                      [x.strip() for x in dbutils.widgets.get("reviewers").split(",") if x.strip()])
dbutils.notebook.exit(json.dumps({"registered": registered, "endpoint": endpoint, "review": review,
                                 "gateway_configured": True}))
