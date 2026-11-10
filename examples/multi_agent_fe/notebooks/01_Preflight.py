# Databricks notebook source
# Serverless notebook: install the resolved dependency lock before running.
import json
import sys
from pathlib import Path

dbutils.widgets.text("project_root", "")
dbutils.widgets.text("runtime_file", "")
root = Path(dbutils.widgets.get("project_root"))
sys.path.insert(0, str(root))
from databricks.sdk import WorkspaceClient
from fe_multi_agent.config import Settings
from fe_multi_agent.preflight import check

settings = Settings(**json.loads(Path(dbutils.widgets.get("runtime_file")).read_text()))
report = check(settings, WorkspaceClient())
print(json.dumps(report, indent=2))
if not report["ready"]:
    raise RuntimeError("Requested package pins differ from installed versions; review before proceeding")
