# Databricks notebook source
import json
import sys
for key in ("project_root", "experiment_id", "judge_model"):
    dbutils.widgets.text(key, "")
sys.path.insert(0, dbutils.widgets.get("project_root"))
from fe_multi_agent.operations import daily_monitor
report = daily_monitor(dbutils.widgets.get("experiment_id"), dbutils.widgets.get("judge_model"))
dbutils.notebook.exit(json.dumps(report, default=str))
