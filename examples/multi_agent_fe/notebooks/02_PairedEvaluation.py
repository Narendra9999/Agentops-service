# Databricks notebook source
import json
import sys
from pathlib import Path
import mlflow

for key in ("project_root", "runtime_file", "before_manifest", "after_manifest", "judge_model", "output_dir"):
    dbutils.widgets.text(key, "")
sys.path.insert(0, dbutils.widgets.get("project_root"))
from fe_multi_agent.config import Settings
from fe_multi_agent.evaluation import run_pair

settings = Settings(**json.loads(Path(dbutils.widgets.get("runtime_file")).read_text()))
mlflow.set_experiment(experiment_id=settings.experiment_id)
mlflow.langchain.autolog()
report = run_pair(settings, dbutils.widgets.get("before_manifest"), dbutils.widgets.get("after_manifest"),
                  dbutils.widgets.get("judge_model"), dbutils.widgets.get("output_dir"))
dbutils.notebook.exit(json.dumps(report, default=str))
