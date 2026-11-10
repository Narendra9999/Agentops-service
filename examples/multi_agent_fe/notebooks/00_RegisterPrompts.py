# Databricks notebook source
import json
import sys
from pathlib import Path
import mlflow

for key in ("project_root", "prompt_prefix", "experiment_id", "output_dir"):
    dbutils.widgets.text(key, "")
sys.path.insert(0, dbutils.widgets.get("project_root"))
from fe_multi_agent.prompts import register_prompts, TEMPLATES

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_id=dbutils.widgets.get("experiment_id"))
prefix = dbutils.widgets.get("prompt_prefix")
output = Path(dbutils.widgets.get("output_dir"))
output.mkdir(parents=True, exist_ok=True)
baseline = dict(TEMPLATES)
baseline["supervisor"] = "Coordinate cost_estimation and platform_docs. Delegate the task and summarize their results."
before = register_prompts(prefix, baseline)
after = register_prompts(prefix, TEMPLATES)
(output / "before-prompts.json").write_text(json.dumps(before, indent=2))
(output / "after-prompts.json").write_text(json.dumps(after, indent=2))
dbutils.notebook.exit(json.dumps({"before": before, "after": after}))
