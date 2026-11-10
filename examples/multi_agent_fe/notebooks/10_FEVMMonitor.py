# Databricks notebook source
from fe_multi_agent.operations import daily_monitor
report=daily_monitor('4384417075366092','databricks-gpt-oss-120b')
dbutils.notebook.exit(json.dumps(report,default=str))
