"""Upload a source snapshot and run candidate validation on serverless FEVM compute."""
import argparse
import base64
import io
import json
from pathlib import Path
import zipfile


def main():
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.workspace import ImportFormat, Language
    from databricks.sdk.service.jobs import SubmitTask, NotebookTask, JobEnvironment
    from databricks.sdk.service.compute import Environment
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    parser.add_argument('--upload-only', action='store_true')
    parser.add_argument('--notebook', type=Path, help='Optional integration notebook to run from the source snapshot')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    w = WorkspaceClient(profile=args.profile)
    user = w.current_user.me().user_name
    remote = f'/Users/{user}/fe-platform-agent-team'
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in root.rglob('*'):
            if p.is_file() and not any(x in p.parts for x in ['.venv','.databricks','__pycache__','.pytest_cache','validation-results','fe_multi_agent']):
                if p.name not in {'runtime.json','prompt-manifest.json'}:
                    z.writestr(str(p.relative_to(root)), p.read_bytes())
        source = root.parents[1] / 'src/teams/platform-intelligence/agent/fe_multi_agent'
        for p in source.rglob('*.py'):
            z.writestr('fe_multi_agent/' + str(p.relative_to(source)), p.read_bytes())
        framework = root.parents[1] / 'src/framework'
        for p in framework.rglob('*.py'):
            z.writestr('framework/' + str(p.relative_to(framework)), p.read_bytes())
        team = root.parents[1] / 'src/teams/platform-intelligence'
        for relative in ['agent/Agent.py','config.yaml','configs/prompts.json']:
            z.writestr('team/' + relative, (team / relative).read_bytes())
    payload = base64.b64encode(archive.getvalue()).decode()
    notebook = '''# Databricks notebook source
import base64,zipfile,io,pathlib,sys,json,importlib.metadata as metadata,subprocess
root=pathlib.Path('/tmp/fe-platform-agent-team')
root.mkdir(exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall(root)
sys.path.insert(0,str(root))
packages=['langgraph','langgraph-prebuilt','langgraph-sdk','langchain-core','databricks-langchain','mlflow','fastapi']
report={'packages':{p:metadata.version(p) for p in packages}}
from langgraph.prebuilt import create_react_agent
from databricks_langchain import ChatDatabricks
from mlflow.pyfunc import ResponsesAgent
from fe_multi_agent.graph import build_supervisor_graph
result=subprocess.run([sys.executable,'-m','pytest',str(root/'tests/test_graph.py'),'-q'],capture_output=True,text=True,env={**__import__('os').environ,'PYTHONPATH':str(root)})
report['graph_tests']={'returncode':result.returncode,'stdout':result.stdout[-6000:],'stderr':result.stderr[-2000:]}
if result.returncode: raise RuntimeError(json.dumps(report))
from databricks.sdk import WorkspaceClient
w=WorkspaceClient()
report['model_imports']='passed'
dbutils.notebook.exit(json.dumps(report,default=str))
'''.replace('PAYLOAD',repr(payload))
    if args.notebook:
        prefix = notebook.split("packages=['langgraph'", 1)[0]
        notebook = prefix + args.notebook.read_text()
    w.workspace.mkdirs(remote)
    path=remote+'/' + (args.notebook.stem if args.notebook else '00_ServerlessBootstrap')
    w.workspace.import_(path=path,format=ImportFormat.SOURCE,language=Language.PYTHON,
                        content=base64.b64encode(notebook.encode()).decode(),overwrite=True)
    if args.upload_only:
        print(json.dumps({'notebook':path,'workspace':w.config.host}))
        return
    dependency_file = root / 'requirements.lock'
    if not dependency_file.is_file():
        dependency_file = root / 'requirements.in'
    dependencies=[line for line in dependency_file.read_text().splitlines() if line and not line.startswith('#')]
    run=w.jobs.submit(run_name='fe-platform-team-candidate-validation',
        tasks=[SubmitTask(task_key='validate',notebook_task=NotebookTask(notebook_path=path),
                          environment_key='candidate',timeout_seconds=1800,max_retries=0)],
        environments=[JobEnvironment(environment_key='candidate',spec=Environment(client='4',dependencies=dependencies))],
        timeout_seconds=1800).response
    print(json.dumps({'run_id':run.run_id,'notebook':path,'workspace':w.config.host}))


if __name__ == '__main__':
    main()
