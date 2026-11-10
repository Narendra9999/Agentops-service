"""Read a serverless deployment run and preserve completed notebook evidence."""
import argparse
import json
import re
from pathlib import Path


def main():
    from databricks.sdk import WorkspaceClient
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',required=True)
    parser.add_argument('--run-id',required=True,type=int)
    args=parser.parse_args()
    workspace=WorkspaceClient(profile=args.profile)
    run=workspace.jobs.get_run(args.run_id)
    root=Path(__file__).resolve().parents[1]
    output=root/'validation-results'
    output.mkdir(exist_ok=True)
    print(json.dumps({'run_id':run.run_id,'state':run.state.as_dict(),'url':run.run_page_url}))
    for task in run.tasks or []:
        print(json.dumps({'task_id':task.run_id,'state':task.state.as_dict()}))
        if not task.state.result_state:
            continue
        result=workspace.jobs.get_run_output(task.run_id)
        if result.error:
            text=re.sub(r'\x1b\[[0-9;]*m','',result.error_trace or result.error)
            (output/f'run-{task.run_id}-error.txt').write_text(text)
            print(json.dumps({'error':result.error,'trace_tail':text[-2500:]}))
        elif result.notebook_output and result.notebook_output.result:
            data=json.loads(result.notebook_output.result)
            (output/f'run-{task.run_id}.json').write_text(json.dumps(data,indent=2))
            if 'lock' in data:
                (root/'requirements.lock').write_text(data['lock'])
                print('Saved requirements.lock from serverless dependency closure')
            if 'paired_evaluation' in data:
                print(json.dumps({'evaluation':data['paired_evaluation'],'prompt_registry':data.get('prompt_registry_status')}))
            else:
                print(json.dumps({k:v for k,v in data.items() if k!='lock'}))


if __name__=='__main__':main()
