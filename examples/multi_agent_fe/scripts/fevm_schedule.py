"""Create a paused monitor job; never update or enable an existing schedule implicitly."""
import argparse,json
from pathlib import Path


def main():
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.jobs import Task,NotebookTask,JobEnvironment,CronSchedule,PauseStatus
    from databricks.sdk.service.compute import Environment
    parser=argparse.ArgumentParser();parser.add_argument('--profile',required=True)
    args=parser.parse_args();w=WorkspaceClient(profile=args.profile)
    name='fe-agentops-multi-monitor-naren'
    matches=list(w.jobs.list(name=name))
    if len(matches)>1:raise ValueError('Multiple matching jobs; choose explicitly')
    if matches:
        job_id=matches[0].job_id
    else:
        root=Path(__file__).resolve().parents[1]
        dependencies=[line for line in (root/'requirements.lock').read_text().splitlines() if line and not line.startswith('#')]
        job_id=w.jobs.create(name=name,max_concurrent_runs=1,
            description='Isolated multi-agent trace monitor; bind this ID to daily_monitor before a full bundle deployment.',
            schedule=CronSchedule(quartz_cron_expression='0 0 7 * * ?',timezone_id='UTC',pause_status=PauseStatus.PAUSED),
            tasks=[Task(task_key='monitor',notebook_task=NotebookTask(notebook_path='/Users/'+w.current_user.me().user_name+'/fe-platform-agent-team/10_FEVMMonitor'),
                        environment_key='monitor',max_retries=0,timeout_seconds=3600)],
            environments=[JobEnvironment(environment_key='monitor',spec=Environment(client='4',dependencies=dependencies))]).job_id
    result={'job_id':job_id,'name':name,'schedule':'PAUSED','bundle_binding_required':'daily_monitor'}
    output=Path(__file__).resolve().parents[1]/'validation-results/monitor-job.json'
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
