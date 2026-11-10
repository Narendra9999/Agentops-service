"""Read deployed smoke trace summaries without downloading conversation payloads."""
import argparse
import json
from pathlib import Path
from databricks.sdk import WorkspaceClient


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',required=True)
    args=parser.parse_args()
    w=WorkspaceClient(profile=args.profile)
    root=Path(__file__).resolve().parents[1]/'validation-results'
    report={}
    for source in ['memory-serving-smoke.json','memory-app-smoke.json']:
        data=json.loads((root/source).read_text())
        for name,scenario in data['scenarios'].items():
            trace_id=scenario['trace_id']
            info=w.api_client.do('GET','/api/3.0/mlflow/traces/'+trace_id)['trace']['trace_info']
            assert info['state']=='OK',(name,info['state'])
            metadata=info.get('trace_metadata',{})
            usage=json.loads(metadata.get('mlflow.trace.tokenUsage','{}'))
            assert usage and usage.get('total_tokens',0)>0,(name,usage)
            signals={k:v for k,v in {**metadata,**info.get('tags',{})}.items() if 'cost' in k.lower() or 'tokenusage' in k.lower()}
            report[source+':'+name]={'trace_id':trace_id,'state':info['state'],'usage':usage,'cost_and_usage_metadata':signals,
                'workers':sorted({e['agent'] for e in scenario.get('activity',[]) if e.get('kind')=='agent_started'}),
                'memory_tools':sorted({e['tool'] for e in scenario.get('activity',[]) if 'memory' in e.get('tool','')})}
    (root/'memory-deployment-traces.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
