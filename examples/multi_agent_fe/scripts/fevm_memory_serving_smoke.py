"""Smoke-test deployed memory tools using a signed isolated evaluation owner.

No end-user identity is impersonated. Small evaluation notes remain for audit.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import uuid
import requests
from databricks.sdk import WorkspaceClient
from fe_multi_agent.memory import sign_context, MemoryWorkspace, lakebase_connection


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',required=True)
    parser.add_argument('--version',required=True)
    args=parser.parse_args()
    w=WorkspaceClient(profile=args.profile)
    e=w.serving_endpoints.get('fe-platform-agent-team')
    if (e.state.ready.value!='READY' or e.state.config_update.value!='NOT_UPDATING'
            or e.config.served_entities[0].entity_version!=args.version):
        raise RuntimeError('Requested model version is not fully ready')
    secret=base64.b64decode(w.secrets.get_secret(scope='fe-agentops-context',key='signing-key').value).decode()
    owner='evaluation:serving-file-memory:'+str(uuid.uuid4())
    report={'owner':owner,'version':args.version,'scenarios':{}}
    out=Path(__file__).resolve().parents[1]/'validation-results/memory-serving-smoke.json'
    out.parent.mkdir(exist_ok=True)
    def call(name,question):
        messages=[{'role':'user','content':question}]
        context={'owner_id':owner,'conversation_id':str(uuid.uuid4()),'summary':'','preferences':{},
                 'expires_at':int(time.time())+300,
                 'messages_sha256':hashlib.sha256(json.dumps(messages,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
        payload={'input':messages,'custom_inputs':{'application_context':sign_context(context,secret)},'stream':True}
        events=[];items=[];completed=False;custom={}
        started=time.monotonic()
        with requests.post(w.config.host.rstrip('/')+'/serving-endpoints/fe-platform-agent-team/invocations',
                           headers=w.config.authenticate(),json=payload,stream=True,timeout=(20,240)) as response:
            response.raise_for_status()
            for raw in response.iter_lines(chunk_size=1,decode_unicode=True):
                if not raw or not raw.startswith('data:'):continue
                raw=raw[5:].strip()
                if raw=='[DONE]':break
                event=json.loads(raw)
                if event.get('error') or event.get('type') in {'error','response.failed'}:
                    raise RuntimeError(json.dumps(event))
                meta=event.get('custom_outputs') or {}
                if meta.get('activity'):events.append(meta['activity'])
                if event.get('type')=='response.output_item.done':
                    items.append(event['item']);custom.update({k:v for k,v in meta.items() if k!='activity'})
                if event.get('type')=='response.completed':
                    completed=True;custom.update((event.get('response') or {}).get('custom_outputs') or {})
        text='\n'.join(c.get('text','') for i in items for c in i.get('content',[]))
        assert completed and text and custom.get('trace_id'),(name,completed,custom)
        assert 'Memory was unavailable' not in text,(name,text)
        report['scenarios'][name]={'answer':text,'trace_id':custom['trace_id'],'activity':events,
                                    'seconds':round(time.monotonic()-started,1)}
        out.write_text(json.dumps(report,indent=2))
        print(json.dumps({'scenario':name,'trace_id':custom['trace_id'],'seconds':report['scenarios'][name]['seconds'],
                          'answer':text[:1200]}),flush=True)
        return text,events
    text,events=call('remember','Remember our workload fact in /memories/workloads/smoke-batch.md: Our scheduled batch runs in eu-west-1. Save it for future sessions. Do not call a worker or estimate costs.')
    assert any(e.get('tool')=='write_memory_file' for e in events),'No explicit memory write observed'
    store=MemoryWorkspace(lambda:lakebase_connection(w,'projects/fe-agentops-multi/branches/production/endpoints/primary',user=w.current_user.me().user_name),owner,'validation-reader')
    assert 'eu-west-1' in store.read_file('/memories/workloads/smoke-batch.md').get('content',''),'Runtime memory write did not persist'
    text,events=call('cross_session_recall','Which region does our scheduled batch use? Read the saved memory file and answer from it. Do not call a worker.')
    assert 'eu-west-1' in text and any(e.get('tool')=='read_memory_file' for e in events)
    text,events=call('platform','Explain MLflow tracing for LangGraph with a retrieved Databricks documentation citation. Do not estimate costs.')
    assert 'docs.databricks.com' in text and any(e.get('tool')=='retrieve_platform_docs' for e in events)
    assert {e.get('agent') for e in events if e.get('kind')=='agent_started'}=={'platform_docs'}
    text,events=call('cost','Ask Cost intelligence to identify the missing inputs for estimating monthly Databricks cost for our remembered scheduled batch. We have not supplied volume, run duration or frequency. Do not invent a numerical quote.')
    assert {e.get('agent') for e in events if e.get('kind')=='agent_started'}=={'cost_estimation'},'Incorrect cost routing'
    print(json.dumps({'status':'passed','evidence':str(out),'scenarios':list(report['scenarios'])}))


if __name__=='__main__':main()
