"""Authenticated live-App smoke test, using the selected user's own test conversations.

Creates labeled validation history, checks fresh-conversation memory recall,
renames a saved conversation, then asks the supervisor to delete its test note.
"""
import argparse
import json
from pathlib import Path
import time
import uuid
import requests
from databricks.sdk import WorkspaceClient


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',required=True)
    args=parser.parse_args()
    w=WorkspaceClient(profile=args.profile)
    base=w.apps.get('fe-platform-team-fevm').url.rstrip('/')
    session=requests.Session()
    session.headers.update(w.config.authenticate())
    report={'app':base,'scenarios':{}}
    # Alphabetic fixture IDs avoid looking like a phone number to PII checks.
    suffix=''.join(chr(97+int(c,16)) for c in uuid.uuid4().hex[:12])
    note='/memories/workloads/app-validation-'+suffix+'.md'
    def request(method,path,**kwargs):
        r=session.request(method,base+path,timeout=(20,240),**kwargs)
        r.raise_for_status();return r
    assert request('GET','/health').json()['status']=='ok'
    identity=request('GET','/api/whoami').json()
    assert identity.get('user')
    request('GET','/api/preferences')
    def turn(name,question):
        cid=str(uuid.uuid4())
        body={'title':'Deployment validation '+name+' '+suffix,'revision':0,
              'turns':[{'role':'user','content':question}]}
        assert request('PUT','/api/conversations/'+cid,json=body).json()['revision']==1
        events=[];result=None;kind=None
        with request('POST','/api/chat/stream',json={'conversation_id':cid,
             'messages':[{'role':'user','content':question}]},stream=True) as response:
            for line in response.iter_lines(chunk_size=1,decode_unicode=True):
                if line.startswith('event:'):kind=line[6:].strip()
                elif line.startswith('data:'):
                    value=json.loads(line[5:])
                    if kind=='error':raise RuntimeError(json.dumps(value))
                    if kind=='activity':events.append(value)
                    if kind=='result':result=value
        assert result and result.get('output'),(name,result)
        answer='\n'.join(c.get('text','') for i in result['output'] for c in i.get('content',[]))
        assert 'Memory was unavailable' not in answer,(name,answer)
        report['scenarios'][name]={'conversation_id':cid,'answer':answer,
            'trace_id':result.get('custom_outputs',{}).get('trace_id'),'activity':events}
        Path(__file__).resolve().parents[1].joinpath('validation-results/memory-app-smoke.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'scenario':name,'trace_id':report['scenarios'][name]['trace_id'],'answer':answer[:1000]}),flush=True)
        return cid,answer,events
    cid,answer,events=turn('remember','Save this deployment-test fact in '+note+': Our app-validation batch uses eu-west-1. Use memory tools only; no specialist is needed.')
    assert any(e.get('tool')=='write_memory_file' for e in events)
    renamed=request('PATCH','/api/conversations/'+cid,json={'title':'Deployment validation — saved memory '+suffix,'revision':1}).json()
    assert renamed['revision']==2
    assert request('GET','/api/conversations/'+cid).json()['title']==renamed['title']
    _,answer,events=turn('fresh-conversation','Read '+note+' and tell me which region our app-validation batch uses. Use saved memory only; no specialist is needed.')
    assert 'eu-west-1' in answer and any(e.get('tool')=='read_memory_file' for e in events)
    _,answer,events=turn('cleanup','Delete the deployment-test memory file '+note+'. The validation is complete. Use memory tools only; no specialist is needed.')
    assert any(e.get('tool')=='delete_memory_file' for e in events)
    assert len(request('GET','/api/conversations').json()['conversations'])>=3
    print(json.dumps({'status':'passed','history':True,'rename':True,'app_to_serving_memory':True,
                      'fresh_conversation_recall':True,'test_note_delete_requested':True}))


if __name__=='__main__':main()
