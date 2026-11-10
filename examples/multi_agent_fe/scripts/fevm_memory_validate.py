"""Validate approved durable Lakebase file memory in an isolated evaluation namespace.

Creates small test files, soft-deletes them and retains version history for audit.
Does not register/deploy a model or invoke Genie.
"""
import argparse
import json
import uuid
from databricks.sdk import WorkspaceClient
from fe_multi_agent.memory import MemoryWorkspace, lakebase_connection


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',required=True)
    args=parser.parse_args()
    w=WorkspaceClient(profile=args.profile)
    endpoint='projects/fe-agentops-multi/branches/production/endpoints/primary'
    user=w.current_user.me().user_name
    connection=lambda:lakebase_connection(w,endpoint,user=user)
    owner='evaluation:file-memory:'+str(uuid.uuid4())
    first=MemoryWorkspace(connection,owner,'session-one')
    later=MemoryWorkspace(connection,owner,'session-two')
    other=MemoryWorkspace(connection,owner+':other','session-other')
    path='/memories/workloads/validation.md'
    content='# Validation workload\nRegion: eu-west-1\nStatus: test fixture'
    assert first.write_file(path,content,0,content,'')['revision']==1
    assert later.read_file(path)['content']==content
    assert other.read_file(path).get('not_found') and other.files()==[]
    assert later.write_file(path,content+'\nFrequency: daily',1,'Frequency: daily','')['revision']==2
    assert not first.write_file(path,content,1,content,'')['written']
    assert [r['revision'] for r in later.history(path)]==[2,1]
    assert later.delete_file(path,2)['deleted']
    assert later.read_file(path).get('not_found') and later.files()==[]
    assert later.write_file(path,content,3,content,'')['revision']==4
    assert later.delete_file(path,4)['deleted']
    print(json.dumps({'persistent_lakebase_files':True,'fresh_session_recall':True,
                     'owner_isolation':True,'stale_write_rejected':True,'restore_from_history':True,
                     'evaluation_owner':owner,'test_file_soft_deleted':True,'endpoint_traffic_changed':False}))


if __name__=='__main__':main()
