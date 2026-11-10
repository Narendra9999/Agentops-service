"""Approval-gated additive FEVM memory provisioning. Never prints secret values.

Run only after approval of the tables, isolated secret scope and listed grants.
"""
import argparse
import secrets
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import AclPermission
from fe_multi_agent.memory import DDL,WORKSPACE_DDL,lakebase_connection


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',required=True)
    args=parser.parse_args()
    w=WorkspaceClient(profile=args.profile)
    app=w.apps.get('fe-platform-team-fevm')
    runtime=w.serving_endpoints.get('fe-platform-agent-team').config.served_entities[0].environment_vars['DATABRICKS_CLIENT_ID']
    endpoint='projects/fe-agentops-multi/branches/production/endpoints/primary'
    with lakebase_connection(w,endpoint,user=w.current_user.me().user_name) as conn:
        cur=conn.cursor()
        # Structured item tables are legacy; provision only the active file workspace.
        for sql in [DDL[0],DDL[4],*WORKSPACE_DDL]:cur.execute(sql)
        role='"'+runtime.replace('"','""')+'"'
        cur.execute('GRANT USAGE ON SCHEMA agent_memory TO '+role)
        cur.execute('GRANT SELECT,INSERT,UPDATE ON agent_memory.files TO '+role)
        cur.execute('GRANT SELECT,INSERT ON agent_memory.file_versions TO '+role)
        cur.execute('GRANT USAGE ON SEQUENCE agent_memory.file_versions_id_seq TO '+role)
        role='"'+app.service_principal_client_id.replace('"','""')+'"'
        cur.execute('GRANT SELECT,INSERT,UPDATE ON app_history.summaries TO '+role)
    scope='fe-agentops-context'
    if scope not in {s.name for s in w.secrets.list_scopes()}:w.secrets.create_scope(scope=scope)
    if 'signing-key' not in {s.key for s in w.secrets.list_secrets(scope=scope)}:
        w.secrets.put_secret(scope=scope,key='signing-key',string_value=secrets.token_urlsafe(48))
    # This scope contains only signing material, never runtime OAuth credentials.
    w.secrets.put_acl(scope=scope,principal=app.service_principal_client_id,permission=AclPermission.READ)
    print('Memory schema, scoped grants and isolated signing secret ready; no secrets returned.')


if __name__=='__main__':main()
