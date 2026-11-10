"""Create isolated App history tables and grant only their runtime permissions."""
import argparse
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel
from databricks.sdk.service.postgres import Role, RoleRoleSpec, RoleIdentityType, RoleAuthMethod
from fe_multi_agent.memory import lakebase_connection

parser=argparse.ArgumentParser()
parser.add_argument('--profile',required=True)
args=parser.parse_args()
w=WorkspaceClient(profile=args.profile)
principal=w.apps.get('fe-platform-team-fevm').service_principal_client_id
branch='projects/fe-agentops-multi/branches/production'
w.permissions.update('database-projects','fe-agentops-multi',access_control_list=[AccessControlRequest(service_principal_name=principal,permission_level=PermissionLevel.CAN_USE)])
if not any((r.spec and r.spec.postgres_role==principal) or (r.name or '').endswith('/roles/app-history-'+principal) for r in w.postgres.list_roles(parent=branch)):
    w.postgres.create_role(parent=branch,role_id="app-history-"+principal,role=Role(spec=RoleRoleSpec(postgres_role=principal,identity_type=RoleIdentityType.SERVICE_PRINCIPAL,auth_method=RoleAuthMethod.LAKEBASE_OAUTH_V1))).wait()
with lakebase_connection(w,branch+'/endpoints/primary',user=w.current_user.me().user_name) as conn:
    cur=conn.cursor()
    cur.execute('CREATE SCHEMA IF NOT EXISTS app_history')
    cur.execute('CREATE TABLE IF NOT EXISTS app_history.conversations (id uuid PRIMARY KEY,owner_id text NOT NULL,title text NOT NULL,turns jsonb NOT NULL,revision integer NOT NULL,updated_at timestamptz NOT NULL DEFAULT now())')
    cur.execute('CREATE INDEX IF NOT EXISTS conversations_owner_updated ON app_history.conversations(owner_id,updated_at DESC)')
    cur.execute('CREATE TABLE IF NOT EXISTS app_history.preferences (owner_id text PRIMARY KEY,settings jsonb NOT NULL)')
    # Principal comes from the workspace API; quote it as an identifier.
    role='"'+principal.replace('"','""')+'"'
    cur.execute('GRANT CONNECT ON DATABASE databricks_postgres TO '+role)
    cur.execute('GRANT USAGE ON SCHEMA app_history TO '+role)
    cur.execute('GRANT SELECT,INSERT,UPDATE ON app_history.conversations,app_history.preferences TO '+role)
print('App history schema and scoped grants ready')
