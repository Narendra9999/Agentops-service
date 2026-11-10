"""User-scoped durable memory and signed App context; no identity from model args."""
import base64
import hashlib
import hmac
import json
import os
import re
import time
import uuid
from contextlib import contextmanager


def sign_context(context, secret=None):
    secret = secret or os.environ['AGENT_CONTEXT_SIGNING_KEY']
    payload = base64.urlsafe_b64encode(json.dumps(context, separators=(',', ':')).encode()).decode()
    signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return {'payload': payload, 'signature': signature}


def verify_context(envelope, messages, secret=None):
    secret = secret or os.environ['AGENT_CONTEXT_SIGNING_KEY']
    payload = envelope.get('payload', '')
    expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, envelope.get('signature', '')):
        raise ValueError('Invalid application context signature')
    data = json.loads(base64.urlsafe_b64decode(payload))
    now = time.time()
    if not data.get('owner_id') or not now <= data.get('expires_at', 0) <= now + 600:
        raise ValueError('Application context expired or missing owner')
    digest = hashlib.sha256(json.dumps(messages, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if not hmac.compare_digest(digest, data.get('messages_sha256', '')):
        raise ValueError('Application context does not match request')
    return data


@contextmanager
def lakebase_connection(workspace, endpoint, database='databricks_postgres', user=None):
    import ssl
    import pg8000.dbapi
    host = workspace.postgres.get_endpoint(name=endpoint).status.hosts.host
    token = workspace.postgres.generate_database_credential(endpoint=endpoint).token
    conn = pg8000.dbapi.connect(host=host, database=database,
        user=user or os.getenv('DATABRICKS_CLIENT_ID') or workspace.current_user.me().user_name,
        password=token, ssl_context=ssl.create_default_context(), timeout=30,
        startup_params={'statement_timeout': '15000'})
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


DDL = [
    'CREATE SCHEMA IF NOT EXISTS agent_memory',
    '''CREATE TABLE IF NOT EXISTS agent_memory.items (
      id uuid PRIMARY KEY, owner_id text NOT NULL, kind text NOT NULL,
      content text NOT NULL, evidence text NOT NULL, status text NOT NULL,
      source_conversation text NOT NULL, source_trace text NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now(), expires_at timestamptz NOT NULL,
      UNIQUE(owner_id,kind,content))''',
    'CREATE INDEX IF NOT EXISTS memory_owner_status ON agent_memory.items(owner_id,status,expires_at)',
    '''CREATE TABLE IF NOT EXISTS agent_memory.audit (
      id bigserial PRIMARY KEY, owner_id text NOT NULL, memory_id uuid,
      action text NOT NULL, trace_id text NOT NULL, created_at timestamptz NOT NULL DEFAULT now())''',
    '''CREATE TABLE IF NOT EXISTS app_history.summaries (
      conversation_id uuid PRIMARY KEY, owner_id text NOT NULL, revision integer NOT NULL,
      summary text NOT NULL, updated_at timestamptz NOT NULL DEFAULT now())''',
]


class MemoryStore:
    def __init__(self, connection, owner, conversation, trace_id='', ttl_days=90):
        self.connection, self.owner, self.conversation = connection, owner, conversation
        self.trace_id, self.ttl_days = trace_id, ttl_days

    def _audit(self, cur, action, memory_id=None):
        cur.execute('INSERT INTO agent_memory.audit(owner_id,memory_id,action,trace_id) VALUES (%s,%s,%s,%s)',
                    (self.owner, memory_id, action, self.trace_id))

    def search(self, query):
        # English text retrieval plus recent explicit facts, always owner/expiry scoped.
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute("""SELECT id::text,kind,content,status,expires_at::text,created_at::text FROM agent_memory.items
              WHERE owner_id=%s AND status='active' AND expires_at>now()
              AND (kind='user_fact' OR to_tsvector('english',content) @@ plainto_tsquery('english',%s))
              ORDER BY ts_rank(to_tsvector('english',content),plainto_tsquery('english',%s)) DESC,created_at DESC LIMIT 8""",
              (self.owner, query[:2000], query[:2000]))
            rows = cur.fetchall()
            self._audit(cur, 'search')
        return [dict(zip(('id','kind','content','status','expires_at','created_at'), row)) for row in rows]

    def store(self, kind, content, evidence, user_message, cost_evidence):
        if kind not in {'user_fact', 'cost_pattern'} or not 3 <= len(content) <= 1200 or not 3 <= len(evidence) <= 1200:
            return {'stored': False, 'reason': 'Invalid kind or memory length'}
        from framework.guardrails.pre_llm import PreLLMGuardrails
        from framework.guardrails.post_llm import PostLLMGuardrails
        combined = content + '\n' + evidence
        if (PreLLMGuardrails({'enabled_checks':['injection','pii']}).check(combined).get('blocked') or
            PostLLMGuardrails({'enabled_checks':['pii_leakage']}).check('', combined).get('blocked') or
            re.search(r'(?i)(api[_ -]?key|password|secret|bearer\s|dapi[a-z0-9])', combined)):
            return {'stored': False, 'reason': 'Sensitive or instruction-like memory rejected'}
        # Both content and evidence must be extractive: no invented durable facts.
        source = user_message if kind == 'user_fact' else cost_evidence
        if content.casefold() not in source.casefold() or evidence.casefold() not in source.casefold():
            return {'stored': False, 'reason': 'Memory must quote its source evidence'}
        status = 'active' if kind == 'user_fact' else 'pending'
        memory_id = str(uuid.uuid4())
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute('''INSERT INTO agent_memory.items
              (id,owner_id,kind,content,evidence,status,source_conversation,source_trace,expires_at)
              VALUES (%s,%s,%s,%s,%s,%s,%s,%s,now()+(%s * interval '1 day'))
              ON CONFLICT(owner_id,kind,content) DO UPDATE SET expires_at=EXCLUDED.expires_at
              RETURNING id::text,status''',
              (memory_id,self.owner,kind,content,evidence,status,self.conversation,self.trace_id,self.ttl_days))
            memory_id,status = cur.fetchone()
            self._audit(cur, 'store', memory_id)
        return {'stored':True,'id':memory_id,'status':status}

    def change(self, memory_id, action, user_message):
        try:
            memory_id = str(uuid.UUID(memory_id))
        except ValueError:
            return {'changed':False,'reason':'Invalid memory ID'}
        command = 'approve pattern' if action == 'approve' else 'forget memory'
        # Require a direct command, not an embedded/quoted instruction.
        if user_message.strip().casefold().rstrip('.') != f'{command} {memory_id}':
            return {'changed':False,'reason':f'Ask explicitly: {command} {memory_id}'}
        with self.connection() as conn:
            cur=conn.cursor()
            if action == 'approve':
                cur.execute("UPDATE agent_memory.items SET status='active' WHERE id=%s AND owner_id=%s AND kind='cost_pattern' AND status='pending' AND expires_at>now() RETURNING id",(memory_id,self.owner))
            else:
                cur.execute('DELETE FROM agent_memory.items WHERE id=%s AND owner_id=%s RETURNING id',(memory_id,self.owner))
            changed=bool(cur.fetchone())
            if changed:self._audit(cur,action,memory_id)
        return {'changed':changed,'id':memory_id}


# Virtual Markdown files: durable across processes; never access container paths.
WORKSPACE_DDL = [
    """CREATE TABLE IF NOT EXISTS agent_memory.files (
      owner_id text NOT NULL, path text NOT NULL, content text NOT NULL,
      revision integer NOT NULL, deleted boolean NOT NULL DEFAULT false,
      expires_at timestamptz NOT NULL, PRIMARY KEY(owner_id,path))""",
    """CREATE TABLE IF NOT EXISTS agent_memory.file_versions (
      id bigserial PRIMARY KEY, owner_id text NOT NULL, path text NOT NULL,
      revision integer NOT NULL, content text NOT NULL, action text NOT NULL,
      source_conversation text NOT NULL, source_trace text NOT NULL,
      source_evidence text NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now())""",
]
DDL.extend(WORKSPACE_DDL)


class MemoryWorkspace(MemoryStore):
    """Owner-scoped virtual files, bounded reads and optimistic concurrent edits.

    Existing structured evidence validation remains available for future migration;
    runtime file tools use only this workspace interface.
    """

    @staticmethod
    def validate_path(path):
        if not isinstance(path, str) or not re.fullmatch(
                r'/memories/(?:[a-z0-9_-]+/)*[a-z0-9_-]+\.md', path) or len(path) > 240:
            raise ValueError('Use a lowercase Markdown path under /memories/; traversal is forbidden')
        return path

    def files(self, query=''):
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute("""SELECT path,revision,left(content,240) FROM agent_memory.files
              WHERE owner_id=%s AND NOT deleted AND expires_at>now()
              AND (%s='' OR position(lower(%s) in lower(path || ' ' || content))>0)
              ORDER BY path LIMIT 40""", (self.owner, query[:200], query[:200]))
            return [dict(zip(('path','revision','preview'), row)) for row in cur.fetchall()]

    def read_file(self, path):
        self.validate_path(path)
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute("""SELECT content,revision FROM agent_memory.files
              WHERE owner_id=%s AND path=%s AND NOT deleted AND expires_at>now()""", (self.owner,path))
            row = cur.fetchone()
        return dict(zip(('content','revision'),row)) if row else {'not_found':True}

    def write_file(self, path, content, expected_revision, user_message, cost_evidence):
        self.validate_path(path)
        if not isinstance(expected_revision,int) or expected_revision < 0:
            return {'written':False,'reason':'Invalid expected revision'}
        if not 3 <= len(content) <= 12000:
            return {'written':False,'reason':'File must contain 3 to 12000 characters'}
        from framework.guardrails.pre_llm import PreLLMGuardrails
        from framework.guardrails.post_llm import PostLLMGuardrails
        if (PreLLMGuardrails({'enabled_checks':['injection','pii']}).check(content).get('blocked') or
            PostLLMGuardrails({'enabled_checks':['pii_leakage']}).check('',content).get('blocked') or
            re.search(r'(?i)(api[_ -]?key|password|secret|bearer\s|dapi[a-z0-9])',content)):
            return {'written':False,'reason':'Sensitive or instruction-like content rejected'}
        # A curation edit can reorganize old evidence and add current observed evidence.
        # Persist provenance alongside every revision; model prose is not verified truth.
        return self._mutate(path,content,expected_revision,'write',
                            json.dumps({'user_sha256':hashlib.sha256(user_message.encode()).hexdigest(),
                                        'cost_worker_sha256':hashlib.sha256(cost_evidence.encode()).hexdigest()}))

    def delete_file(self, path, expected_revision):
        self.validate_path(path)
        return self._mutate(path,'',expected_revision,'delete')

    def _mutate(self, path, content, expected_revision, action, source_evidence='{}'):
        with self.connection() as conn:
            cur = conn.cursor()
            if expected_revision == 0 and action == 'write':
                cur.execute("""INSERT INTO agent_memory.files(owner_id,path,content,revision,expires_at)
                  VALUES (%s,%s,%s,1,now()+(%s * interval '1 day'))
                  ON CONFLICT DO NOTHING RETURNING revision""", (self.owner,path,content,self.ttl_days))
            else:
                cur.execute("""UPDATE agent_memory.files SET content=%s,revision=revision+1,deleted=%s,
                  expires_at=now()+(%s * interval '1 day')
                  WHERE owner_id=%s AND path=%s AND revision=%s RETURNING revision""",
                  (content,action=='delete',self.ttl_days,self.owner,path,expected_revision))
            row = cur.fetchone()
            if not row:
                return {'written':False,'reason':'Revision conflict or missing file; read before editing'}
            revision = row[0]
            cur.execute("""INSERT INTO agent_memory.file_versions
              (owner_id,path,revision,content,action,source_conversation,source_trace,source_evidence)
              VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
              (self.owner,path,revision,content,action,self.conversation,self.trace_id,source_evidence))
        return {'written':True,'path':path,'revision':revision,'deleted':action=='delete'}

    def history(self, path):
        self.validate_path(path)
        with self.connection() as conn:
            cur=conn.cursor()
            cur.execute("""SELECT revision,content,action,created_at::text FROM agent_memory.file_versions
              WHERE owner_id=%s AND path=%s ORDER BY revision DESC LIMIT 10""",(self.owner,path))
            return [dict(zip(('revision','content','action','created_at'),row)) for row in cur.fetchall()]
