"""FastAPI take-home UI. Databricks Apps perimeter authenticates demo users."""
import logging
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID

from databricks.sdk import WorkspaceClient
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, PrivateAttr
from requests.exceptions import Timeout

app = FastAPI(title="Platform Engineering Agent Team")
logger = logging.getLogger(__name__)
app.mount('/static', StaticFiles(directory=Path(__file__).with_name('static')), name='static')


class Message(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=12000)


class Chat(BaseModel):
    conversation_id: UUID | None = None
    _agent_context: dict | None = PrivateAttr(default=None)
    messages: list[Message] = Field(min_length=1, max_length=30)
    model_config = {"extra": "forbid"}


def authorized(request):
    if os.getenv("FE_MULTI_ALLOW_LOCAL") != "true" and not request.headers.get("x-forwarded-user"):
        raise HTTPException(401, "Open this app through Databricks Apps authentication")
    origin = request.headers.get("origin")
    host = request.headers.get("x-forwarded-host", request.headers.get("host", ""))
    if origin and urlparse(origin).netloc != host:
        raise HTTPException(403, "Cross-origin requests are not allowed")


@app.get("/", response_class=HTMLResponse)
def index():
    return Path(__file__).with_name("index.html").read_text()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get('/api/whoami')
def whoami(request: Request):
    authorized(request)
    return {'user': request.headers.get('x-forwarded-email') or request.headers.get('x-forwarded-user'),
            'execution_identity': 'Dedicated agent service principal',
            'experiment_id': os.getenv('TRACE_EXPERIMENT_ID'),
            'workspace_id': os.getenv('WORKSPACE_ID'),
            'workspace_url': WorkspaceClient().config.host}


@app.post("/api/chat")
def chat(body: Chat, request: Request):
    validate_chat(body, request)
    return invoke(body)


def validate_chat(body, request):
    authorized(request)
    if any(m.role not in {"user", "assistant"} for m in body.messages):
        raise HTTPException(422, "Only user and assistant messages are allowed")
    if body.messages[-1].role != "user":
        raise HTTPException(422, "The last message must be a user message")
    if os.getenv('AGENT_MEMORY_ENABLED') == 'true':
        prepare_agent_context(body, request)


def invoke(body):
    # Older tabs and API clients share the same heartbeat-capable transport.
    # Avoid duplicate Genie executions; do not replay invocations.
    upstream = invoke_stream(body)
    try:
        for kind, data in upstream:
            if kind == 'result':
                return data
        raise RuntimeError('Serving stream ended without a final response')
    except AgentExecutionError as exc:
        raise HTTPException(504 if exc.details.get('code') == 'genie_timeout' else 502, failure_details(body, exc.details)['message'])
    except Timeout:
        logger.exception("Serving transport timed out")
        raise HTTPException(504, "Agent connection timed out. Execution may still complete. Check the trace before retrying.")
    except Exception:
        logger.exception("Serving invocation failed")
        raise HTTPException(502, "Agent request failed. Check the serving trace before retrying.")
    finally:
        upstream.close()


class AgentExecutionError(RuntimeError):
    def __init__(self, details):
        self.details = details
        super().__init__('Agent execution failed')


def failure_details(body, details=None):
    details = details or {}
    message = ('Genie did not finish within its 180-second wait limit. Open the Genie conversation to check its status; no automatic retry was made.'
               if details.get('code') == 'genie_timeout' else 'The agent execution failed. Inspect the execution trace for the failed operation.')
    return {**{k: details[k] for k in ('code', 'trace_id', 'space_id', 'conversation_id', 'message_id', 'status', 'mode') if k in details}, 'message': message}


def invoke_stream(body):
    """Proxy actual ResponsesAgent events; one request, no polling or retries."""
    import requests
    w = WorkspaceClient()
    url = w.config.host.rstrip('/') + '/serving-endpoints/' + os.environ['SERVING_ENDPOINT'] + '/invocations'
    payload = {'input': [m.model_dump() for m in body.messages],
               'stream': True, 'custom_inputs': {'stream_activity': True}}
    if body._agent_context:
        payload['custom_inputs']['application_context'] = body._agent_context
    output, custom, usage, completed_received = [], {}, None, False
    with requests.post(url, headers=w.config.authenticate(), json=payload,
                       stream=True, timeout=(20, 240)) as response:
        response.raise_for_status()
        response.encoding = 'utf-8'
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line or not line.startswith('data:'):
                continue
            raw = line[5:].strip()
            if raw == '[DONE]':
                break
            event = json.loads(raw)
            if event.get('error') or event.get('type') in {'error', 'response.failed'}:
                raise AgentExecutionError((event.get('custom_outputs') or {}).get('failure') or {})
            meta = event.get('custom_outputs') or {}
            if meta.get('activity'):
                yield 'activity', meta['activity']
            else:
                yield 'heartbeat', {}
            if event.get('type') == 'response.completed':
                completed_received = True
                completed = event.get('response') or {}
                usage = completed.get('usage')
                custom.update(completed.get('custom_outputs') or {})
            if event.get('type') == 'response.output_item.done':
                output.append(event['item'])
                custom.update({key: value for key, value in meta.items() if key != 'activity'})
        if not output or not completed_received:
            raise RuntimeError('Serving stream ended without a final response')
        yield 'result', {'output': output, 'custom_outputs': custom, **({'usage': usage} if usage is not None else {})}


@app.post('/api/chat/stream')
def stream_chat(body: Chat, request: Request):
    validate_chat(body, request)

    def events():
        started = time.monotonic()
        def event(kind, data):
            return 'event: ' + kind + '\ndata: ' + json.dumps(data) + '\n\n'

        yield event('accepted', {'message': 'Request received', 'elapsed_seconds': 0})
        upstream = invoke_stream(body)
        try:
            for kind, data in upstream:
                yield event(kind, {**data, 'elapsed_seconds': round(time.monotonic() - started)})
        except Exception as exc:
            logger.exception('Agent stream failed')
            details = exc.details if isinstance(exc, AgentExecutionError) else {}
            yield event('error', failure_details(body, details))
        finally:
            upstream.close()

    return StreamingResponse(events(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

# History is scoped to the Databricks Apps authenticated subject, never a client email.
from contextlib import contextmanager
import ssl


@contextmanager
def history_db():
    import pg8000.dbapi
    w = WorkspaceClient()
    endpoint = os.environ['HISTORY_LAKEBASE_ENDPOINT']
    if not __import__('re').fullmatch(r'projects/[^/]+/branches/[^/]+/endpoints/[^/]+', endpoint):
        raise ValueError('History requires a Lakebase project endpoint resource')
    # SDK `postgres` is the Databricks Lakebase Autoscaling API.
    host = w.postgres.get_endpoint(name=endpoint).status.hosts.host
    token = w.postgres.generate_database_credential(endpoint=endpoint).token
    connection = pg8000.dbapi.connect(
        host=host, database='databricks_postgres',
        user=os.environ.get('DATABRICKS_CLIENT_ID') or w.current_user.me().user_name,
        password=token, ssl_context=ssl.create_default_context(), timeout=30,
        startup_params={'statement_timeout': '15000'})
    try:
        yield connection.cursor()
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def history_owner(request):
    authorized(request)
    subject = request.headers.get('x-forwarded-user')
    if not subject:
        raise HTTPException(401, 'Sign in to access saved conversations')
    return subject


class SavedConversation(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    revision: int = Field(ge=0)
    turns: list[dict] = Field(max_length=500)


class Preferences(BaseModel):
    display_name: str = Field(default='', max_length=80)
    chart_type: str = 'bar'


class ConversationTitle(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    revision: int = Field(ge=1)


@app.patch('/api/conversations/{conversation_id}')
def rename_conversation(conversation_id: UUID, body: ConversationTitle, request: Request):
    owner = history_owner(request)
    title = body.title.strip()
    if not title:
        raise HTTPException(422, 'Enter a conversation name')
    with history_db() as cur:
        cur.execute('UPDATE app_history.conversations SET title=%s,revision=revision+1,updated_at=now() WHERE id=%s AND owner_id=%s AND revision=%s RETURNING revision',
                    (title, str(conversation_id), owner, body.revision))
        row = cur.fetchone()
        if not row:
            raise HTTPException(409, 'This conversation changed or is unavailable. Reload history before renaming.')
        return {'title': title, 'revision': row[0]}


@app.get('/api/conversations')
def conversations(request: Request):
    owner = history_owner(request)
    with history_db() as cur:
        cur.execute('SELECT id::text,title,revision,updated_at::text FROM app_history.conversations WHERE owner_id=%s ORDER BY updated_at DESC', (owner,))
        return {'conversations': [dict(zip(('id','title','revision','updated_at'), row)) for row in cur.fetchall()]}


@app.get('/api/conversations/{conversation_id}')
def conversation(conversation_id: UUID, request: Request):
    owner = history_owner(request)
    with history_db() as cur:
        cur.execute('SELECT title,revision,turns FROM app_history.conversations WHERE owner_id=%s AND id=%s', (owner, str(conversation_id)))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, 'Conversation not found')
        return {'id': str(conversation_id), 'title': row[0], 'revision': row[1], 'turns': row[2] if isinstance(row[2], list) else json.loads(row[2])}


@app.put('/api/conversations/{conversation_id}')
def save_conversation(conversation_id: UUID, body: SavedConversation, request: Request):
    owner = history_owner(request)
    encoded = json.dumps(body.turns)
    if len(encoded.encode()) > 2_000_000:
        raise HTTPException(413, 'Conversation is too large. Start a new conversation.')
    if any(t.get('role') not in {'user', 'assistant', 'notice'} for t in body.turns):
        raise HTTPException(422, 'Invalid saved turn')
    with history_db() as cur:
        if body.revision == 0:
            cur.execute('INSERT INTO app_history.conversations (id,owner_id,title,turns,revision) VALUES (%s,%s,%s,%s::jsonb,1) ON CONFLICT DO NOTHING RETURNING revision',
                        (str(conversation_id), owner, body.title, encoded))
        else:
            cur.execute('UPDATE app_history.conversations SET title=%s,turns=%s::jsonb,revision=revision+1,updated_at=now() WHERE id=%s AND owner_id=%s AND revision=%s RETURNING revision',
                        (body.title, encoded, str(conversation_id), owner, body.revision))
        row = cur.fetchone()
        if not row:
            raise HTTPException(409, 'This conversation changed in another tab. Reload it before continuing.')
        return {'revision': row[0]}


@app.get('/api/preferences')
def preferences(request: Request):
    owner = history_owner(request)
    with history_db() as cur:
        cur.execute('SELECT settings FROM app_history.preferences WHERE owner_id=%s', (owner,))
        row = cur.fetchone()
        return row[0] if row and isinstance(row[0], dict) else json.loads(row[0]) if row else Preferences().model_dump()


@app.put('/api/preferences')
def save_preferences(body: Preferences, request: Request):
    owner = history_owner(request)
    if body.chart_type not in {'bar', 'line'}:
        raise HTTPException(422, 'Choose bar or line')
    with history_db() as cur:
        cur.execute('INSERT INTO app_history.preferences (owner_id,settings) VALUES (%s,%s::jsonb) ON CONFLICT (owner_id) DO UPDATE SET settings=excluded.settings', (owner, body.model_dump_json()))
    return body


def prepare_agent_context(body, request):
    """Rebuild prompt history from owner-scoped storage; sign exact request content."""
    import hashlib
    import hmac
    import base64
    owner=history_owner(request)
    if body.conversation_id is None:
        raise HTTPException(422,'A saved conversation is required for agent memory')
    question=body.messages[-1].content
    with history_db() as cur:
        cur.execute('SELECT turns,revision FROM app_history.conversations WHERE owner_id=%s AND id=%s',(owner,str(body.conversation_id)))
        row=cur.fetchone()
        if not row:raise HTTPException(404,'Conversation not found')
        turns=row[0] if isinstance(row[0],list) else json.loads(row[0])
        history=[{'role':t['role'],'content':str(t.get('content',''))} for t in turns if t.get('role') in {'user','assistant'}]
        if not history or history[-1]!={'role':'user','content':question}:
            raise HTTPException(409,'Conversation changed. Reload before sending.')
        # Bound recent context by characters as well as turn count. Preserve current question.
        recent=[history[-1]];size=len(question)
        for turn in reversed(history[:-1]):
            if len(recent)>=12 or size+len(turn['content'])>20000:break
            recent.insert(0,turn);size+=len(turn['content'])
        if recent[0]['role']=='assistant':recent.pop(0)
        older=history[:-len(recent)]
        summary='Extractive conversation history (may omit details):\n'+'\n'.join(t['role']+': '+t['content'][:500] for t in older[-16:])
        summary=summary[:6000] if older else ''
        cur.execute('INSERT INTO app_history.summaries(conversation_id,owner_id,revision,summary) VALUES (%s,%s,%s,%s) ON CONFLICT(conversation_id) DO UPDATE SET revision=EXCLUDED.revision,summary=EXCLUDED.summary,updated_at=now() WHERE app_history.summaries.owner_id=EXCLUDED.owner_id',(str(body.conversation_id),owner,row[1],summary))
        cur.execute('SELECT settings FROM app_history.preferences WHERE owner_id=%s',(owner,))
        pref=cur.fetchone()
        preferences=pref[0] if pref and isinstance(pref[0],dict) else json.loads(pref[0]) if pref else {}
    body.messages=[Message(**m) for m in recent]
    data={'owner_id':owner,'conversation_id':str(body.conversation_id),'summary':summary,
          'preferences':{'chart_type':preferences.get('chart_type','bar')},'expires_at':int(time.time())+300,
          'messages_sha256':hashlib.sha256(json.dumps(recent,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
    payload=base64.urlsafe_b64encode(json.dumps(data,separators=(',',':')).encode()).decode()
    key=os.environ['AGENT_CONTEXT_SIGNING_KEY']
    body._agent_context={'payload':payload,'signature':hmac.new(key.encode(),payload.encode(),hashlib.sha256).hexdigest()}
