"""Genie STANDARD agent-mode SSE transport and bounded query-data presentation."""
import json
import time
from contextlib import closing
from requests import post
from requests.exceptions import Timeout


def sse_events(lines):
    parts=[]
    for line in lines:
        if isinstance(line,bytes):
            line=line.decode('utf-8')
        if not line:
            if parts:
                payload='\n'.join(parts);parts=[]
                if payload!='[DONE]':yield json.loads(payload)
        elif line.startswith('data:'):
            parts.append(line[5:].lstrip())
    if parts and '\n'.join(parts)!='[DONE]':yield json.loads('\n'.join(parts))


def markdown_tables(text):
    """Read returned SQL-result previews, never rerun SQL to obtain chart rows."""
    import re
    lines=text.splitlines();tables=[]
    def cells(line):
        # Escaped pipes are data rather than column delimiters.
        return [v.strip().replace(r'\|','|') for v in re.split(r'(?<!\\)\|',line.strip().strip('|'))]
    for i in range(len(lines)-1):
        if '|' not in lines[i]:continue
        columns=cells(lines[i]);separator=cells(lines[i+1])
        if len(columns)!=len(separator) or not all(re.fullmatch(r':?-{3,}:?',v) for v in separator):continue
        rows=[];j=i+2
        while j<len(lines) and lines[j].strip().startswith('|'):
            row=cells(lines[j])
            if len(row)!=len(columns):break
            rows.append([None if v.lower()=='null' else v[:2000] for v in row[:30]])
            j+=1
        if rows:
            title=next((v.strip().strip('*# ').strip() for v in reversed(lines[:i]) if v.strip()),'Genie query result')
            tables.append({'title':title[:200],'columns':[v[:200] for v in columns[:30]],
                           'rows':rows[:100],'truncated':len(rows)>100 or len(columns)>30,
                           'source':'genie_agent_query_preview'})
    return tables


def agent_presentation(space_id,response):
    tables=[];texts=[]
    for item in response.get('output',[]):
        if item.get('type')=='function_call_output':
            tables.extend(markdown_tables(str(item.get('output',''))))
        elif item.get('type')=='message' and item.get('role')=='assistant':
            texts.extend(c['text'] for c in item.get('content',[]) if c.get('type')=='output_text' and isinstance(c.get('text'),str))
    return {'answer':'\n'.join(texts),'presentation':{
        'space_id':space_id,'conversation_id':response.get('conversation_id'),
        'message_id':response.get('id'),'mode':'agent','execution_mode':'STANDARD',
        'tables':tables[:10],'has_visualization':False}}


def ask_agent(workspace,space_id,question):
    import mlflow
    from .progress import emit
    from .services import GenieWaitTimeout
    latest={'space_id':space_id,'status':'STARTING','mode':'agent'}
    started=time.monotonic();completed=None
    with mlflow.start_span(name='genie.agent.query',span_type='TOOL') as span:
        span.set_attributes({'genie.space_id':space_id,'genie.mode':'agent','genie.execution_mode':'STANDARD'})
        try:
            # One submission, no automatic retry or silent fallback to chat mode.
            with closing(post(workspace.config.host.rstrip('/')+'/api/2.0/genie/agents/'+space_id+'/responses',
                    headers={**workspace.config.authenticate(),'Content-Type':'application/json'},
                    json={'input':[{'type':'message','role':'user','content':[{'type':'input_text','text':question}]}],
                          'execution_mode':'STANDARD','enable_viz':True},
                    stream=True,timeout=(20,180))) as upstream:
                upstream.raise_for_status()
                if 'text/event-stream' not in upstream.headers.get('Content-Type','').lower():
                    raise RuntimeError('Genie agent endpoint did not return an event stream')
                upstream.encoding='utf-8'
                for event in sse_events(upstream.iter_lines(decode_unicode=True)):
                    response=event.get('response') or {}
                    latest.update({k:response[v] for k,v in [('conversation_id','conversation_id'),('message_id','id')] if response.get(v)})
                    kind=event.get('type');item=event.get('item') or {}
                    # Public execution status only; do not expose private reasoning text.
                    status=None
                    if kind=='response.created':status='AGENT_STARTED'
                    elif kind=='response.output_item.added' and item.get('type')=='function_call':status='EXECUTING_QUERY'
                    elif kind=='response.output_item.done' and item.get('type')=='function_call_output':status='QUERY_COMPLETED'
                    elif kind=='response.output_item.added' and item.get('type')=='message':status='PREPARING_ANSWER'
                    elif kind=='response.completed':status='COMPLETED';completed=response
                    elif kind=='response.failed':raise RuntimeError('Genie agent execution failed; inspect its conversation')
                    if status:
                        latest['status']=status
                        emit('genie_status','cost_estimation','Genie agent: '+status,tool='genie_agent_query',status=status)
                    if time.monotonic()-started>180:raise TimeoutError('Genie agent wait expired')
                    if completed is not None:break
                if completed is None:raise RuntimeError('Genie agent stream ended without a completed response')
        except (Timeout,TimeoutError) as exc:
            raise GenieWaitTimeout(dict(latest)) from exc
        finally:
            span.set_attributes({'genie.'+k:v for k,v in latest.items()})
        result=agent_presentation(space_id,completed)
        if not result['answer']:raise RuntimeError('Genie agent returned no assistant answer')
        return {'conversation_id':completed.get('conversation_id'),'message_id':completed.get('id'),
                'mode':'agent','answer':result['answer'],'presentation':result['presentation']}
