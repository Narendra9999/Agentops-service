"""Team-owned calls to existing workers; no worker deployment or evaluation logic."""
import asyncio
import json
from urllib.parse import quote

import httpx


def answer_text(response):
    return '\n'.join(part.get('text', '') for item in response.get('output', [])
                     if item.get('type') == 'message' and item.get('role') == 'assistant'
                     for part in item.get('content', []) if part.get('type') == 'output_text')


async def call_platform(workspace, endpoint, task):
    headers = await asyncio.to_thread(workspace.config.authenticate)
    async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=20)) as client:
        response = await client.post(
            workspace.config.host.rstrip('/') + '/serving-endpoints/' + quote(endpoint, safe='') + '/invocations',
            headers=headers, json={'input': [{'role': 'user', 'content': task}]})
        response.raise_for_status()
        text = answer_text(response.json())
    if not text.strip():
        raise RuntimeError('Platform worker returned no Responses-compatible answer')
    return text


async def call_genie(workspace, space_id, task):
    """One STANDARD agent-mode submission; never retry or fall back to chat mode."""
    headers = await asyncio.to_thread(workspace.config.authenticate)
    body = {'input': [{'type': 'message', 'role': 'user',
                      'content': [{'type': 'input_text', 'text': task}]}],
            'execution_mode': 'STANDARD', 'stream': True}
    async with asyncio.timeout(180), httpx.AsyncClient(timeout=httpx.Timeout(180, connect=20)) as client:
        async with client.stream('POST', workspace.config.host.rstrip('/') +
                                 '/api/2.0/genie/agents/' + quote(space_id, safe='') + '/responses',
                                 headers=headers, json=body) as response:
            response.raise_for_status()
            if 'text/event-stream' not in response.headers.get('content-type', '').lower():
                raise RuntimeError('Genie did not return an agent-mode event stream')
            parts = []
            async for line in response.aiter_lines():
                if line.startswith('data:'):
                    parts.append(line[5:].lstrip())
                elif not line and parts:
                    data = '\n'.join(parts)
                    parts = []
                    if data == '[DONE]':
                        break
                    event = json.loads(data)
                    if event.get('type') in {'response.failed', 'error'}:
                        raise RuntimeError('Genie agent execution failed')
                    if event.get('type') == 'response.completed':
                        text = answer_text(event.get('response') or {})
                        if not text.strip():
                            raise RuntimeError('Genie returned no assistant answer')
                        return text
    raise RuntimeError('Genie stream ended without a completed response')
