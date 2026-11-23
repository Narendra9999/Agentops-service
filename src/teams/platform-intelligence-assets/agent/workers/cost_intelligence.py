"""Connector to the existing Genie cost-intelligence agent."""
import json
import asyncio
from urllib.parse import quote

import httpx

from ._responses import answer_text


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
