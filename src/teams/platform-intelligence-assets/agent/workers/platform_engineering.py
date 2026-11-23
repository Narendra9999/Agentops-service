"""Connector to the existing Platform Engineering serving endpoint."""
import asyncio
from urllib.parse import quote

import httpx

from ._responses import answer_text


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


