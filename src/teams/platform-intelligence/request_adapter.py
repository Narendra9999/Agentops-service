"""Authenticated memory request contract for the original shared pipeline."""
import hashlib
import json
import os
import sys
import time
import uuid


def prepare(config, dbutils, team):
    path = str(team / 'agent')
    if path not in sys.path:
        sys.path.insert(0, path)
    secret = config['pipeline']['context_signing_secret']
    os.environ['AGENT_CONTEXT_SIGNING_KEY'] = dbutils.secrets.get(secret['scope'], secret['key'])


def build_request(query, owner):
    from fe_multi_agent.memory import sign_context
    if not owner.startswith('evaluation:'):
        raise ValueError('Pipeline memory requires an isolated evaluation owner')
    messages = [{'role': 'user', 'content': query}]
    context = {'owner_id': owner, 'conversation_id': str(uuid.uuid4()),
               'summary': '', 'preferences': {}, 'expires_at': int(time.time()) + 300,
               'messages_sha256': hashlib.sha256(json.dumps(
                   messages, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
    return {'input': messages, 'custom_inputs': {'application_context': sign_context(context)}}


def build_input_example(query, config):
    settings = config.get('multi_agent', {}).get('settings', {})
    if settings.get('memory_enabled') and settings.get('unsigned_context_policy', 'reject') != 'stateless':
        return None  # Signed-only teams must obtain fresh context from their authenticated App.
    return {'input': [{'role': 'user', 'content': query}]}
