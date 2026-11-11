"""Opt-in request/response transformations selected by existing guardrail config.

This middleware owns the outer agent boundary, not customer model/tool internals.
It does not scrub trace payloads captured before this boundary.
"""
from copy import deepcopy
import re
import mlflow

from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from .pre_llm import PII_PATTERNS


class PIIRedactionMiddleware:
    """Redact text using the same PII patterns as the blocking input guardrail."""

    def __init__(self, config):
        self.pre = self._settings(config.get('pre_llm') or {}, 'pii')
        self.post = self._settings(config.get('post_llm') or {}, 'pii_leakage')

    @staticmethod
    def _settings(stage, check):
        policy = stage.get(check) or {}
        action = policy.get('action', 'block')
        if action not in {'block', 'redact'}:
            raise ValueError(f'{check}.action must be block or redact')
        selected = stage.get('enabled_checks')
        if action != 'redact' or (selected is not None and check not in selected):
            return ()
        entities = policy.get('entities', list(PII_PATTERNS))
        if not isinstance(entities, list) or not entities or any(name not in PII_PATTERNS for name in entities):
            raise ValueError(f'{check}.entities must select known PII entity names')
        return tuple((re.compile(PII_PATTERNS[name]), '[' + name.upper() + '-REDACTED]')
                     for name in dict.fromkeys(entities))

    @staticmethod
    def _tag(stage, changed):
        mlflow.update_current_trace(tags={
            f"agentops.guardrail.{stage}.pii.action": "redact",
            f"agentops.guardrail.{stage}.pii.redacted": str(changed).lower(),
        })

    @property
    def buffers_output(self):
        return bool(self.post)

    @staticmethod
    def _redact(value, patterns):
        if isinstance(value, str):
            for pattern, replacement in patterns:
                value = pattern.sub(replacement, value)
            return value
        if isinstance(value, list):
            return [PIIRedactionMiddleware._redact(item, patterns) for item in value]
        if isinstance(value, dict):
            return {key: PIIRedactionMiddleware._redact(item, patterns) for key, item in value.items()}
        return value

    async def process_request(self, request):
        if not self.pre:
            return request
        payload = request.model_dump(exclude_none=True)
        # Preserve roles, message IDs, opaque content and authentication metadata.
        for message in payload['input']:
            if 'content' in message:
                message['content'] = self._content(message['content'], self.pre)
        changed = payload['input'] != request.model_dump(exclude_none=True)['input']
        if not changed:
            self._tag('pre_llm', False)
            return request
        if (request.custom_inputs or {}).get('application_context') is not None:
            raise ValueError('Input redaction requires a team adapter to verify and rebind signed application context')
        self._tag('pre_llm', True)
        return ResponsesAgentRequest.model_validate(payload)

    @classmethod
    def _content(cls, content, patterns):
        if isinstance(content, str):
            return cls._redact(content, patterns)
        if isinstance(content, list):
            result = deepcopy(content)
            for part in result:
                if isinstance(part, dict) and part.get('type') in {'text', 'input_text', 'output_text'}:
                    if isinstance(part.get('text'), str):
                        part['text'] = cls._redact(part['text'], patterns)
            return result
        return content

    async def process_response(self, response):
        if not self.post:
            return response
        payload = response.model_dump(exclude_none=True)
        for item in payload['output']:
            if item.get('type') == 'message':
                item['content'] = self._content(item.get('content'), self.post)
        custom = payload.get('custom_outputs') or {}
        # Genie tables are user-visible content; usage, IDs and trace tags are not.
        artifact = custom.get('genie_result')
        if isinstance(artifact, dict):
            for field in ('tables', 'charts', 'text', 'summary'):
                if field in artifact:
                    artifact[field] = self._redact(artifact[field], self.post)
        self._tag('post_llm', payload != response.model_dump(exclude_none=True))
        return ResponsesAgentResponse.model_validate(payload)
