"""Text extraction shared by ResponsesAgent guardrails and evaluation."""
from collections.abc import Mapping
import json


def _value(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(exclude_none=True)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


def _content(value):
    value = _value(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts = []
        for part in value:
            part = _value(part)
            if isinstance(part, Mapping) and part.get('type') in {'text','input_text','output_text'}:
                text = part.get('text')
                if isinstance(text, str):
                    texts.append(text)
        return '\n'.join(texts)
    return ''


def extract_request_text(request):
    data = _value(request)
    if isinstance(data, str):
        return data
    if not isinstance(data, Mapping):
        return ''
    messages = data.get('input', data.get('messages', []))
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        for message in reversed(messages):
            message = _value(message)
            if isinstance(message, Mapping) and message.get('role') == 'user':
                return _content(message.get('content', ''))
    return ''


def extract_response_text(response):
    data = _value(response)
    if isinstance(data, str):
        return data
    if not isinstance(data, Mapping):
        return ''
    if isinstance(data.get('response'), Mapping):
        return extract_response_text(data['response'])
    messages = data.get('output', data.get('messages', []))
    if isinstance(messages, list):
        texts = []
        for message in messages:
            message = _value(message)
            if isinstance(message, Mapping) and message.get('role', 'assistant') == 'assistant':
                text = _content(message.get('content', ''))
                if text:
                    texts.append(text)
        if texts:
            return '\n'.join(texts)
    choices = data.get('choices') or []
    if choices:
        choice = _value(choices[0])
        if isinstance(choice, Mapping):
            message = _value(choice.get('message') or {})
            return _content(message.get('content', choice.get('text', '')))
    return _content(data.get('content', ''))
