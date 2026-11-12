"""Explicit Chat Completions target protocol; orchestration stays with the target."""
from contextlib import aclosing
from inspect import isasyncgenfunction
import uuid

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentResponse, ResponsesAgentStreamEvent
from framework.inference.async_runtime import call_async_method, iterate_blocking
from framework.observability.token_usage import as_dict, normalize_usage
from framework.observability.usage_tracking import record_response_usage


def chat_messages(request):
    """Translate supported Responses messages and function handoffs; fail explicitly otherwise."""
    data = request.model_dump(exclude_none=True)
    messages = []
    if data.get('instructions'):
        messages.append({'role': 'system', 'content': data['instructions']})
    for item in data['input']:
        kind = item.get('type', 'message')
        if kind == 'function_call':
            messages.append({'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': item['call_id'], 'type': 'function',
                'function': {'name': item['name'], 'arguments': item['arguments']}}]})
        elif kind == 'function_call_output':
            messages.append({'role': 'tool', 'tool_call_id': item['call_id'], 'content': item['output']})
        elif kind == 'message' and item.get('role') in {'system', 'developer', 'user', 'assistant', 'tool'}:
            content = item.get('content', '')
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get('type') in {'text', 'input_text', 'output_text'}:
                        parts.append({'type': 'text', 'text': part['text']})
                    elif part.get('type') == 'input_image':
                        image = {'url': part['image_url']}
                        if part.get('detail'): image['detail'] = part['detail']
                        parts.append({'type': 'image_url', 'image_url': image})
                    else:
                        raise ValueError('Unsupported Chat Completions input content: ' + str(part.get('type')))
                content = parts
            messages.append({'role': item['role'], 'content': content})
        else:
            raise ValueError('Unsupported Chat Completions input item: ' + str(kind))
    return messages


def _finish(reason):
    if reason not in {'stop', 'tool_calls', 'function_call'}:
        raise RuntimeError('Chat Completions target did not complete successfully: ' + str(reason))


def _items(message):
    output = []
    if message.get('content'):
        if not isinstance(message['content'], str):
            raise ValueError('Chat Completions output content must be text')
        output.append(ResponsesAgent.create_text_output_item(text=message['content'], id=str(uuid.uuid4())))
    if message.get('refusal'):
        raise RuntimeError('Chat Completions target refused the request')
    for call in message.get('tool_calls') or []:
        if call.get('type', 'function') != 'function':
            raise ValueError('Unsupported Chat Completions tool type')
        function = call.get('function') or {}
        if not call.get('id') or not function.get('name'):
            raise ValueError('Incomplete Chat Completions function call')
        output.append({'type': 'function_call', 'id': str(uuid.uuid4()),
                       'call_id': call['id'], 'name': function['name'],
                       'arguments': function.get('arguments', '')})
    if not output:
        raise RuntimeError('Chat Completions target returned no output')
    return output


class ChatCompletionsAdapter:
    """Targets implement predict/apredict(messages=..., custom_inputs=...).

    Streaming methods accept the same arguments and return decoded completion
    chunks (dicts or SDK objects). The adapter does not run a tool-execution loop.
    """
    def __init__(self, target, config):
        self.target, self.config = target, config
        self.max_stream_chars = config.get('inference', {}).get('max_stream_chars', 1_000_000)

    @property
    def supports_streaming(self):
        return callable(getattr(self.target, 'apredict_stream', None)) or callable(getattr(self.target, 'predict_stream', None))

    def _usage(self, usage, model, context):
        normalized = normalize_usage(usage)
        if normalized is not None:
            record_response_usage({'usage': usage}, context.state,
                                  model or self.config.get('llm', {}).get('endpoint', ''))
        if normalized is None:
            return None
        return {k: normalized[k] for k in ('input_tokens', 'output_tokens', 'total_tokens')} | {
            'input_tokens_details': {'cached_tokens': normalized['cached_tokens']},
            'output_tokens_details': {'reasoning_tokens': normalized['reasoning_tokens']}}

    async def invoke(self, request, context):
        raw = as_dict(await call_async_method(self.target, 'apredict', 'predict',
            messages=chat_messages(request), custom_inputs=request.custom_inputs or {}))
        if raw.get('error'): raise RuntimeError('Chat Completions target failed')
        choices = raw.get('choices') or []
        if len(choices) != 1: raise ValueError('Chat Completions adapter requires exactly one choice')
        choice = as_dict(choices[0]); _finish(choice.get('finish_reason'))
        usage = self._usage(raw.get('usage'), raw.get('model'), context)
        return ResponsesAgentResponse(output=_items(as_dict(choice.get('message'))),
            custom_outputs=raw.get('custom_outputs'), **({'usage': usage} if usage is not None else {}))

    async def stream(self, request, context):
        kwargs = {'messages': chat_messages(request), 'custom_inputs': request.custom_inputs or {}}
        native = getattr(self.target, 'apredict_stream', None)
        if isasyncgenfunction(native):
            upstream = native(**kwargs)
        elif native is not None:
            upstream = await call_async_method(self.target, 'apredict_stream', 'predict_stream', **kwargs)
        else:
            upstream = iterate_blocking(self.target.predict_stream, **kwargs)
        text, calls, usage, model, finish, size = [], {}, None, None, None, 0
        text_id = str(uuid.uuid4())
        custom = {}
        async with aclosing(upstream):
            async for chunk in upstream:
                raw = as_dict(chunk)
                if raw.get('error'): raise RuntimeError('Chat Completions target stream failed')
                custom.update(as_dict(raw.get('custom_outputs')))
                model = raw.get('model') or model
                if raw.get('usage') is not None: usage = raw['usage']
                choices = raw.get('choices') or []
                if not choices: continue  # usage-only terminal chunk
                if len(choices) != 1 or as_dict(choices[0]).get('index', 0) != 0:
                    raise ValueError('Chat Completions adapter requires exactly one choice')
                if finish is not None: raise RuntimeError('Content received after Chat Completions finish')
                choice = as_dict(choices[0]); delta = as_dict(choice.get('delta'))
                if delta.get('refusal'): raise RuntimeError('Chat Completions target refused the request')
                if delta.get('content'):
                    value = delta['content']
                    if not isinstance(value, str): raise ValueError('Stream content must be text')
                    text.append(value); size += len(value)
                    yield ResponsesAgentStreamEvent(type='response.output_text.delta', item_id=text_id, delta=value)
                for fragment in delta.get('tool_calls') or []:
                    fragment = as_dict(fragment); index = fragment['index']
                    call = calls.setdefault(index, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                    if fragment.get('type', 'function') != 'function': raise ValueError('Unsupported tool type')
                    call['id'] += fragment.get('id') or ''
                    function = as_dict(fragment.get('function'))
                    for key in ('name', 'arguments'):
                        value = function.get(key) or ''; call['function'][key] += value; size += len(value)
                if size > self.max_stream_chars: raise RuntimeError('Chat Completions stream exceeds configured size limit')
                if choice.get('finish_reason') is not None:
                    finish = choice['finish_reason']; _finish(finish)
        _finish(finish)
        items = _items({'content': ''.join(text), 'tool_calls': [calls[k] for k in sorted(calls)]})
        if text: items[0]['id'] = text_id
        normalized = self._usage(usage, model, context)
        for index, item in enumerate(items):
            yield ResponsesAgentStreamEvent(type='response.output_item.done', item=item,
                custom_outputs=custom if index == len(items)-1 else None,
                **({'usage': normalized} if normalized is not None and index == len(items)-1 else {}))
