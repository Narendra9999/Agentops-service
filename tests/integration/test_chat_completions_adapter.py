"""Chat targets run inside the same middleware without ResponsesAgent inheritance."""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from framework.inference.agent_middleware import AgentOPSMiddleware
from framework.inference.chat_completions import chat_messages
from framework.inference.request_context import RequestContext

CONFIG = {'guardrails': {'enabled': False}, 'inference': {'target_protocol': 'chat_completions'}}

def request(text='Explain Databricks'):
    return ResponsesAgentRequest(input=[{'role': 'user', 'content': text}])

def complete(content='answer', usage=True):
    return {'model': 'databricks-gpt-oss-120b', 'choices': [{'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}],
            **({'usage': {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13}} if usage else {})}

@pytest.fixture(autouse=True)
def no_remote_trace():
    with patch('mlflow.update_current_trace'):
        yield

@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('streaming', [False, True])
def test_chat_target_in_shared_lifecycle(native, streaming):
    seen=[]
    def invoke(messages, custom_inputs):
        seen.append(messages)
        return NS(**complete())
    def stream(messages, custom_inputs):
        seen.append(messages)
        yield {'choices': [{'index': 0, 'delta': {'content': 'an'}, 'finish_reason': None}]}
        yield {'choices': [{'index': 0, 'delta': {'content': 'swer'}, 'finish_reason': 'stop'}]}
        yield {'choices': [], 'model':'databricks-gpt-oss-120b', 'usage': complete()['usage']}
    class Sync:
        predict = staticmethod(invoke)
        predict_stream = staticmethod(stream)
    class Native:
        async def apredict(self, **kwargs): return invoke(**kwargs)
        async def apredict_stream(self, **kwargs):
            for chunk in stream(**kwargs): yield chunk
    target=Native() if native else Sync()
    m=AgentOPSMiddleware(CONFIG, target)
    ctx=RequestContext()
    async def run():
        if streaming:
            events=[e async for e in m.execute_stream(request(), invoke=m.adapter.invoke, stream=m.adapter.stream, context=ctx)]
            assert [e.type for e in events]==['response.output_item.done']
            assert events[0].item['content'][0]['text']=='answer'
            assert events[0].usage['total_tokens']==13
        else:
            result=await m.execute(request(), invoke=m.adapter.invoke, context=ctx)
            ResponsesAgentResponse.model_validate(result)
            assert result.output[0].content[0]['text']=='answer'
            assert result.usage.total_tokens==13
    asyncio.run(run())
    assert len(seen)==1 and seen[0][0]['role']=='user'
    assert len(ctx.state['token_usage_history'])==1
    assert ctx.resources is None


def test_stream_function_arguments_are_reassembled_and_not_executed():
    class Target:
        async def apredict_stream(self, **kwargs):
            yield {'choices':[{'delta':{'tool_calls':[{'index':0,'id':'call_1','function':{'name':'lookup','arguments':'{"x":'}}]}}]}
            yield {'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'arguments':'1}'}}]},'finish_reason':'tool_calls'}]}
    m=AgentOPSMiddleware(CONFIG,Target())
    async def run(): return [e async for e in m.apredict_stream(request())]
    events=asyncio.run(run())
    assert len(events)==1
    assert events[0].item['type']=='function_call'
    assert events[0].item['arguments']=='{"x":1}'
    assert events[0].item['call_id']=='call_1'

@pytest.mark.parametrize('reason', ['length','content_filter',None])
def test_incomplete_stream_cannot_be_returned_as_success(reason):
    closed=[]
    class Target:
        async def apredict_stream(self, **kwargs):
            try:
                yield {'choices':[{'delta':{'content':'partial'},'finish_reason':reason}]}
            finally: closed.append(True)
    m=AgentOPSMiddleware(CONFIG,Target())
    async def run():
        events=[]
        with pytest.raises(RuntimeError,match='did not complete'):
            async for e in m.apredict_stream(request()): events.append(e)
        assert events==[]
    asyncio.run(run()); assert closed==[True]


def test_function_history_and_images_translate_without_silent_loss():
    req=ResponsesAgentRequest(input=[
        {'type':'function_call','call_id':'call1','name':'lookup','arguments':'{}'},
        {'type':'function_call_output','call_id':'call1','output':'found'},
        {'role':'user','content':[{'type':'input_text','text':'describe'}, {'type':'input_image','image_url':'https://example.com/image.png'}]}])
    messages=chat_messages(req)
    assert messages[0]['tool_calls'][0]['id']=='call1'
    assert messages[1]=={'role':'tool','tool_call_id':'call1','content':'found'}
    assert messages[2]['content'][1]['type']=='image_url'


def test_public_target_error_is_not_retried():
    calls=[]
    class Target:
        def predict(self, **kwargs):
            calls.append(True); raise TypeError('original target failure')
    m=AgentOPSMiddleware(CONFIG,Target())
    with pytest.raises(TypeError,match='original target failure'): m.predict(request())
    assert calls==[True]


def test_input_and_output_checks_apply_to_chat_targets():
    calls=[]
    class Target:
        def predict(self, **kwargs): calls.append('target'); return complete('sensitive answer')
    class Guard:
        def __init__(self, stage, block): self.stage,self.block=stage,block
        async def acheck(self,*a,**kw):
            calls.append(self.stage); return {'blocked':self.block,'message':'filtered'}
    m=AgentOPSMiddleware(CONFIG,Target());m.guardrails_enabled=True
    m.pre_llm_guardrails=Guard('pre',True);m.post_llm_guardrails=Guard('post',False)
    assert m.predict(request()).output[0].content[0]['text']=='filtered';assert calls==['pre']
    calls.clear();m.pre_llm_guardrails=Guard('pre',False);m.post_llm_guardrails=Guard('post',True)
    assert m.predict(request()).output[0].content[0]['text']=='filtered';assert calls==['pre','target','post']
