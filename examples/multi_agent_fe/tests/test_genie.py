from types import SimpleNamespace
import pytest
from fe_multi_agent.services import GenieCostService, GenieWaitTimeout, genie_presentation


def test_genie_timeout_keeps_original_conversation_without_restart(monkeypatch):
    from fe_multi_agent import genie_agent
    from requests.exceptions import ReadTimeout
    import json
    calls=[]
    class Response:
        headers={'Content-Type':'text/event-stream'}
        def close(self):pass
        def raise_for_status(self):pass
        def iter_lines(self,**kwargs):
            yield 'data: '+json.dumps({'type':'response.created','response':{'conversation_id':'c','id':'m'}})
            yield ''
            raise ReadTimeout('wait expired')
    def post(*args,**kwargs):calls.append(kwargs);return Response()
    monkeypatch.setattr(genie_agent,'post',post)
    workspace=SimpleNamespace(config=SimpleNamespace(host='https://workspace',authenticate=lambda:{}))
    with pytest.raises(GenieWaitTimeout) as error:
        GenieCostService(workspace,'s').ask('Do not change assumptions')
    assert error.value.details=={'space_id':'s','conversation_id':'c','message_id':'m','status':'AGENT_STARTED','mode':'agent'}
    assert len(calls)==1 and calls[0]['json']['execution_mode']=='STANDARD'
    assert calls[0]['json']['enable_viz'] is True


def test_query_preview_preserves_columns_nulls_and_limits_without_thoughts():
    message = SimpleNamespace(conversation_id='c', id='m', message_id='m')
    attachments = [{'query': {'title': 'Costs', 'thoughts': ['private']}, 'query_result': {
        'statement_response': {'manifest': {'schema': {'columns': [{'name': 'cost'}, {'name': 'notes'}]}},
                               'result': {'data_array': [['1.20', None]] * 101}}}},
                   {'viz': {'query_attachment_id': 'q'}}]
    preview = genie_presentation('s', message, attachments)
    assert preview['tables'][0]['columns'] == ['cost', 'notes']
    assert preview['tables'][0]['rows'][0] == ['1.20', None]
    assert len(preview['tables'][0]['rows']) == 100 and preview['tables'][0]['truncated']
    assert preview['has_visualization']
    assert 'private' not in str(preview)



def test_agent_result_tables_and_answer_exclude_private_reasoning():
    from fe_multi_agent.genie_agent import agent_presentation, sse_events
    response={'id':'m','conversation_id':'c','output':[
        {'type':'reasoning','content':[{'text':'private chain of thought'}]},
        {'type':'function_call_output','output':'**Cost Estimate**\n\n| scenario | annual_usd | notes |\n| --- | ---: | --- |\n| LOW | 12.50 | null |\n| HIGH | 25.00 | a \\| b |'},
        {'type':'message','role':'assistant','content':[{'type':'output_text','text':'Cost estimate ready'}]}]}
    result=agent_presentation('s',response)
    assert result['answer']=='Cost estimate ready'
    assert 'private chain' not in str(result)
    table=result['presentation']['tables'][0]
    assert table['columns']==['scenario','annual_usd','notes']
    assert table['rows']==[['LOW','12.50',None],['HIGH','25.00','a | b']]
    assert result['presentation']['mode']=='agent'
    assert list(sse_events(['event: response.created','data: {"type":','data: "response.created"}','','data: [DONE]','']))==[{'type':'response.created'}]


def test_agent_stream_requires_terminal_success_and_never_falls_back(monkeypatch):
    from fe_multi_agent import genie_agent
    class Response:
        headers={'Content-Type':'text/event-stream'}
        def close(self):pass
        def raise_for_status(self):pass
        def iter_lines(self,**kwargs):yield from []
    calls=[]
    monkeypatch.setattr(genie_agent,'post',lambda *a,**k: (calls.append(k) or Response()))
    workspace=SimpleNamespace(config=SimpleNamespace(host='https://workspace',authenticate=lambda:{}))
    with pytest.raises(RuntimeError,match='without a completed'):
        GenieCostService(workspace,'s').ask('cost')
    assert len(calls)==1
