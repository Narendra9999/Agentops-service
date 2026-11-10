import pytest
from fe_multi_agent.evaluation import response_text

PAYLOAD={'output':[{'id':'answer1','type':'message','role':'assistant',
    'content':[{'type':'output_text','text':'Case is investigating, version 2','annotations':[]}]}]}


def test_responses_dictionary_content_is_scored_as_text():
    assert response_text(PAYLOAD)=='Case is investigating, version 2'


def test_mlflow_responses_object_is_scored_as_text():
    pytest.importorskip('numpy')
    from mlflow.types.responses import ResponsesAgentResponse
    assert response_text(ResponsesAgentResponse(**PAYLOAD))=='Case is investigating, version 2'
