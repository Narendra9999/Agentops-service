import json
from types import SimpleNamespace as NS
from langchain_core.messages import AIMessage, HumanMessage
from fe_multi_agent.platform_worker import PlatformKnowledgeWorker


def test_exhausted_worker_synthesizes_from_evidence_without_restarting_tools():
    worker=PlatformKnowledgeWorker.__new__(PlatformKnowledgeWorker)
    source='https://docs.databricks.com/aws/en/dev-tools/databricks-apps/'
    worker.sources=[source]
    worker.evidence=[{'text':'Apps run on serverless compute.','url':source}]
    sentinel=AIMessage(content='Sorry, need more steps to process this request.')
    graph_calls=[];model_calls=[]
    def graph(state,config):
        graph_calls.append(config)
        return {'messages':[*state['messages'],sentinel]}
    def synthesize(messages):
        model_calls.append(messages)
        return AIMessage(content='The retrieved Apps page does not cover serverless jobs. '+source)
    worker.graph=NS(invoke=graph)
    worker.model=NS(invoke=synthesize)
    question=HumanMessage(content='Explain serverless jobs')
    result=worker.invoke({'messages':[question]})
    assert len(graph_calls)==len(model_calls)==1
    assert model_calls[0][1] is question
    assert json.loads(model_calls[0][-1].content.removeprefix('Retrieved evidence: '))==worker.evidence
    assert result['messages'][-1].content.startswith('The retrieved Apps page')
    assert result['source_urls']==[source]


def test_completed_worker_does_not_make_an_extra_model_call():
    worker=PlatformKnowledgeWorker.__new__(PlatformKnowledgeWorker)
    worker.sources=['https://docs.databricks.com/example']
    worker.graph=NS(invoke=lambda state,config:{'messages':[AIMessage(content='Answer with citation')]})
    worker.model=NS(invoke=lambda _: (_ for _ in ()).throw(AssertionError('Unexpected synthesis')))
    assert worker.invoke({'messages':[]})['messages'][-1].content=='Answer with citation'
