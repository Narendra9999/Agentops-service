"""Execute native graph routing without model credentials or remote services."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from fe_multi_agent.graph import Execution, build_supervisor_graph


class Router:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.calls = []

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.schema(**next(self.decisions))


class Worker:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def invoke(self, state):
        self.calls.append(state)
        return {"messages": [*state["messages"],
                             AIMessage(content="", tool_calls=[{"id": "t", "name": "lookup", "args": {}}]),
                             ToolMessage(content="private tool evidence", tool_call_id="t"),
                             AIMessage(content=self.answer)]}


def decision(route, task="delegated task", answer=""):
    return {"next_agent": route, "task": task, "answer": answer}


def setup(decisions, limit=6):
    router = Router(decisions)
    workers = {n: Worker(n + " evidence") for n in ("cost_estimation", "platform_docs")}
    graph = build_supervisor_graph(router, workers, "Coordinate workers", Execution("E", "T", "A"),
                                   max_delegations=limit)
    return router, workers, graph


def test_multiple_workers_then_synthesis_without_tool_history_leakage():
    router, workers, graph = setup([decision("cost_estimation", "Estimate cost"),
                                    decision("platform_docs", "Explain the cost evidence"),
                                    decision("finish", answer="Combined answer")])
    result = graph.invoke({"messages": [HumanMessage(content="Estimate and explain")]})
    assert result["messages"][-1].content == "Combined answer"
    assert len(workers["cost_estimation"].calls) == 1
    assert len(workers["platform_docs"].calls) == 1
    assert [m.name for m in result["messages"][1:]] == ["cost_estimation", "platform_docs", "supervisor"]
    assert all(m.type != "tool" for call in router.calls for m in call)
    assert all(call[-1].type == "human" for call in router.calls)
    assert "private tool evidence" not in str(workers["platform_docs"].calls)
    assert "Explain the cost evidence" in workers["platform_docs"].calls[0]["messages"][-1].content


def test_finish_without_delegation():
    _, workers, graph = setup([decision("finish", answer="Please provide the platform question")])
    result = graph.invoke({"messages": [HumanMessage(content="Help")]})
    assert result["messages"][-1].content == "Please provide the platform question"
    assert not any(w.calls for w in workers.values())


def test_delegation_can_omit_answer_and_finish_can_omit_task():
    _, workers, graph = setup([
        {"next_agent": "cost_estimation", "task": "Ask for missing runtime"},
        {"next_agent": "finish", "answer": "What is the runtime per batch?"},
    ])
    result = graph.invoke({"messages": [HumanMessage(content="Estimate 30 monthly runs")]})
    assert len(workers["cost_estimation"].calls) == 1
    assert not workers["platform_docs"].calls
    assert result["messages"][-1].content == "What is the runtime per batch?"


@pytest.mark.parametrize("payload", [
    {"next_agent": "cost_estimation"},
    {"next_agent": "finish"},
])
def test_missing_action_specific_field_still_rejects(payload):
    _, workers, graph = setup([payload])
    with pytest.raises(ValueError):
        graph.invoke({"messages": [HumanMessage(content="Help")]})
    assert not any(w.calls for w in workers.values())


def test_delegation_limit_stops_repeated_worker_calls():
    _, workers, graph = setup([decision("platform_docs")], limit=1)
    result = graph.invoke({"messages": [HumanMessage(content="Explain tracing")]})
    assert len(workers["platform_docs"].calls) == 1
    assert "limit" in result["messages"][-1].content


@pytest.mark.parametrize("route,task,answer", [("unregistered", "task", ""), ("disputes", "task", ""),
                                               ("platform_docs", "", ""), ("finish", "", "")])
def test_invalid_routing_does_not_invoke_workers(route, task, answer):
    _, workers, graph = setup([decision(route, task, answer)])
    with pytest.raises(ValueError):
        graph.invoke({"messages": [HumanMessage(content="Help")]})
    assert not any(w.calls for w in workers.values())


def test_prebuilt_worker_executes_tool_and_returns_final_answer():
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    calls = []

    @tool
    def lookup(query: str) -> str:
        """Return platform documentation for a query."""
        calls.append(query)
        return "Tracing documentation"

    class ToolModel(BaseChatModel):
        @property
        def _llm_type(self):
            return "local-contract-model"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if messages[-1].type == "tool":
                message = AIMessage(content="Tracing documentation")
            else:
                message = AIMessage(content="", tool_calls=[
                    {"id": "lookup1", "name": "lookup", "args": {"query": "MLflow"}}])
            return ChatResult(generations=[ChatGeneration(message=message)])

    worker = create_react_agent(ToolModel(), tools=[lookup], prompt="Retrieve platform documentation")
    result = worker.invoke({"messages": [HumanMessage(content="Explain MLflow")]})
    assert calls == ["MLflow"]
    assert result["messages"][-1].content == "Tracing documentation"


def test_completed_worker_cannot_be_selected_again():
    _, workers, graph = setup([decision('platform_docs'),decision('platform_docs')])
    with pytest.raises(ValueError):
        graph.invoke({'messages':[HumanMessage(content='Explain tracing')]})
    assert len(workers['platform_docs'].calls)==1


def test_retrieved_urls_survive_synthesis_even_when_model_omits_them():
    _, workers, graph = setup([decision('platform_docs'),decision('finish',answer='Tracing explains the calls')])
    original=workers['platform_docs'].invoke
    def retrieve(state):
        return {**original(state),'source_urls':['https://docs.databricks.com/aws/en/mlflow/']}
    workers['platform_docs'].invoke=retrieve
    result=graph.invoke({'messages':[HumanMessage(content='Explain tracing with citations')]})
    assert 'https://docs.databricks.com/aws/en/mlflow/' in result['messages'][-1].content


def test_progress_reports_actual_plan_and_worker_sequence_without_tool_evidence():
    from fe_multi_agent.progress import progress_sink
    events = []
    _, _, graph = setup([decision('platform_docs', 'Retrieve platform guidance'),
                         decision('finish', answer='Current status')])
    with progress_sink(events.append):
        graph.invoke({'messages': [HumanMessage(content='Explain tracing')]})
    assert [event['kind'] for event in events] == [
        'supervisor_started', 'plan', 'agent_started', 'agent_completed',
        'supervisor_started', 'synthesis_completed']
    assert events[1]['next_agent'] == 'platform_docs'
    assert events[1]['message'] == 'Retrieve platform guidance'
    assert 'private tool evidence' not in str(events)


def test_progress_sinks_do_not_leak_between_concurrent_requests():
    from concurrent.futures import ThreadPoolExecutor
    from fe_multi_agent.progress import progress_sink, emit
    from threading import Barrier
    barrier = Barrier(2)
    def request(name):
        events = []
        with progress_sink(events.append):
            barrier.wait()
            emit('agent_started', name, 'Calling specialist')
        emit('agent_completed', 'outside', 'No active sink')
        return events
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(request, ['first', 'second']))
    assert [event['agent'] for event in a] == ['first']
    assert [event['agent'] for event in b] == ['second']


def test_genie_evidence_survives_supervisor_synthesis():
    router, workers, graph = setup([decision('cost_estimation'), decision('finish', answer='Summary')])
    preview = {'space_id': 's', 'conversation_id': 'c', 'tables': [{'columns': ['cost'], 'rows': [['13.50']]}]}
    workers['cost_estimation'].invoke = lambda state: {
        'messages': [AIMessage(content='Query evidence')], 'genie_result': preview}
    result = graph.invoke({'messages': [HumanMessage(content='Estimate costs')]})
    assert result['genie_result'] == preview
    assert result['messages'][-1].content == 'Summary'
