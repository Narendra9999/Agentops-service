import ast
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse, ResponsesAgentStreamEvent

from framework.agent_base import AgentOPSBase
from framework.inference import extract_request_text, extract_response_text


class EchoAgent(AgentOPSBase):
    def _process_request(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        self.received = request
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=extract_request_text(request), id="answer")],
            custom_outputs={"source": "echo"},
        )


def make_agent():
    with patch.dict(os.environ, {"AGENTOPS_ENVIRONMENT": "test"}), patch(
        "framework.agent_base.SessionStore"
    ) as store:
        store.return_value.enabled = False
        store.return_value.memory_enabled = False
        store.return_value._store_error = None
        store.return_value._uc_last_error = None
        return EchoAgent({"guardrails": {"enabled": False}})


def test_responses_contract_preserves_team_hook_and_metadata():
    agent = make_agent()
    assert isinstance(agent, ResponsesAgent)
    response = agent.predict(ResponsesAgentRequest(
        input=[{"role": "user", "content": "hello"}],
        custom_inputs={"thread_id": "thread-1", "user_id": "user-1"},
    ))
    assert isinstance(response, ResponsesAgentResponse)
    assert response.output[0].content[0]["text"] == "hello"
    assert response.custom_outputs == {
        "thread_id": "thread-1", "user_id": "user-1", "source": "echo",
    }
    assert isinstance(agent.received, ResponsesAgentRequest)
    assert extract_request_text(agent.received) == "hello"
    assert agent.received.custom_inputs["thread_id"] == "thread-1"


def test_responses_pre_guardrail_blocks_before_team_hook():
    agent = make_agent()
    agent.guardrails_enabled = True
    agent.pre_llm_guardrails = MagicMock()
    agent.pre_llm_guardrails.check.return_value = {"blocked": True, "message": "blocked"}
    response = agent.predict(ResponsesAgentRequest(input=[{"role": "user", "content": "unsafe"}]))
    assert response.output[0].content[0]["text"] == "blocked"
    assert not hasattr(agent, "received")


def test_post_guardrail_replaces_unsafe_response():
    agent = make_agent()
    agent.guardrails_enabled = True
    agent.pre_llm_guardrails = MagicMock()
    agent.pre_llm_guardrails.check.return_value = {"blocked": False}
    agent.post_llm_guardrails = MagicMock()
    agent.post_llm_guardrails.check.return_value = {"blocked": True, "message": "filtered"}
    response = agent.predict(ResponsesAgentRequest(input=[{"role": "user", "content": "hello"}]))
    assert extract_response_text(response) == "filtered"


def test_non_streaming_hook_has_valid_streaming_fallback():
    agent = make_agent()
    events = list(agent.predict_stream({
        "input": [{"role": "user", "content": "hello"}],
        "custom_inputs": {"thread_id": "thread-1"},
    }))
    assert len(events) == 1
    assert events[0].type == "response.output_item.done"
    assert events[0].item["content"][0]["text"] == "hello"
    assert events[0].custom_outputs["thread_id"] == "thread-1"


class StreamingEchoAgent(EchoAgent):
    def _process_request_stream(self, request: ResponsesAgentRequest):
        for text in ("hel", "lo"):
            yield ResponsesAgentStreamEvent(type="response.output_text.delta", item_id="answer", delta=text)
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done", item=self.create_text_output_item(text="hello", id="answer"),
        )


def test_streaming_deltas_and_done_item_share_id_and_metadata():
    agent = make_agent()
    agent.__class__ = StreamingEchoAgent
    events = list(agent.predict_stream({
        "input": [{"role": "user", "content": "hello"}],
        "custom_inputs": {"thread_id": "thread-1", "user_id": "user-1"},
    }))
    assert [event.type for event in events] == [
        "response.output_text.delta", "response.output_text.delta", "response.output_item.done",
    ]
    assert events[0].item_id == events[1].item_id == events[-1].item["id"]
    assert events[-1].item["content"][0]["text"] == "hello"
    assert events[-1].custom_outputs == {"thread_id": "thread-1", "user_id": "user-1"}


def test_streaming_pre_guardrail_returns_completed_block_message():
    agent = make_agent()
    agent.__class__ = StreamingEchoAgent
    agent.guardrails_enabled = True
    agent.pre_llm_guardrails = MagicMock()
    agent.pre_llm_guardrails.check.return_value = {"blocked": True, "message": "blocked"}
    events = list(agent.predict_stream({"input": [{"role": "user", "content": "unsafe"}]}))
    assert events[-1].type == "response.output_item.done"
    assert events[-1].item["content"][0]["text"] == "blocked"
    assert events[-1].custom_outputs["thread_id"]


def test_response_text_excludes_tool_outputs():
    response = {
        "output": [
            {"type": "function_call_output", "call_id": "call-1", "output": "private"},
            ResponsesAgent.create_text_output_item(text="answer", id="answer"),
        ],
    }
    assert extract_response_text(response) == "answer"


def test_session_history_round_trip():
    agent = make_agent()
    agent.session_store.enabled = True
    agent.session_store.get_history.return_value = [{"role": "user", "content": "previous"}]
    agent.predict(ResponsesAgentRequest(
        input=[{"role": "user", "content": "hello"}],
        custom_inputs={"thread_id": "thread-1"},
    ))
    assert [message.content for message in agent.received.input] == ["previous", "hello"]
    saved = agent.session_store.save_full_session.call_args.kwargs
    assert saved["session_id"] == "thread-1"
    assert saved["response_text"] == "hello"


def load_wrapper(inner):
    import importlib

    with patch("mlflow.models.set_model"):
        module = importlib.import_module("framework.inference.cost_tracking_model")
    wrapper = module.CostTrackingResponsesModel()
    with patch.object(module, "_resolve_inner_model", return_value=inner):
        from types import SimpleNamespace
        wrapper.load_context(SimpleNamespace(model_config=getattr(inner, "config", {"guardrails": {"enabled": False}})))
    return wrapper


def test_wrapper_dispatches_responses_agent():
    wrapper = load_wrapper(make_agent())
    response = wrapper.predict({"input": [{"role": "user", "content": "hello"}]})
    assert extract_response_text(response) == "hello"
    events = list(wrapper.predict_stream({"input": [{"role": "user", "content": "hello"}]}))
    assert events[-1].type == "response.output_item.done"


def test_wrapper_adapts_plain_pyfunc_and_streams_done_event():
    from mlflow.pyfunc import PythonModel

    class PlainModel(PythonModel):
        def predict(self, context, model_input):
            assert model_input["custom_inputs"] == {"thread_id": "thread-1"}
            return model_input["input"][-1]["content"]

    wrapper = load_wrapper(PlainModel())
    request = {
        "input": [{"role": "user", "content": "hello"}],
        "custom_inputs": {"thread_id": "thread-1"},
    }
    assert extract_response_text(wrapper.predict(request)) == "hello"
    assert list(wrapper.predict_stream(request))[-1].item["content"][0]["text"] == "hello"


def test_mlflow_save_load_infers_responses_signature(tmp_path):
    import mlflow
    from types import SimpleNamespace

    model_path = str(tmp_path / "model")
    agent = make_agent()
    agent.session_store = SimpleNamespace(
        enabled=False, memory_enabled=False, _store_error=None, _uc_last_error=None,
    )
    mlflow.pyfunc.save_model(path=model_path, python_model=agent, pip_requirements=[])
    model = mlflow.pyfunc.load_model(model_path)
    assert "input" in model.metadata.signature.inputs.input_names()
    result = model.predict({"input": [{"role": "user", "content": "hello"}]})
    assert extract_response_text(result) == "hello"
    events = list(model.predict_stream({"input": [{"role": "user", "content": "hello"}]}))
    assert events[-1]["type"] == "response.output_item.done"


def test_structured_text_input_is_preserved_for_team_hook():
    agent = make_agent()
    request = ResponsesAgentRequest(input=[{
        "role": "user", "content": [{"type": "input_text", "text": "hello"}],
    }])
    response = agent.predict(request)
    assert extract_response_text(response) == "hello"
    assert agent.received.input == request.input


def test_tool_history_does_not_replace_user_input_for_guardrails():
    agent = make_agent()
    agent.guardrails_enabled = True
    agent.pre_llm_guardrails = MagicMock()
    agent.pre_llm_guardrails.check.return_value = {"blocked": False}
    agent.post_llm_guardrails = MagicMock()
    agent.post_llm_guardrails.check.return_value = {"blocked": False}
    agent.predict(ResponsesAgentRequest(input=[
        {"role": "user", "content": "hello"},
        {"type": "function_call", "id": "tool-1", "call_id": "call-1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call-1", "output": "tool result"},
    ]))
    assert agent.pre_llm_guardrails.check.call_args.args[0] == "hello"
    assert agent.received.input[-1].type == "function_call_output"
    assert agent.received.input[-1].call_id == "call-1"


def test_native_tool_and_reasoning_outputs_pass_through_unchanged():
    agent = make_agent()
    response = ResponsesAgentResponse(output=[
        {"type": "function_call", "id": "tool-1", "call_id": "call-1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call-1", "output": "found"},
        {"type": "reasoning", "id": "reasoning-1", "summary": []},
        agent.create_text_output_item(text="answer", id="answer"),
    ])
    with patch.object(agent, "_process_request", return_value=response):
        result = agent.predict(ResponsesAgentRequest(input=[{"role": "user", "content": "hello"}]))
    assert result.output == response.output
    assert [item.type for item in response.output] == ["function_call", "function_call_output", "reasoning", "message"]
    assert response.output[0].call_id == response.output[1].call_id == "call-1"
    assert extract_response_text(response) == "answer"


def test_monitoring_extracts_responses_and_legacy_payloads():
    assert extract_request_text({"input": [{
        "role": "user", "content": [{"type": "input_text", "text": "question"}],
    }]}) == "question"
    assert extract_request_text({"messages": [{"role": "user", "content": "question"}]}) == "question"
    assert extract_response_text({"messages": [{"role": "assistant", "content": "answer"}]}) == "answer"
    assert extract_response_text({"choices": [{"message": {"content": "answer"}}]}) == "answer"


def provider_response():
    return {
        "id": "response-1", "object": "response", "created_at": 0,
        "model": "databricks-gpt-oss-120b", "status": "completed", "error": None,
        "incomplete_details": None, "instructions": None, "metadata": {},
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "temperature": 0.1, "top_p": 1.0,
        "output": [{
            "type": "message", "id": "answer", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "answer", "annotations": [], "logprobs": []}],
        }],
        "usage": {
            "input_tokens": 10, "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 0},
            "output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 1},
            "total_tokens": 15,
        },
    }


def mock_provider(monkeypatch, agent, events=None):
    from openai import DefaultHttpxClient, OpenAI
    from openai.types.responses import Response
    try:
        import httpx2 as httpx
    except ImportError:
        import httpx

    payload = provider_response()
    Response.model_validate(payload)
    requests = []

    def respond(request):
        requests.append(request)
        if events is not None:
            content = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
            return httpx.Response(200, text=content + "data: [DONE]\n\n", headers={"Content-Type": "text/event-stream"})
        return httpx.Response(200, json=payload)

    client = OpenAI(
        api_key="local-test", base_url="https://example.test/serving-endpoints", max_retries=0,
        http_client=DefaultHttpxClient(transport=httpx.MockTransport(respond)),
    )
    workspace = MagicMock()
    workspace.serving_endpoints.get_open_ai_client.return_value = client
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda: workspace)
    agent.llm_endpoint = "databricks-gpt-oss-120b"
    agent.max_tokens = 2048
    agent.temperature = 0.1
    agent._request_context = {}
    return client, requests


@pytest.mark.parametrize("api", ["open-responses", "responses"])
def test_provider_uses_responses_transport_and_tracks_native_usage(monkeypatch, api):
    agent = make_agent()
    agent.config["llm"] = {"responses_api": api}
    client, requests = mock_provider(monkeypatch, agent)
    with client:
        response = agent._call_llm(ResponsesAgentRequest(
            input=[{"role": "system", "content": "system instruction"}, {"role": "user", "content": "question"}],
            custom_inputs={"thread_id": "private"}, max_output_tokens=128, temperature=0,
        ))
    assert requests[0].url.path == f"/serving-endpoints/{api}"
    body = json.loads(requests[0].content)
    assert body["input"] == [{"type": "message", "role": "user", "content": "question"}]
    assert body["instructions"] == "system instruction"
    assert body["max_output_tokens"] == 128
    assert body["temperature"] == 0
    assert "messages" not in body and "custom_inputs" not in body
    assert extract_response_text(response) == "answer"
    usage = agent._request_context["token_usage_history"][0]
    assert usage["prompt_tokens"] == 10 and usage["completion_tokens"] == 5
    assert usage["reasoning_tokens"] == 1 and usage["cached_tokens"] == 2


def test_provider_stream_forwards_native_events_and_tracks_final_usage(monkeypatch):
    payload = provider_response()
    events = [
        {"type": "response.created", "sequence_number": 0,
         "response": {**payload, "status": "in_progress", "output": []}},
        {"type": "response.in_progress", "sequence_number": 0,
         "response": {**payload, "status": "in_progress", "output": []}},
        {"type": "response.output_item.added", "sequence_number": 0,
         "output_index": 0, "item": payload["output"][0]},
        {"type": "response.output_text.delta", "sequence_number": 1, "item_id": "answer",
         "output_index": 0, "content_index": 0, "delta": "answer", "logprobs": []},
        {"type": "response.output_item.done", "sequence_number": 2, "output_index": 0, "item": payload["output"][0]},
        {"type": "response.completed", "sequence_number": 3, "response": payload},
    ]
    agent = make_agent()
    client, requests = mock_provider(monkeypatch, agent, events)
    with client:
        streamed = list(agent._call_llm_stream(ResponsesAgentRequest(input=[{"role": "user", "content": "question"}])))
    assert [event.type for event in streamed] == ["response.output_text.delta", "response.output_item.done"]
    assert streamed[0].item_id == streamed[1].item["id"] == "answer"
    assert json.loads(requests[0].content)["stream"] is True
    assert agent._request_context["token_usage_history"][0]["total_tokens"] == 15


def test_provider_stream_rejects_truncated_stream(monkeypatch):
    agent = make_agent()
    client, _ = mock_provider(monkeypatch, agent, [])
    with client, pytest.raises(RuntimeError, match="without a response.completed"):
        list(agent._call_llm_stream(ResponsesAgentRequest(input=[{"role": "user", "content": "question"}])))


def test_lakebase_preserves_native_tools_and_structured_text():
    from framework.session.session_store import SessionStore, _get_content

    store = SessionStore({"session_history": {"enabled": True, "lakebase": {"enabled": True}}})
    backend = MagicMock()
    messages = [
        {"role": "user", "content": [{"type": "input_text", "text": "question"}]},
        {"type": "function_call", "id": "tool-1", "call_id": "call-1", "name": "lookup",
         "arguments": "{}", "encrypted_content": "opaque"},
        {"type": "function_call_output", "call_id": "call-1", "output": "found"},
    ]
    output = [ResponsesAgent.create_text_output_item(text="answer", id="answer")]
    with patch.object(store, "_get_store", return_value=backend):
        store.save_full_session("thread-1", messages, "answer", 1, response_items=output)
        saved = backend.put.call_args.args[2]["messages"]
        assert saved == messages + output
        backend.get.return_value.value = {"messages": [{"role": "user", "content": "old"}] + saved}
        assert store._read_conversation("thread-1", max_turns=1) == saved
    assert _get_content(messages[0]) == "question"


@pytest.mark.parametrize("relative_path, class_name", [
    ("src/teams/platform-engineering-test/agent/Agent.py", "ChatModel"),
    ("src/agent_development/agent/notebooks/Agent.py", "DatabricksDocsAgent"),
])
def test_shipped_agent_hooks_accept_native_requests_and_preserve_tool_history(relative_path, class_name):
    source_path = Path(__file__).resolve().parents[2] / relative_path
    tree = ast.parse(source_path.read_text())
    definitions = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))]
    namespace = {"__name__": "native_agent_contract"}
    with patch.dict(sys.modules, {"tools.agent_tools": MagicMock(), "tools.tool_loader": MagicMock()}):
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source_path), "exec"), namespace)
    agent_class = namespace[class_name]
    agent = agent_class.__new__(agent_class)
    agent._request_context = {}
    agent.max_history_turns = 1
    agent.max_retrieval_turns = 1
    agent.system_prompt = "system instruction"
    agent._retrieve_context = MagicMock(return_value=("documentation", [{"text": "documentation"}]))
    agent._execute_tools = MagicMock(return_value="tool context")
    response = ResponsesAgentResponse(output=[agent.create_text_output_item(text="answer", id="answer")])
    agent._call_llm = MagicMock(return_value=response)
    request = ResponsesAgentRequest(input=[
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": [{"type": "input_text", "text": "question"}]},
        {"type": "function_call", "id": "tool-1", "call_id": "call-1", "name": "lookup",
         "arguments": "{}", "encrypted_content": "opaque"},
        {"type": "function_call_output", "call_id": "call-1", "output": "found"},
    ])
    assert agent._process_request(request) is response
    augmented = agent._call_llm.call_args.args[0]
    assert augmented.input[-2:] == request.input[-2:]
    assert extract_request_text(augmented) == "question"
    assert "old question" not in str(augmented.input)
    assert agent._request_context["retrieved_docs"] == [{"text": "documentation"}]
    event = ResponsesAgentStreamEvent(type="response.output_item.done", item=response.output[0].model_dump(exclude_none=True))
    agent._call_llm_stream = MagicMock(return_value=iter([event]))
    assert list(agent._process_request_stream(request)) == [event]


def test_cca_native_hook_preserves_clarification_history_and_streaming_fallback():
    import logging
    import mlflow

    source_path = Path(__file__).resolve().parents[2] / "src/agent_development/agent/notebooks/CCA_Agent.py"
    tree = ast.parse(source_path.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ChatModel")
    namespace = {
        "__name__": "cca_native_contract", "_ChatModelBase": AgentOPSBase, "mlflow": mlflow,
        "ResponsesAgentRequest": ResponsesAgentRequest, "ResponsesAgentResponse": ResponsesAgentResponse,
        "extract_request_text": extract_request_text, "extract_response_text": extract_response_text,
        "ROLE_USER": "user", "ROLE_ASSISTANT": "assistant",
        "UNKNOWN": "unknown", "logger": logging.getLogger(__name__), "initialize_logger_context": MagicMock(),
    }
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source_path), "exec"), namespace)
    agent = namespace["ChatModel"].__new__(namespace["ChatModel"])
    with patch.dict(os.environ, {"AGENTOPS_ENVIRONMENT": "test"}), patch("framework.agent_base.SessionStore") as store:
        store.return_value.enabled = False
        store.return_value.memory_enabled = False
        AgentOPSBase.__init__(agent, {"guardrails": {"enabled": False}})
    agent._get_current_user_info = MagicMock(return_value={"user_name": "test-user"})
    agent._get_request_id = MagicMock(return_value="cca-answer")
    agent.build_memory_from_messages = MagicMock()
    agent.build_memory_from_messages.return_value.load_memory_variables.return_value = {"chat_history": "earlier conversation"}
    agent.chatbot = MagicMock(return_value="CCA answer")
    request = ResponsesAgentRequest(input=[
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": [{"type": "output_text", "text": "Clarification Required: which table?"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "transactions"}]},
        {"type": "function_call_output", "call_id": "call-1", "output": "not the user question"},
    ])
    response = agent.predict(request)
    assert isinstance(response, ResponsesAgentResponse)
    assert extract_response_text(response) == "CCA answer"
    agent.chatbot.assert_called_once_with("transactions", "earlier conversation", {"awaiting_clarification": True})
    agent.build_memory_from_messages.assert_called_once_with([
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "Clarification Required: which table?"},
        {"role": "user", "content": "transactions"},
    ], k=10)
    events = list(agent.predict_stream(request))
    assert len(events) == 1 and events[0].type == "response.output_item.done"
    assert events[0].item["id"] == "cca-answer"
    assert events[0].item["content"][0]["text"] == "CCA answer"


@pytest.mark.parametrize('streaming', [False, True])
def test_provider_native_trace_metrics_match_agentops_once(tmp_path, monkeypatch, streaming):
    import mlflow
    from framework.observability import usage_tracking
    from agent_development.agent.tools.token_tracker import TokenTracker
    monkeypatch.setattr(usage_tracking, 'TokenTracker', TokenTracker)
    # File storage has no server cost aggregator; exercise the Databricks client path.
    monkeypatch.setattr('mlflow.tracing.processor.base_mlflow.should_compute_cost_client_side', lambda: True)
    # Databricks MLflow reprices at LiveSpan.end before user processors run.
    monkeypatch.setattr('mlflow.entities.span.should_compute_cost_client_side', lambda: True)
    monkeypatch.setattr('mlflow.entities.span.set_span_cost_attribute',
                        lambda span: span.set_attribute('mlflow.llm.cost',
                            {'input_cost': 1.0, 'output_cost': 2.0, 'total_cost': 3.0})
                        if span.span_type == 'LLM' else None)
    old_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri((tmp_path / 'mlruns').as_uri())
    mlflow.set_experiment('native-provider-metrics')
    try:
        agent = make_agent()
        payload = provider_response()
        events = [
            {'type': 'response.output_item.done', 'sequence_number': 0,
             'output_index': 0, 'item': payload['output'][0]},
            {'type': 'response.completed', 'sequence_number': 1, 'response': payload},
        ] if streaming else None
        client, requests = mock_provider(monkeypatch, agent, events)
        with client, mlflow.start_span(name='request', span_type='AGENT') as parent:
            if streaming:
                list(agent._call_llm_stream(ResponsesAgentRequest(input=[{'role': 'user', 'content': 'question'}])))
            else:
                agent._call_llm(ResponsesAgentRequest(input=[{'role': 'user', 'content': 'question'}]))
            assert parent.get_attribute('mlflow.chat.tokenUsage') is None
            assert parent.get_attribute('mlflow.llm.cost') is None
        mlflow.flush_trace_async_logging()
        trace = mlflow.get_trace(parent.trace_id)
        spans = [s for s in trace.data.spans if s.get_attribute('mlflow.chat.tokenUsage') is not None]
        assert len(requests) == len(spans) == 1
        assert spans[0].span_type == 'LLM'
        assert spans[0].get_attribute('mlflow.llm.model') == agent.llm_endpoint
        usage = spans[0].get_attribute('mlflow.chat.tokenUsage')
        assert usage == {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15,
                         'cached_tokens': 2, 'reasoning_tokens': 1}
        history = agent._request_context['token_usage_history']
        assert len(history) == 1 and history[0]['total_tokens'] == usage['total_tokens']
        cost = spans[0].get_attribute('mlflow.llm.cost')
        assert cost['total_cost'] == pytest.approx(history[0]['total_cost_usd'])
        # Includes two cached input tokens at the existing discounted rate.
        assert cost['total_cost'] == pytest.approx(0.000019)
        native_total = json.loads(trace.info.request_metadata['mlflow.trace.tokenUsage'])
        assert native_total['total_tokens'] == usage['total_tokens']
        native_cost = json.loads(trace.info.request_metadata['mlflow.trace.cost'])
        assert native_cost['total_cost'] == pytest.approx(cost['total_cost'])
    finally:
        mlflow.set_tracking_uri(old_uri)
        mlflow.tracing.reset()
