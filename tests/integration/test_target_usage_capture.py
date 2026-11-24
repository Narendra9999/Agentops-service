"""Opaque team agents retain usage through sync/async and streaming boundaries."""
import asyncio
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse, ResponsesAgentStreamEvent

from framework.inference.agent_middleware import AgentOPSMiddleware


@pytest.mark.parametrize('streaming', [False, True])
def test_plain_target_captures_every_local_call_once(streaming):
    class TeamAgent(ResponsesAgent):
        def predict(self, request):
            model = FakeMessagesListChatModel(responses=[AIMessage(
                content='Databricks governs data.',
                usage_metadata={'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15},
                response_metadata={'model_name': 'databricks-claude-sonnet-4-6'})])
            model.invoke('First routing call')
            model.invoke('Final synthesis call')
            return ResponsesAgentResponse(output=[self.create_text_output_item(
                text='Databricks governs data.', id='answer')])

        def predict_stream(self, request):
            for item in self.predict(request).output:
                yield ResponsesAgentStreamEvent(type='response.output_item.done', item=item)

    async def run():
        config = {'llm': {'endpoint': 'databricks-claude-sonnet-4-6'},
                  'tracing': {'instrumentation': {'enabled': True, 'integration': 'langchain'}},
                  'guardrails': {'enabled': False}, 'session_history': {'enabled': False},
                  'long_term_memory': {'enabled': False}}
        with patch('framework.agent_base.configure_trace_destination'):
            middleware = AgentOPSMiddleware(config, TeamAgent())
        request = ResponsesAgentRequest(input=[{'role': 'user', 'content': 'Explain Databricks'}])
        if streaming:
            events = [event async for event in middleware.apredict_stream(request)]
            response = next(event.item for event in events if event.type == 'response.output_item.done')
            assert response
            # Final stream completion carries the same middleware custom outputs.
            outputs = next(event.custom_outputs for event in reversed(events) if event.custom_outputs)
        else:
            response = await middleware.apredict(request)
            outputs = response.custom_outputs
        tags = outputs['tags']
        assert tags['agentops.tokens.total'] == '30'
        assert tags['agentops.tokens.prompt'] == '20'
        assert tags['agentops.llm_requests'] == '2'
        assert float(tags['agentops.cost_usd']) > 0
        assert tags['agentops.usage.scope'] == 'observed_local_model_calls'
    asyncio.run(run())
