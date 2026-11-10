"""Standard AgentOPS team entry point; customer graph owns all routing."""
from copy import deepcopy
import os
import json
from pathlib import Path

import mlflow
from mlflow.pyfunc import ResponsesAgent
from framework.agent_base import AgentOPSBase
from fe_multi_agent.agent import PlatformEngineeringAgent, stream_prediction
from fe_multi_agent.config import Settings


class _GuardedTurn(AgentOPSBase):
    """One instance per request isolates mutable guardrail/session context."""
    def __init__(self, config, graph_agent):
        super().__init__(config)
        self.graph_agent = graph_agent

    def _process_request(self, request):
        response = self.graph_agent.predict(request)
        artifact = (response.custom_outputs or {}).get('genie_result')
        if artifact and self.guardrails_enabled:
            from framework.inference import extract_request_text
            checked = self.post_llm_guardrails.check(extract_request_text(request),json.dumps(artifact),self._request_context)
            if checked.get('blocked'):
                mlflow.update_current_trace(tags={'agentops.guardrail.artifacts.blocked':'true'})
                payload = response.model_dump(exclude_none=True)
                payload['custom_outputs'].pop('genie_result',None)
                payload['output']=[self.create_text_output_item(text=checked.get('message','Result filtered for safety.'),id='filtered-result')]
                return type(response)(**payload)
        return response


class ChatModel(ResponsesAgent):
    def load_context(self, context):
        self.config = deepcopy(context.model_config or {})
        block = self.config.get('multi_agent') or {}
        if block.get('pattern') != 'langgraph_supervisor':
            raise ValueError('multi_agent.pattern must be langgraph_supervisor')
        if set(block.get('agents', {})) != {'supervisor', 'platform_docs', 'cost_estimation'}:
            raise ValueError('Configure supervisor, platform_docs and cost_estimation only')
        values = dict(block.get('settings') or {})
        values['prompt_manifest'] = str(Path(__file__).resolve().parent.parent / 'configs/prompts.json')
        values['experiment_id'] = os.getenv('MLFLOW_EXPERIMENT_ID') or self.config.get('tracing', {}).get('experiment_id', '')
        self.graph_agent = PlatformEngineeringAgent(Settings(**values))
        self.graph_agent.settings.validate()
        self.graph_agent.load_context(context)
        mlflow.langchain.autolog()

    @mlflow.trace(name='agentops.platform_intelligence', span_type='AGENT')
    def predict(self, request):
        # Apply existing AgentOPS pre/post guardrails around the entire graph.
        # No shared _request_context across concurrent serving requests.
        return _GuardedTurn(self.config, self.graph_agent).predict(request)

    def predict_stream(self, request):
        # Public activity streams immediately; final output/Genie tables/usage are
        # released only after AgentOPS post-guardrails accept the final answer.
        yield from stream_prediction(self.predict, request)


mlflow.models.set_model(ChatModel())
