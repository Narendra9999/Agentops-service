"""Customer supervisor target. AgentOPS serving wrapper owns all middleware."""
import os
from pathlib import Path

import mlflow
from fe_multi_agent.supervisor_agent import SupervisorAgent
from fe_multi_agent.config import Settings


class ChatModel(SupervisorAgent):
    def load_context(self, context):
        self.config = dict(context.model_config or {})
        block = self.config.get('multi_agent') or {}
        if block.get('pattern') != 'langgraph_supervisor':
            raise ValueError('multi_agent.pattern must be langgraph_supervisor')
        if set(block.get('agents', {})) != {'supervisor', 'platform_docs', 'cost_estimation'}:
            raise ValueError('Configure supervisor, platform_docs and cost_estimation only')
        values = dict(block.get('settings') or {})
        llm = self.config.get('llm') or {}
        values['supervisor_model'] = llm['endpoint']
        values['model_transport'] = llm.get('api', 'model_serving')
        values['prompt_manifest'] = str(Path(__file__).resolve().parent.parent / 'configs/prompts.json')
        values['experiment_id'] = os.getenv('MLFLOW_EXPERIMENT_ID') or self.config.get('tracing', {}).get('experiment_id', '')
        self.settings = Settings(**values)
        self.settings.validate()
        super().load_context(context)
        mlflow.langchain.autolog()


mlflow.models.set_model(ChatModel())
