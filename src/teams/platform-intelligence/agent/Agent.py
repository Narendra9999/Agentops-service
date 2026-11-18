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
        from framework.orchestration.config import validate_supervisor
        workers = validate_supervisor(block)
        values = dict(block.get('settings') or {})
        supervisor = block['supervisor']
        values['workers'] = workers
        values['supervisor_model'] = supervisor['model']
        values['model_transport'] = supervisor.get('api', self.config.get('llm', {}).get('api', 'model_serving'))
        instructions = supervisor.get('instructions')
        if instructions:
            team_root = Path(__file__).resolve().parent.parent
            path = (team_root / instructions).resolve()
            if not path.is_relative_to(team_root):
                raise ValueError('Supervisor instructions must be inside the team directory')
            values['supervisor_instructions'] = path.read_text()
        values['prompt_manifest'] = str(Path(__file__).resolve().parent.parent / 'configs/prompts.json')
        values['experiment_id'] = os.getenv('MLFLOW_EXPERIMENT_ID') or self.config.get('tracing', {}).get('experiment_id', '')
        self.settings = Settings(**values)
        self.settings.validate()
        super().load_context(context)
        mlflow.langchain.autolog()


mlflow.models.set_model(ChatModel())
