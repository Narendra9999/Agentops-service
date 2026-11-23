"""Real MLflow packaging preserves valid relative asset paths in unchanged code."""
from pathlib import Path
import json
import shutil
import sys

import mlflow
import pytest

from framework.inference.packaging import team_code_paths


@pytest.mark.parametrize('nested', [False, True])
def test_saved_model_loads_unchanged_customer_assets_without_source_checkout(tmp_path, nested):
    team = tmp_path / 'team'
    agent = team / 'agent'
    agent.mkdir(parents=True)
    configs = team / 'configs'
    configs.mkdir()
    (team / 'config.yaml').write_text('agent:\n  name: packaging_probe\n')
    (configs / 'prompts.json').write_text(json.dumps({'prompt': 'original prompt'}))
    if nested:
        implementation = agent / 'packaging_probe_nested'
        implementation.mkdir()
        (implementation / '__init__.py').write_text('')
        (implementation / 'assets').mkdir()
        (implementation / 'assets/prompts.json').write_text(json.dumps({'prompt': 'original prompt'}))
        prompt_expression = "Path(__file__).resolve().parent / 'assets/prompts.json'"
        (agent / 'Agent.py').write_text('from packaging_probe_nested.supervisor import ChatModel\n')
        target_file = implementation / 'supervisor.py'
    else:
        target_file = agent / 'Agent.py'
        prompt_expression = "Path(__file__).resolve().parent.parent / 'configs/prompts.json'"
    source = f'''import json
from pathlib import Path
class ChatModel:
    def load_context(self, context):
        self.prompt = json.loads(({prompt_expression}).read_text())["prompt"]
    def predict(self, value):
        return self.prompt
'''
    target_file.write_text(source)
    # A tiny generic loader exercises MLflow staging and the real AgentOPS target
    # resolver; it adds no path-handling code to the customer's implementation.
    loader = tmp_path / 'probe_loader.py'
    loader.write_text('''import mlflow
from mlflow.pyfunc import PythonModel
from framework.inference.cost_tracking_model import _resolve_inner_model
class Probe(PythonModel):
    def load_context(self, context):
        self.target = _resolve_inner_model(context.model_config)
        self.target.load_context(context)
    def predict(self, context, model_input, params=None):
        return self.target.predict(model_input)
mlflow.models.set_model(Probe())
''')
    destination = tmp_path / 'saved_model'
    paths = team_code_paths(str(agent), str(team / 'config.yaml'))
    try:
        mlflow.pyfunc.save_model(path=str(destination), python_model=str(loader),
            code_paths=paths, model_config={'inference': {'target_class': 'ChatModel'}},
            pip_requirements=['mlflow'])
        staged = destination / 'code/agent' / target_file.relative_to(agent)
        assert staged.read_text() == source
        assert (destination / 'code/configs/prompts.json').is_file()
        shutil.rmtree(team)
        loaded = mlflow.pyfunc.load_model(str(destination))
        assert loaded.predict('hello') == 'original prompt'
    finally:
        # MLflow deliberately adds staged imports process-wide. Keep each test
        # independent so imports cannot accidentally reuse another model's code.
        sys.path[:] = [p for p in sys.path if not str(p).startswith(str(destination))]
        for name in list(sys.modules):
            if name == '_team_agent' or name.startswith('packaging_probe_nested'):
                sys.modules.pop(name, None)


def test_optional_assets_and_legacy_subpackage_aliases(tmp_path):
    agent = tmp_path / 'agent'; agent.mkdir()
    for name in ['workers', '.cache', '__pycache__']:
        (agent / name).mkdir()
    paths = team_code_paths(str(agent), str(tmp_path / 'missing.yaml'))
    assert paths == [str(agent), str(agent / 'workers')]
