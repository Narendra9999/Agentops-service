"""Real MLflow packaging preserves valid relative asset paths in unchanged code."""
from pathlib import Path
import json
import shutil
import runpy
import sys

import mlflow
import pytest

from framework.inference.packaging import team_code_paths


@pytest.mark.parametrize('nested,mapped', [(False, False), (True, False), (True, True)])
def test_saved_model_loads_unchanged_customer_assets_without_source_checkout(tmp_path, nested, mapped):
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
        (agent / '__init__.py').write_text('')
        if mapped:
            prompt_expression = "Path(__file__).resolve().parent.parent / 'configs/prompts.json'"
        module = 'agent.packaging_probe_nested' if mapped else 'packaging_probe_nested'
        (agent / 'Agent.py').write_text(f'from {module}.supervisor import ChatModel\n')
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
    if mapped:
        # Demonstrate the exact unchanged nested implementation cannot find the
        # sibling prompt before registration applies the explicit mapping.
        with pytest.raises(FileNotFoundError):
            runpy.run_path(str(target_file))["ChatModel"]().load_context(None)
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
    paths = team_code_paths(str(agent), str(team / 'config.yaml'),
        asset_mappings=[{'source': 'configs', 'destination': 'agent/configs'}] if mapped else None)
    assert not (agent / 'configs').exists()
    assert target_file.read_text() == source
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
            if name in {'_team_agent', 'agent'} or name.startswith(('packaging_probe_nested', 'agent.packaging_probe_nested')):
                sys.modules.pop(name, None)
        if mapped:
            shutil.rmtree(Path(paths[0]).parent)


def test_optional_assets_and_legacy_subpackage_aliases(tmp_path):
    agent = tmp_path / 'agent'; agent.mkdir()
    for name in ['workers', '.cache', '__pycache__']:
        (agent / name).mkdir()
    paths = team_code_paths(str(agent), str(tmp_path / 'missing.yaml'))
    assert paths == [str(agent), str(agent / 'workers')]


@pytest.mark.parametrize('mapping', [
    {}, [{'source': 'configs'}], [{'source': '../outside', 'destination': 'agent/configs'}],
    [{'source': '/tmp', 'destination': 'agent/configs'}],
    [{'source': 'configs', 'destination': '../escape'}],
    [{'source': 'configs', 'destination': 'configs'}],
    [{'source': 'configs', 'destination': 'agent/existing'}],
    [{'source': 'missing', 'destination': 'agent/configs'}],
])
def test_invalid_mappings_rejected_without_source_mutation(tmp_path, mapping):
    agent = tmp_path / 'agent'; agent.mkdir()
    (agent / 'existing').write_text('unchanged')
    (tmp_path / 'configs').mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        team_code_paths(str(agent), asset_mappings=mapping)
    assert (agent / 'existing').read_text() == 'unchanged'
    assert sorted(p.name for p in agent.iterdir()) == ['existing']


def test_mapping_rejects_external_symlink(tmp_path):
    team = tmp_path / 'team'; team.mkdir()
    agent = team / 'agent'; agent.mkdir()
    configs = team / 'configs'; configs.mkdir()
    outside = tmp_path / 'outside'; outside.write_text('external')
    (configs / 'external').symlink_to(outside)
    with pytest.raises(ValueError, match='escapes'):
        team_code_paths(str(agent), asset_mappings=[{'source': 'configs', 'destination': 'agent/configs'}])


def test_mapping_is_explicit_and_empty_list_preserves_defaults(tmp_path):
    agent = tmp_path / 'agent'; agent.mkdir()
    nested = agent / 'supervisors'; nested.mkdir()
    configs = tmp_path / 'configs'; configs.mkdir()
    (configs / 'prompts.json').write_text('original')
    expected = [str(agent), str(nested), str(configs)]
    assert team_code_paths(str(agent)) == expected
    assert team_code_paths(str(agent), asset_mappings=[]) == expected
    # Existing path resolution from the nested implementation has no prompt.
    assert not (nested.parent / 'configs/prompts.json').exists()
    paths = team_code_paths(str(agent), asset_mappings=[{'source': 'configs', 'destination': 'agent/configs'}])
    try:
        assert (Path(paths[0]) / 'configs/prompts.json').read_text() == 'original'
        assert not (agent / 'configs').exists()
        assert len(paths) == len(expected)  # No new import aliases from injected assets.
    finally:
        shutil.rmtree(Path(paths[0]).parent)


def test_registration_inline_validation_matches_mapped_package_root(tmp_path, monkeypatch):
    """Execute the notebook's real prevalidation path block before MLflow staging."""
    import ast
    agent = tmp_path / 'team/agent'; agent.mkdir(parents=True)
    (agent / '__init__.py').write_text('')
    (agent / 'nested_inline_probe').mkdir()
    (agent / 'nested_inline_probe/__init__.py').write_text('VALUE = 42')
    (tmp_path / 'team/configs').mkdir()
    paths = team_code_paths(str(agent), asset_mappings=[{'source':'configs', 'destination':'agent/configs'}])
    notebook = Path(__file__).resolve().parents[2] / 'src/agent_development/agent/notebooks/RegisterModel.py'
    tree = ast.parse(notebook.read_text())
    block = next(node for node in tree.body if isinstance(node, ast.If)
        and ast.unparse(node.test) == "model_cfg.get('asset_mappings') and use_team_agent")
    monkeypatch.setattr(sys, 'path', list(sys.path))
    namespace = {'model_cfg':{'asset_mappings':[{'source':'configs','destination':'agent/configs'}]},
        'use_team_agent':True, 'code_paths':['tools','framework']+paths, 'os':__import__('os'), 'sys':sys}
    try:
        exec(compile(ast.Module(body=[block],type_ignores=[]),str(notebook),'exec'),namespace)
        import importlib
        assert importlib.import_module('agent.nested_inline_probe').VALUE == 42
    finally:
        for name in ['agent.nested_inline_probe','agent']:sys.modules.pop(name,None)
        shutil.rmtree(Path(paths[0]).parent)


def test_registration_retains_prompt_manifest_directory_binding():
    import ast
    notebook = Path(__file__).resolve().parents[2] / 'src/agent_development/agent/notebooks/RegisterModel.py'
    tree = ast.parse(notebook.read_text())
    binding = next(node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'team_configs_dir' for target in node.targets))
    namespace = {'os': __import__('os'), 'team_agent_dir': '/team/agent'}
    exec(compile(ast.Module(body=[binding],type_ignores=[]), str(notebook),'exec'),namespace)
    assert namespace['team_configs_dir'] == '/team/configs'


def test_file_mapping_preserves_source_and_rejects_duplicate_destination(tmp_path):
    agent = tmp_path / 'agent'; agent.mkdir()
    configs = tmp_path / 'configs'; configs.mkdir()
    prompt = configs / 'prompt.txt'; prompt.write_text('unchanged file prompt')
    mapping = {'source':'configs/prompt.txt','destination':'agent/assets/prompt.txt'}
    paths = team_code_paths(str(agent), asset_mappings=[mapping])
    try:
        assert (Path(paths[0]) / 'assets/prompt.txt').read_bytes() == prompt.read_bytes()
        assert not (agent / 'assets').exists()
        with pytest.raises(ValueError, match='overwrite'):
            team_code_paths(str(agent), asset_mappings=[mapping,mapping])
    finally:
        shutil.rmtree(Path(paths[0]).parent)
