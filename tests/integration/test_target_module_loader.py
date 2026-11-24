"""Entry-point precedence, module fallback and failures for the shared loader."""
from unittest.mock import patch
import sys

import pytest
from framework.inference.target_loader import team_target_entry
from framework.inference.cost_tracking_model import _resolve_inner_model


def test_configured_module_wins_over_agent_file(tmp_path):
    (tmp_path / 'Agent.py').write_text('raise AssertionError("must not import default")\n')
    (tmp_path / 'custom.py').write_text('class Target:\n    source = "module"\n')
    with patch.object(sys, 'path', [str(tmp_path)]):
        target = _resolve_inner_model({'inference': {'target_module': 'custom', 'target_class': 'Target'}})
    assert target.source == 'module'


def test_target_module_loads_from_staged_agent_directory(tmp_path):
    root = tmp_path / 'agent'; root.mkdir()
    (root / 'custom.py').write_text('class Target:\n    source = "module"\n')
    cfg = {'inference': {'target_module': 'custom', 'target_class': 'Target'}}
    assert team_target_entry(root, cfg).name == 'custom.py'
    with patch.object(sys, 'path', [str(tmp_path)]):
        assert _resolve_inner_model(cfg).source == 'module'


def test_broken_explicit_module_does_not_fall_back(tmp_path):
    (tmp_path / 'custom.py').write_text('raise RuntimeError("broken agent")\n')
    (tmp_path / 'Agent.py').write_text('class ChatModel: pass\n')
    with patch.object(sys, 'path', [str(tmp_path)]), pytest.raises(RuntimeError, match='broken agent'):
        _resolve_inner_model({'inference': {'target_module': 'custom', 'target_class': 'ChatModel'}})


def test_module_import_and_class_errors_are_reported(tmp_path):
    (tmp_path / 'custom.py').write_text('class Wrong: pass\n')
    with patch.object(sys, 'path', [str(tmp_path)]), pytest.raises(ImportError, match='target_class'):
        _resolve_inner_model({'inference': {'target_module': 'custom', 'target_class': 'Missing'}})
    with patch.object(sys, 'path', [str(tmp_path)]), pytest.raises(ImportError, match='No packaged'):
        _resolve_inner_model({'inference': {'target_module': 'absent', 'target_class': 'ChatModel'}})


@pytest.mark.parametrize('module', ['../custom', 'custom.py', '/tmp/custom', 'a/b', ''])
def test_invalid_module_names_are_rejected(tmp_path, module):
    with pytest.raises(ValueError, match='target_module'):
        team_target_entry(tmp_path, {'inference': {'target_module': module}})


def test_dotted_module_supports_relative_imports(tmp_path):
    package = tmp_path / 'target_example'; package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'helper.py').write_text('VALUE = 42\n')
    (package / 'supervisor.py').write_text('from .helper import VALUE\nclass Target:\n    value = VALUE\n')
    with patch.object(sys, 'path', [str(tmp_path)]):
        target = _resolve_inner_model({'inference': {
            'target_module': 'target_example.supervisor', 'target_class': 'Target'}})
    assert target.value == 42


def test_registration_detects_module_only_team(tmp_path):
    from pathlib import Path
    import yaml
    root = tmp_path / 'agent'; root.mkdir()
    (root / 'custom.py').write_text('class Target: pass\n')
    config_path = tmp_path / 'config.yaml'
    config_path.write_text('inference:\n  target_module: custom\n  target_class: Target\n')
    source = Path('src/agent_development/agent/notebooks/RegisterModel.py').read_text()
    start = source.index('from framework.inference.target_loader import team_target_entry')
    end = source.index('\ncost_wrapper =', start)
    scope = {'team_agent_dir': str(root), 'config_path': str(config_path), 'yaml': yaml}
    exec(compile(source[start:end], '<registration-entry-selection>', 'exec'), scope)
    assert scope['use_team_agent'] is True
    assert scope['team_entry'] == root / 'custom.py'


def test_unrelated_notebook_agent_does_not_shadow_staged_team_module(tmp_path):
    staged = tmp_path / 'staged'; staged.mkdir()
    team = staged / 'agent'; team.mkdir()
    notebooks = tmp_path / 'notebooks'; notebooks.mkdir()
    (team / 'custom.py').write_text('class Target:\n    source = "team"\n')
    (notebooks / 'Agent.py').write_text('raise AssertionError("unrelated notebook")\n')
    with patch.object(sys, 'path', [str(staged), str(notebooks)]):
        target = _resolve_inner_model({'inference': {'target_module': 'custom', 'target_class': 'Target'}})
    assert target.source == 'team'


def test_default_agent_used_only_without_module_config(tmp_path):
    (tmp_path / 'Agent.py').write_text('class ChatModel:\n    source = "default"\n')
    with patch.object(sys, 'path', [str(tmp_path)]):
        assert _resolve_inner_model({}).source == 'default'
        with pytest.raises(ImportError, match='No packaged'):
            _resolve_inner_model({'inference': {'target_module': 'missing', 'target_class': 'ChatModel'}})


@pytest.mark.parametrize('inference', [{}, {'target_module': 'custom'}, {'target_class': 'ChatModel'}])
def test_incomplete_inference_settings_use_agent_file(tmp_path, inference):
    (tmp_path / 'Agent.py').write_text('class ChatModel:\n    source = "default"\n')
    (tmp_path / 'custom.py').write_text('raise AssertionError("incomplete pair")\n')
    with patch.object(sys, 'path', [str(tmp_path)]):
        assert _resolve_inner_model({'inference': inference}).source == 'default'
