from pathlib import Path
import pytest
import yaml
from framework.model_resources import model_dependencies


def test_legacy_dependency_defaults_extend_and_replace():
    assert model_dependencies({}, '.', ['old']) == ['old']
    assert model_dependencies({'model': {'pip_dependencies': ['new', 'old']}}, '.', ['old']) == ['old', 'new']
    assert model_dependencies({'model': {'pip_dependency_mode': 'replace', 'pip_dependencies': ['new']}}, '.', ['old']) == ['new']


def test_team_lock_is_only_dependency_source():
    team = Path(__file__).resolve().parents[2] / 'src/teams/platform-intelligence'
    config = yaml.safe_load((team / 'config.yaml').read_text())
    packages = model_dependencies(config, team, ['ignored'])
    assert 'mlflow==3.11.1' in packages and 'pg8000==1.31.5' in packages
    bundle = yaml.safe_load((team / 'deployment/databricks.yml').read_text())
    job = bundle['resources']['jobs']['agentops_pipeline']
    assert job['environments'][0]['spec']['dependencies'] == [
        '-r ${workspace.file_path}/src/teams/platform-intelligence/configs/requirements.lock']
    assert len(packages) > 100


def test_dependency_file_rejects_ambiguous_config_and_path_escape(tmp_path):
    with pytest.raises(ValueError, match='not both'):
        model_dependencies({'model': {'pip_dependencies': ['x'], 'pip_dependencies_file': 'lock'}}, tmp_path, [])
    with pytest.raises(ValueError, match='inside'):
        model_dependencies({'model': {'pip_dependencies_file': '../lock'}}, tmp_path, [])
    (tmp_path / 'lock').write_text('--extra-index-url https://example.invalid\n')
    with pytest.raises(ValueError, match='pip options'):
        model_dependencies({'model': {'pip_dependencies_file': 'lock'}}, tmp_path, [])


def test_null_model_dependencies_keep_legacy_alias():
    assert model_dependencies({'model': {'pip_dependencies': None}, 'pip_dependencies': ['alias']}, '.', []) == ['alias']


def test_resources_deduplicate_across_explicit_and_legacy_declarations():
    from types import SimpleNamespace
    from framework.model_resources import unique_model_resources
    resource = SimpleNamespace(to_dict=lambda: {'vector_search_index': {'index_name': 'catalog.schema.docs'}})
    assert unique_model_resources([resource, resource]) == [resource]
