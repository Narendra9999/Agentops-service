"""Team-scoped OBO registration policy with unchanged system-only defaults."""
from pathlib import Path

import pytest
import yaml
from mlflow.models import Model
from framework.model_resources import language_model_resources, model_auth_kwargs

TEAM = Path('src/teams/platform-intelligence-supervisor')


def test_team_obo_policy_serializes_and_preserves_system_resources():
    config = yaml.safe_load((TEAM / 'config.yaml').read_text())
    resources = language_model_resources(config, config['llm']['endpoint'])
    kwargs = model_auth_kwargs(config, resources)
    assert set(kwargs) == {'auth_policy'}  # MLflow disallows resources + auth_policy.
    policy = kwargs['auth_policy'].to_dict()
    assert policy['user_auth_policy']['api_scopes'] == ['genie', 'sql']
    assert config['llm']['endpoint'] in str(policy['system_auth_policy'])
    assert config['supervisor']['workers']['platform_engineering']['endpoint_name'] in str(policy['system_auth_policy'])
    assert Model(auth_policy=kwargs['auth_policy']).auth_policy == policy


def test_other_teams_keep_resources_only_registration():
    resources = language_model_resources({}, 'single-agent-model')
    assert model_auth_kwargs({}, resources) == {'resources': resources}


@pytest.mark.parametrize('options', [[], {}, {'unexpected': []}, {'user_api_scopes': []},
    {'user_api_scopes': 'genie'}, {'user_api_scopes': ['']}, {'user_api_scopes': [42]}])
def test_invalid_user_scope_configuration_fails(options):
    with pytest.raises(ValueError, match='model.auth_policy'):
        model_auth_kwargs({'model': {'auth_policy': options}}, [])


def test_user_scope_names_are_trimmed_and_deduplicated():
    policy = model_auth_kwargs({'model': {'auth_policy': {
        'user_api_scopes': [' genie ', 'sql', 'genie']}}}, [])['auth_policy'].to_dict()
    assert policy['user_auth_policy']['api_scopes'] == ['genie', 'sql']
