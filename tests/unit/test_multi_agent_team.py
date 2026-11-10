"""Team contract and resource declarations; no workspace calls."""
import sys
from pathlib import Path
from types import ModuleType
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
TEAM = ROOT / 'src/teams/platform-intelligence'
sys.path.insert(0,str(ROOT/'src'))


def resource_types(monkeypatch):
    resources = ModuleType('mlflow.models.resources')
    class Endpoint:
        def __init__(self, endpoint_name):self.name=endpoint_name
    class Genie:
        def __init__(self, genie_space_id):self.space=genie_space_id
    resources.DatabricksServingEndpoint=Endpoint
    resources.DatabricksGenieSpace=Genie
    monkeypatch.setitem(sys.modules,'mlflow.models.resources',resources)
    return Endpoint,Genie


def test_standard_team_loader_and_existing_worker_bindings():
    from framework.team_config import load_team_settings
    settings=load_team_settings('platform-intelligence',str(ROOT))
    assert settings['agent_name']=='platform_team'
    assert settings['schema']=='agentops_multi'
    assert settings['agent_endpoint']=='fe-platform-agent-team'
    cfg=yaml.safe_load((TEAM/'config.yaml').read_text())
    assert set(cfg['multi_agent']['agents'])=={'supervisor','platform_docs','cost_estimation'}
    assert not cfg['session_history']['enabled']  # App owns authenticated history.
    assert cfg['model']['pip_dependency_mode']=='replace'
    assert (TEAM/'agent/Agent.py').is_file()
    assert 'vector_search' not in cfg


def test_resources_reference_existing_workers_only(monkeypatch):
    Endpoint,Genie=resource_types(monkeypatch)
    from framework.model_resources import language_model_resources
    cfg={'multi_agent':{'pattern':'langgraph_supervisor','settings':{
        'platform_endpoint':'existing-knowledge','genie_space_id':'existing-genie',
        'model_transport':'ai_gateway','supervisor_model':'system.ai.claude-sonnet-4-6'}}}
    result=language_model_resources(cfg,'system.ai.claude-sonnet-4-6')
    assert len(result)==2 and isinstance(result[0],Endpoint) and isinstance(result[1],Genie)
    assert result[0].name=='existing-knowledge' and result[1].space=='existing-genie'
    cfg['multi_agent']['settings']['cost_endpoint']='existing-cost'
    result=language_model_resources(cfg,'system.ai.claude-sonnet-4-6')
    assert [r.name for r in result]==['existing-knowledge','existing-cost']
    cfg['multi_agent']['settings'].update(model_transport='model_serving',supervisor_model='existing-supervisor-llm')
    result=language_model_resources(cfg,'existing-supervisor-llm')
    assert [r.name for r in result]==['existing-knowledge','existing-cost','existing-supervisor-llm']


def test_unbound_worker_fails_before_registration(monkeypatch):
    resource_types(monkeypatch)
    from framework.model_resources import language_model_resources
    cfg=yaml.safe_load((TEAM/'config.yaml').read_text())
    with pytest.raises(ValueError,match='platform_endpoint'):
        language_model_resources(cfg,cfg['llm']['endpoint'])


def test_single_agent_resource_behavior_is_preserved(monkeypatch):
    Endpoint,_=resource_types(monkeypatch)
    from framework.model_resources import language_model_resources
    result=language_model_resources({},'single-agent-llm')
    assert len(result)==1 and isinstance(result[0],Endpoint) and result[0].name=='single-agent-llm'
