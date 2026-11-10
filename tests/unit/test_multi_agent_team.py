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
    class Index:
        def __init__(self,index_name):self.index=index_name
    resources.DatabricksVectorSearchIndex=Index
    resources.DatabricksServingEndpoint=Endpoint
    resources.DatabricksGenieSpace=Genie
    monkeypatch.setitem(sys.modules,'mlflow.models.resources',resources)
    return Endpoint,Genie


def test_standard_team_loader_and_existing_worker_bindings():
    from framework.team_config import load_team_settings
    settings=load_team_settings('platform-intelligence',str(ROOT))
    assert settings['agent_name']=='platform_intelligence'
    assert settings['schema']=='agentops_multi'
    assert settings['agent_endpoint']=='fe-platform-intelligence'
    cfg=yaml.safe_load((TEAM/'config.yaml').read_text())
    assert set(cfg['multi_agent']['agents'])=={'supervisor','platform_docs','cost_estimation'}
    assert not cfg['session_history']['enabled']  # App owns authenticated history.
    assert cfg['model']['pip_dependency_mode']=='replace'
    assert (TEAM/'agent/Agent.py').is_file()
    assert 'vector_search' not in cfg


def test_resources_reference_existing_workers_only(monkeypatch):
    Endpoint,Genie=resource_types(monkeypatch)
    from framework.model_resources import language_model_resources
    cfg=yaml.safe_load((TEAM/'config.yaml').read_text())
    cfg['multi_agent']['settings'].update(platform_mode='remote',platform_endpoint='existing-knowledge',genie_space_id='existing-genie')
    result=language_model_resources(cfg,cfg['llm']['endpoint'])
    assert len(result)==2 and isinstance(result[0],Endpoint) and isinstance(result[1],Genie)
    assert result[0].name=='existing-knowledge' and result[1].space=='existing-genie'
    cfg['multi_agent']['settings']['cost_endpoint']='existing-cost'
    result=language_model_resources(cfg,cfg['llm']['endpoint'])
    assert [r.name for r in result]==['existing-knowledge','existing-cost']
    cfg['llm'].update(api='model_serving',endpoint='existing-supervisor-llm')
    result=language_model_resources(cfg,cfg['llm']['endpoint'])
    assert [r.name for r in result]==['existing-knowledge','existing-cost','existing-supervisor-llm']


def test_unbound_worker_fails_before_registration(monkeypatch):
    resource_types(monkeypatch)
    from framework.model_resources import language_model_resources
    cfg=yaml.safe_load((TEAM/'config.yaml').read_text())
    cfg['multi_agent']['settings'].update(platform_mode='remote',platform_endpoint='REQUIRED_PLATFORM_ENDPOINT')
    with pytest.raises(ValueError,match='platform_endpoint'):
        language_model_resources(cfg,cfg['llm']['endpoint'])


def test_single_agent_resource_behavior_is_preserved(monkeypatch):
    Endpoint,_=resource_types(monkeypatch)
    from framework.model_resources import language_model_resources
    result=language_model_resources({},'single-agent-llm')
    assert len(result)==1 and isinstance(result[0],Endpoint) and result[0].name=='single-agent-llm'


def test_in_process_worker_declares_index_without_worker_endpoint(monkeypatch):
    resource_types(monkeypatch)
    from framework.model_resources import language_model_resources
    cfg=yaml.safe_load((TEAM/'config.yaml').read_text())
    result=language_model_resources(cfg,cfg['llm']['endpoint'])
    assert len(result)==2 and result[0].index==cfg['multi_agent']['settings']['docs_index']
    assert result[1].space==cfg['multi_agent']['settings']['genie_space_id']


def test_resources_are_topology_independent_and_deduplicated(monkeypatch):
    Endpoint,_=resource_types(monkeypatch)
    from framework.model_resources import language_model_resources
    config={'model':{'resources':[{'type':'serving_endpoint','name':'worker'},
                                {'type':'serving_endpoint','name':'worker'}]}}
    result=language_model_resources(config,'unused')
    assert len(result)==1 and result[0].name=='worker'
    with pytest.raises(ValueError,match='declare model.resources'):
        language_model_resources({'multi_agent':{'pattern':'customer_graph'}},'unused')


def test_supervisor_uses_native_tracing_without_optional_observer():
    cfg = yaml.safe_load((TEAM / 'config.yaml').read_text())
    assert cfg['multi_agent']['settings']['observer_enabled'] is False
    assert cfg['multi_agent']['settings']['memory_enabled'] is True
