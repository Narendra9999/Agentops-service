"""Team contract and resource declarations; no workspace calls."""
import sys
from pathlib import Path
from types import ModuleType
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
TEAM = ROOT / 'src/teams/platform-intelligence-supervisor'
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
    settings=load_team_settings('platform-intelligence-supervisor',str(ROOT))
    assert settings['agent_name']=='platform_intelligence_supervisor'
    assert settings['agent_endpoint']=='fe-platform-intelligence-supervisor'
    cfg=yaml.safe_load((TEAM/'config.yaml').read_text())
    assert set(cfg['supervisor']['workers'])=={'platform_engineering','cost_intelligence'}
    assert not (TEAM/'agent/Agent.py').exists()
    assert cfg['inference']['target_module']=='PlatformintelligenceSupervisor'


def test_resources_reference_existing_workers_and_supervisor_model(monkeypatch):
    Endpoint,Genie=resource_types(monkeypatch)
    from framework.model_resources import language_model_resources
    cfg=yaml.safe_load((TEAM/'config.yaml').read_text())
    cfg['supervisor']['workers']['platform_engineering']['endpoint_name']='existing-knowledge'
    cfg['supervisor']['workers']['cost_intelligence']['space_id']='existing-genie'
    result=language_model_resources(cfg,cfg['llm']['endpoint'])
    assert len(result)==3 and isinstance(result[0],Endpoint) and isinstance(result[1],Genie)
    assert result[0].name=='existing-knowledge' and result[1].space=='existing-genie'
    assert isinstance(result[2],Endpoint)
    assert result[2].name=='databricks-claude-sonnet-4-6'


def test_single_agent_resource_behavior_is_preserved(monkeypatch):
    Endpoint,_=resource_types(monkeypatch)
    from framework.model_resources import language_model_resources
    result=language_model_resources({},'single-agent-llm')
    assert len(result)==1 and isinstance(result[0],Endpoint) and result[0].name=='single-agent-llm'


def test_resources_are_topology_independent_and_deduplicated(monkeypatch):
    Endpoint,_=resource_types(monkeypatch)
    from framework.model_resources import language_model_resources
    config={'model':{'resources':[{'type':'serving_endpoint','name':'worker'},
                                {'type':'serving_endpoint','name':'worker'}]}}
    result=language_model_resources(config,'unused')
    assert len(result)==1 and result[0].name=='worker'
    with pytest.raises(ValueError,match='declare model.resources'):
        language_model_resources({'multi_agent':{'pattern':'customer_graph'}},'unused')


def test_supervisor_uses_native_tracing_without_framework_orchestration():
    cfg = yaml.safe_load((TEAM / 'config.yaml').read_text())
    assert cfg['tracing']['instrumentation']['enabled']
    source = (TEAM/'agent/PlatformintelligenceSupervisor.py').read_text()
    assert 'framework.orchestration' not in source
    assert not (ROOT/'src/framework/orchestration').exists()
