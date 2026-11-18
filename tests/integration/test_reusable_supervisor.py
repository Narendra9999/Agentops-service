"""Exercise shared routing with arbitrary worker names, without Databricks access."""
import asyncio
from types import SimpleNamespace
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from framework.orchestration.config import validate_supervisor, build_workers
from framework.orchestration.supervisor import build_supervisor_graph
from framework.orchestration.memory_supervisor import build_memory_graph


def config():
    return {'pattern':'supervisor','implementation':'langgraph',
            'supervisor':{'model':'system.ai.example'}, 'workers':{
                'billing':{'transport':'genie','space_id':'space','description':'Analyze bills'},
                'support':{'transport':'serving_endpoint','endpoint_name':'support-endpoint','description':'Resolve issues'},
                'research':{'transport':'serving_endpoint','endpoint_name':'research-endpoint','description':'Research topics'}}}


def test_registry_arbitrary_workers():
    cfg=config()
    calls=[]
    workers=build_workers(validate_supervisor(cfg), {
        'genie':lambda name,spec:calls.append((name,spec['space_id'])) or name,
        'serving_endpoint':lambda name,spec:calls.append((name,spec['endpoint_name'])) or name})
    assert set(workers)=={'billing','support','research'}
    assert calls==[('billing','space'),('support','support-endpoint'),('research','research-endpoint')]


@pytest.mark.parametrize('change', ['reserved','transport','url','missing','local'])
def test_invalid_registry_before_factories(change):
    cfg=config(); calls=[]
    if change=='reserved':cfg['workers']['supervisor']=cfg['workers'].pop('billing')
    if change=='transport':cfg['workers']['billing']['transport']='unknown'
    if change=='url':cfg['workers']['support']['endpoint_name']='https://workspace/endpoint'
    if change=='missing':del cfg['workers']['billing']['space_id']
    if change=='local':cfg['workers']['billing'].update(transport='in_process',implementation='untrusted')
    with pytest.raises(ValueError):
        build_workers(cfg['workers'],{key:lambda *args:calls.append(args) for key in ('genie','serving_endpoint','in_process')})
    assert calls==[]


class Worker:
    def __init__(self):self.calls=[]
    async def ainvoke(self,state):
        self.calls.append(state)
        return {'messages':[AIMessage(content='Evidence from worker')], 'source_urls':['https://example.com/docs']}


class Router:
    def __init__(self):self.calls=0
    def with_structured_output(self,schema):self.schema=schema;return self
    async def ainvoke(self,messages):
        self.calls+=1
        if self.calls==1:return self.schema(next_agent='research',task='Investigate this request')
        # The completed worker must no longer be an allowed destination.
        with pytest.raises(Exception):self.schema(next_agent='research',task='Repeat')
        return self.schema(next_agent='finish',answer='Evidence-based answer')


def test_shared_supervisor_routes_arbitrary_worker_once():
    worker=Worker();router=Router()
    graph=build_supervisor_graph(router, {'research':worker}, 'Research requests',
                                 SimpleNamespace(execution_id='test'))
    result=asyncio.run(graph.ainvoke({'messages':[HumanMessage(content='Help research')]}))
    assert len(worker.calls)==1 and router.calls==2
    assert 'Evidence-based answer' in result['messages'][-1].content
    assert 'https://example.com/docs' in result['messages'][-1].content


class Memory:
    def files(self,query):return []


class ToolRouter:
    def __init__(self):self.calls=0;self.allowed=[]
    def bind_tools(self,tools,**kwargs):
        self.allowed.append([t.name for t in tools]);return self
    async def ainvoke(self,messages):
        sequence=[('list_memory_files',{'query':''}),('research',{'task':'Find evidence'}),
                  ('finish_response',{'answer':'Research complete'})]
        name,args=sequence[self.calls];self.calls+=1
        return AIMessage(content='',tool_calls=[{'id':str(self.calls),'name':name,'args':args}])


def test_memory_graph_advertises_configured_workers_and_enforces_memory_first():
    router=ToolRouter();worker=Worker()
    graph=build_memory_graph(router,{'research':worker},'Research with memory',Memory(),{})
    result=asyncio.run(graph.ainvoke({'messages':[HumanMessage(content='Research this')]}))
    assert router.allowed[0]==['list_memory_files']
    assert 'research' in router.allowed[1]
    assert 'cost_estimation' not in router.allowed[1] and 'platform_knowledge' not in router.allowed[1]
    assert len(worker.calls)==1
    assert result['messages'][-1].content.startswith('Research complete')


@pytest.mark.parametrize('patch', [
    {'pattern':'parallel'}, {'implementation':'unknown'}, {'workers':{}}, {'supervisor':{}},
])
def test_unsupported_supervisor_configuration(patch):
    cfg=config();cfg.update(patch)
    with pytest.raises(ValueError):validate_supervisor(cfg)


def test_platform_team_configuration_loads_settings_and_instructions(monkeypatch):
    from pathlib import Path
    import ast
    import yaml
    team=Path(__file__).resolve().parents[2]/'src/teams/platform-intelligence'
    monkeypatch.syspath_prepend(str(team/'agent'))
    from fe_multi_agent.config import Settings
    cfg=yaml.safe_load((team/'config.yaml').read_text())
    # Execute the actual team loader without creating live clients or autologging.
    tree=ast.parse((team/'agent/Agent.py').read_text())
    target=next(node for node in tree.body if isinstance(node,ast.ClassDef))
    class Target:
        def load_context(self,context):pass
    import os
    scope={'SupervisorAgent':Target,'Settings':Settings,'Path':Path,'os':os,
           '__file__':str(team/'agent/Agent.py'),
           'mlflow':SimpleNamespace(langchain=SimpleNamespace(autolog=lambda:None))}
    exec(compile(ast.Module(body=[target],type_ignores=[]),'Agent.py','exec'),scope)
    agent=scope['ChatModel']()
    agent.load_context(SimpleNamespace(model_config=cfg))
    assert set(agent.settings.workers)=={'platform_docs','cost_estimation'}
    assert agent.settings.memory_enabled
    assert 'platform_docs(task)' in agent.settings.supervisor_instructions
    assert agent.settings.supervisor_model==cfg['multi_agent']['supervisor']['model']
    cfg['multi_agent']['supervisor']['instructions']='../outside.md'
    with pytest.raises(ValueError,match='inside the team directory'):
        agent.load_context(SimpleNamespace(model_config=cfg))


def test_migrated_instructions_preserve_active_prompt_policies():
    import json
    from pathlib import Path
    team=Path(__file__).resolve().parents[2]/'src/teams/platform-intelligence'
    active=json.loads((team/'configs/prompts.json').read_text())['prompts']['supervisor']['template']
    expected=active.replace('platform_knowledge(task)', 'platform_docs(task)').replace(
        'platform_docs is an internal worker identifier, not a tool name.',
        'Worker tool names match the configured worker identifiers.').replace(
        'which invokes the embedded platform_docs worker.', 'which invokes the configured platform_docs worker.')
    assert (team/'configs/supervisor.md').read_text().strip()==expected.strip()
    assert 'MANDATORY COST DELEGATION' in expected and 'COST EVIDENCE POLICY' in expected
