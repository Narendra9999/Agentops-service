"""Shared pipeline contracts for standard and authenticated team agents."""
from types import SimpleNamespace
from pathlib import Path
import pytest
import yaml
from framework.team_runtime import TeamRuntime, candidate_model_uri, serving_environment
from framework.team_runtime import resolve_experiment_name

ROOT = Path(__file__).resolve().parents[2]


def test_default_request_needs_no_team_or_secret():
    runtime = TeamRuntime(ROOT, '')
    assert runtime.request('Hello') == {'input':[{'role':'user','content':'Hello'}]}
    assert not runtime.enabled


def test_experiment_name_requires_no_preexisting_id():
    client = SimpleNamespace(get_experiment=lambda _: pytest.fail('ID lookup not expected'))
    assert resolve_experiment_name({'tracing': {'experiment_name': '/Shared/new-agent'}}, '/fallback', client) == '/Shared/new-agent'
    assert resolve_experiment_name({}, '/fallback', client) == '/fallback'


def test_explicit_experiment_id_and_conflicting_settings():
    client = SimpleNamespace(get_experiment=lambda _: SimpleNamespace(name='/existing', lifecycle_stage='active'))
    assert resolve_experiment_name({'tracing': {'experiment_id': '123'}}, '/fallback', client) == '/existing'
    with pytest.raises(ValueError, match='not both'):
        resolve_experiment_name({'tracing': {'experiment_id': '123', 'experiment_name': '/new'}}, '/fallback', client)
    client.get_experiment = lambda _: None
    with pytest.raises(ValueError, match='exist and be active'):
        resolve_experiment_name({'tracing': {'experiment_id': '123'}}, '/fallback', client)


def test_registration_experiment_overrides_stale_serving_destination(monkeypatch):
    import sys
    from types import ModuleType
    errors = ModuleType('databricks.sdk.errors')
    errors.NotFound = type('NotFound', (Exception,), {})
    monkeypatch.setitem(sys.modules, 'databricks.sdk.errors', errors)
    entity = SimpleNamespace(environment_vars={'MLFLOW_EXPERIMENT_ID': 'old', 'KEEP': 'value'})
    workspace = SimpleNamespace(serving_endpoints=SimpleNamespace(get=lambda _: SimpleNamespace(config=SimpleNamespace(served_entities=[entity]))))
    env = serving_environment(ROOT, '', workspace, 'endpoint', experiment_id='new')
    assert env == {'MLFLOW_EXPERIMENT_ID': 'new', 'KEEP': 'value'}


def test_first_deployment_sets_experiment_without_existing_endpoint(monkeypatch):
    import sys
    from types import ModuleType
    errors = ModuleType('databricks.sdk.errors')
    errors.NotFound = type('NotFound', (Exception,), {})
    monkeypatch.setitem(sys.modules, 'databricks.sdk.errors', errors)
    def missing(_):
        raise errors.NotFound('not deployed')
    workspace = SimpleNamespace(serving_endpoints=SimpleNamespace(get=missing))
    assert serving_environment(ROOT, '', workspace, 'new-endpoint', experiment_id='created') == {'MLFLOW_EXPERIMENT_ID': 'created'}


def test_team_memory_requests_use_evaluation_owner_and_bound_signature(monkeypatch):
    secret = SimpleNamespace(get=lambda scope, key: 'unit-test-signing-key')
    runtime = TeamRuntime(ROOT, 'platform-intelligence', SimpleNamespace(secrets=secret))
    from fe_multi_agent.memory import verify_context
    request = runtime.request('Read the saved region')
    context = verify_context(request['custom_inputs']['application_context'], request['input'])
    assert context['owner_id'].startswith('evaluation:agentops:')
    assert context['owner_id'] == runtime.owner
    with pytest.raises(ValueError, match='does not match'):
        verify_context(request['custom_inputs']['application_context'],[{'role':'user','content':'Modified'}])


def test_registration_example_never_reuses_signed_evaluation_context():
    secret = SimpleNamespace(get=lambda scope, key: 'unit-test-signing-key')
    runtime = TeamRuntime(ROOT, 'platform-intelligence', SimpleNamespace(secrets=secret))
    example = runtime.input_example('Hello')
    assert example == {'input': [{'role': 'user', 'content': 'Hello'}]}
    assert 'application_context' in runtime.request('Hello')['custom_inputs']
    runtime.config['multi_agent']['settings']['unsigned_context_policy'] = 'reject'
    assert runtime.input_example('Hello') is None


def test_candidate_uses_run_version_and_legacy_fallback():
    dbutils = SimpleNamespace(jobs=SimpleNamespace(taskValues=SimpleNamespace(get=lambda **_: '21')))
    assert candidate_model_uri(dbutils,'c.s.m') == 'models:/c.s.m/21'
    def missing(**kwargs):raise ValueError('standalone')
    dbutils.jobs.taskValues.get = missing
    assert candidate_model_uri(dbutils,'c.s.m') == 'models:/c.s.m@champion'


def test_existing_serving_secret_references_are_preserved(monkeypatch):
    import sys
    from types import ModuleType
    errors = ModuleType('databricks.sdk.errors')
    errors.NotFound = type('NotFound', (Exception,), {})
    monkeypatch.setitem(sys.modules, 'databricks.sdk.errors', errors)
    entity=SimpleNamespace(environment_vars={'CLIENT_SECRET':'{{secrets/scope/key}}'})
    workspace=SimpleNamespace(serving_endpoints=SimpleNamespace(get=lambda _:SimpleNamespace(config=SimpleNamespace(served_entities=[entity]))))
    assert serving_environment(ROOT,'',workspace,'endpoint')['CLIENT_SECRET']=='{{secrets/scope/key}}'


def test_bundle_uses_original_notebooks_only():
    bundle=yaml.safe_load((ROOT/'src/teams/platform-intelligence/deployment/databricks.yml').read_text())
    tasks=bundle['resources']['jobs']['agentops_pipeline']['tasks']
    canonical=yaml.safe_load((ROOT/'resources/pipeline-resource.yml').read_text())['resources']['jobs']['agentops_pipeline']['tasks']
    assert [t['task_key'] for t in tasks]==[t['task_key'] for t in canonical]
    for actual,expected in zip(tasks,canonical):
        assert actual['notebook_task']['notebook_path'].endswith(expected['notebook_task']['notebook_path'][3:])
        assert actual.get('depends_on')==expected.get('depends_on')
    assert not (ROOT/'src/teams/platform-intelligence/agent/fe_multi_agent/pipeline.py').exists()


def test_framework_build_stages_only_framework_sources(tmp_path, monkeypatch):
    from framework.mlops_utils import stage_framework_source
    import tempfile
    project = tmp_path / 'project'
    (project / 'src/framework').mkdir(parents=True)
    (project / 'src/framework/__init__.py').write_text('')
    (project / 'src/customer_secrets').mkdir()
    (project / 'src/customer_secrets/secret.txt').write_text('excluded')
    (project / 'pyproject.toml').write_text('[project]\nname="framework"\n')
    staging = tmp_path / 'staging'
    staging.mkdir()
    monkeypatch.setattr(tempfile, 'mkdtemp', lambda **_: str(staging))
    result = Path(stage_framework_source(str(project)))
    assert (result / 'src/framework/__init__.py').exists()
    assert not (result / 'src/customer_secrets').exists()


def test_team_quality_gate_covers_all_six_native_scorers():
    from agent_development.agent_evaluation.evaluation.scorer_loader import get_thresholds
    config = yaml.safe_load((ROOT/'src/teams/platform-intelligence/config.yaml').read_text())
    thresholds = get_thresholds(config['evaluation'])
    assert thresholds == {name + '/mean': 0.8 for name in (
        'correctness', 'relevance_to_query', 'safety', 'cost_grounding',
        'platform_evidence', 'routing',
    )}


@pytest.mark.parametrize('metrics', [{}, {'routing/mean': float('nan')}, {'routing/mean': .5}])
def test_combined_notebook_gate_rejects_missing_nonfinite_and_low_scores(metrics):
    from framework.evaluation.evaluation_pipeline import compute_quality_gate
    gate = compute_quality_gate({'routing/mean': .8}, metrics, 1.0, .8)
    assert not gate['gate_passed']
