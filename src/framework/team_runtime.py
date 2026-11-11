"""Team request contracts shared by registration, evaluation and smoke notebooks.

Adapters live in trusted team source. They adapt requests, never replace pipeline
stages. With no adapter, the standard ResponsesAgent chat contract is unchanged.
"""
from pathlib import Path
import importlib.util
import uuid
import yaml


class TeamRuntime:
    def __init__(self, bundle_root, team_name, dbutils=None):
        self.owner = 'evaluation:agentops:' + str(uuid.uuid4())
        self.adapter = None
        self.config = {}
        if not team_name:
            return
        team = (Path(bundle_root) / 'src/teams' / team_name).resolve()
        if team.parent != (Path(bundle_root) / 'src/teams').resolve():
            raise ValueError('Invalid team name')
        with (team / 'config.yaml').open() as source:
            self.config = yaml.safe_load(source) or {}
        adapter = team / 'request_adapter.py'
        if adapter.is_file():
            spec = importlib.util.spec_from_file_location('agentops_team_request_adapter', adapter)
            self.adapter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.adapter)
            self.adapter.prepare(self.config, dbutils, team)

    @property
    def enabled(self):
        return self.adapter is not None

    def query(self, stage, default):
        """Team-specific startup questions belong to pipeline.sanity_queries."""
        return (self.config.get('pipeline') or {}).get('sanity_queries', {}).get(stage, default)

    def invoke(self, endpoint, query, *, workspace=None):
        from framework.inference.client import invoke_endpoint
        timeout = (self.config.get('pipeline') or {}).get('request_timeout_seconds', 240)
        return invoke_endpoint(endpoint, self.request(query), timeout_seconds=timeout, workspace=workspace)

    async def ainvoke(self, endpoint, query, *, workspace=None):
        from framework.inference.client import ainvoke_endpoint
        from framework.inference.async_runtime import run_blocking
        timeout = (self.config.get('pipeline') or {}).get('request_timeout_seconds', 240)
        payload = await run_blocking(self.request, query)
        return await ainvoke_endpoint(endpoint, payload, timeout_seconds=timeout, workspace=workspace)

    def request(self, query):
        if self.adapter:
            return self.adapter.build_request(str(query), self.owner)
        return {'input': [{'role': 'user', 'content': str(query)}]}

    def input_example(self, query):
        """Reusable public example; never persist short-lived evaluation credentials."""
        if self.adapter and hasattr(self.adapter, 'build_input_example'):
            return self.adapter.build_input_example(str(query), self.config)
        return {'input': [{'role': 'user', 'content': str(query)}]}

    def evaluation_output(self, result):
        from framework.inference import extract_response_text
        if not self.adapter:
            return extract_response_text(result)
        data = result.model_dump() if hasattr(result, 'model_dump') else result
        return {'response': extract_response_text(data),
                'execution_trace_id': (data.get('custom_outputs') or {}).get('trace_id')}


def resolve_experiment_name(config, default_name, client):
    """Choose a portable experiment name, or explicitly reuse an existing ID."""
    tracing = config.get('tracing') or {}
    name = str(tracing.get('experiment_name') or '').strip()
    experiment_id = str(tracing.get('experiment_id') or '').strip()
    if name and experiment_id:
        raise ValueError('Configure tracing.experiment_name or experiment_id, not both')
    if experiment_id:
        experiment = client.get_experiment(experiment_id)
        if experiment is None or experiment.lifecycle_stage != 'active':
            raise ValueError('Configured MLflow experiment must exist and be active')
        return experiment.name
    return name or default_name


def serving_environment(bundle_root, team_name, workspace, endpoint, experiment_id=''):
    """Preserve existing runtime identity/secret references; apply team overrides."""
    from databricks.sdk.errors import NotFound
    environment = {}
    try:
        current = workspace.serving_endpoints.get(endpoint)
        for entity in current.config.served_entities or []:
            environment.update(entity.environment_vars or {})
    except NotFound:
        pass
    if team_name:
        with (Path(bundle_root) / 'src/teams' / team_name / 'config.yaml').open() as source:
            config = yaml.safe_load(source) or {}
        environment.update((config.get('deployment') or {}).get('environment_vars') or {})
    # This run's registration output wins over stale endpoint/config values.
    if experiment_id:
        environment['MLFLOW_EXPERIMENT_ID'] = str(experiment_id)
    return environment


def candidate_model_uri(dbutils, model_name):
    """Select this pipeline run's candidate instead of an unrelated latest version."""
    try:
        version = dbutils.jobs.taskValues.get(taskKey='register_model', key='model_version')
    except Exception:
        version = None  # Standalone notebook/legacy pipeline: preserve alias fallback.
    if version:
        return f'models:/{model_name}/{version}'
    return f'models:/{model_name}@champion'
