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

    def request(self, query):
        if self.adapter:
            return self.adapter.build_request(str(query), self.owner)
        return {'input': [{'role': 'user', 'content': str(query)}]}

    def evaluation_output(self, result):
        from framework.inference import extract_response_text
        if not self.adapter:
            return extract_response_text(result)
        data = result.model_dump() if hasattr(result, 'model_dump') else result
        return {'response': extract_response_text(data),
                'execution_trace_id': (data.get('custom_outputs') or {}).get('trace_id')}


def serving_environment(bundle_root, team_name, workspace, endpoint):
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
