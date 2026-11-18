"""Team worker adapters and persistence bindings for the reusable supervisor."""
from dataclasses import dataclass
from .progress import emit
from .usage import agent_usage_scope


@dataclass(frozen=True)
class Execution:
    execution_id: str
    tenant: str
    actor: str


def build_graph(settings, workspace, prompts, execution, observer=None, application_context=None):
    from .services import ExistingResponsesWorker, ExistingGenieWorker, chat_model
    from framework.orchestration.config import build_workers
    from dataclasses import replace

    def local_worker(spec):
        from .platform_worker import PlatformKnowledgeWorker
        options = spec.get('settings', {})
        allowed = {'docs_index', 'platform_model', 'docs_text_column', 'docs_url_column'}
        if set(options) - allowed or not options.get('docs_index') or not options.get('platform_model'):
            raise ValueError('platform_knowledge requires docs_index and platform_model settings')
        return PlatformKnowledgeWorker(replace(settings, **options), workspace)

    workers = build_workers(settings.workers, {
        'serving_endpoint': lambda name, spec: ExistingResponsesWorker(workspace, spec['endpoint_name'], name),
        'genie': lambda name, spec: ExistingGenieWorker(workspace, spec['space_id']),
        'in_process': lambda name, spec: local_worker(spec),
    }, local_implementations={'platform_knowledge'})
    descriptions = {name: spec['description'] for name, spec in settings.workers.items()}
    if settings.memory_enabled:
        from .memory import MemoryWorkspace, lakebase_connection
        from .memory_graph import build_memory_graph
        import mlflow
        active=mlflow.get_current_active_span()
        context=application_context or {}
        if not context.get('owner_id'):
            raise ValueError('Authenticated application context required for agent memory')
        memory=MemoryWorkspace(lambda: lakebase_connection(workspace,settings.memory_endpoint),
                           context['owner_id'],context.get('conversation_id',''),
                           active.trace_id if active else '',settings.memory_ttl_days)
        return build_memory_graph(chat_model(settings,workspace,settings.supervisor_model),
            workers,prompts['supervisor'],memory,context,
            worker_descriptions=descriptions)
    return build_supervisor_graph(
        chat_model(settings, workspace, settings.supervisor_model),
        workers,
        prompts["supervisor"], execution, observer, worker_descriptions=descriptions,
    )


def build_supervisor_graph(model, workers, prompt, execution, observer=None, max_delegations=6, **kwargs):
    from framework.orchestration.supervisor import build_supervisor_graph as build
    return build(model, workers, prompt, execution, observer, max_delegations,
                 emit=emit, usage_scope=agent_usage_scope, **kwargs)
