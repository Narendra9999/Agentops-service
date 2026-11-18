"""Team memory policy bindings for the shared supervisor."""
from .progress import emit, operation
from .usage import agent_usage_scope
from framework.orchestration.memory_supervisor import build_memory_graph as build


def build_memory_graph(model, workers, prompt, memory, application_context, max_steps=24, **kwargs):
    return build(model, workers, prompt, memory, application_context, max_steps,
                 emit=emit, operation=operation, usage_scope=agent_usage_scope,
                 evidence_workers={'cost_estimation'}, **kwargs)
