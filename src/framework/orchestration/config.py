"""Validate optional supervisor configuration before constructing any clients."""
import re


def validate_supervisor(config):
    if config.get('pattern') != 'supervisor' or config.get('implementation') != 'langgraph':
        raise ValueError('Expected pattern: supervisor and implementation: langgraph')
    supervisor = config.get('supervisor') or {}
    if not supervisor.get('model'):
        raise ValueError('supervisor.model is required')
    workers = config.get('workers')
    if not isinstance(workers, dict) or not workers:
        raise ValueError('Configure at least one worker')
    reserved = {'supervisor', 'finish', 'finish_response', 'list_memory_files',
                'read_memory_file', 'write_memory_file', 'delete_memory_file', 'memory_file_history'}
    for name, spec in workers.items():
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', name) or name in reserved:
            raise ValueError('Invalid or reserved worker name: ' + name)
        if not isinstance(spec, dict) or not str(spec.get('description', '')).strip():
            raise ValueError('Worker requires a description: ' + name)
        transport = spec.get('transport')
        key = {'serving_endpoint': 'endpoint_name', 'genie': 'space_id',
               'in_process': 'implementation'}.get(transport)
        if key is None:
            raise ValueError('Unsupported worker transport: ' + str(transport))
        value = spec.get(key)
        if not isinstance(value, str) or not value.strip() or '/' in value or 'REQUIRED' in value:
            raise ValueError(f'Configure workers.{name}.{key}; use a name or ID, not a URL')
    return workers


def build_workers(specs, factories, *, local_implementations=()):
    """Factories return workers implementing invoke/ainvoke on messages state.

    Local implementations must be explicitly registered by the host; configuration
    cannot import arbitrary Python code. Validate the entire registry before I/O.
    """
    validate_supervisor({'pattern': 'supervisor', 'implementation': 'langgraph',
                         'supervisor': {'model': 'bound'}, 'workers': specs})
    for spec in specs.values():
        if spec['transport'] not in factories:
            raise ValueError('No factory for transport: ' + spec['transport'])
        if spec['transport'] == 'in_process' and spec['implementation'] not in local_implementations:
            raise ValueError('Unregistered local implementation: ' + spec['implementation'])
    return {name: factories[spec['transport']](name, spec) for name, spec in specs.items()}
