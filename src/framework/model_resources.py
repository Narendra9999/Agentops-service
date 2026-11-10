"""Configuration-driven model resources and dependencies, independent of topology."""
from pathlib import Path


def config_value(config, path):
    value = config
    for key in path.split('.'):
        if not isinstance(value, dict) or key not in value:
            raise ValueError('Missing configuration binding: ' + path)
        value = value[key]
    return value


def language_model_resources(config, resolved_llm_endpoint):
    """Resolve explicit model.resources; preserve the legacy LLM default when absent.

    Resource conditions and bindings use dotted configuration paths. UC AI Gateway
    model services must not be declared as serving endpoint resources.
    """
    from mlflow.models.resources import (
        DatabricksServingEndpoint, DatabricksGenieSpace, DatabricksVectorSearchIndex,
    )
    declarations = (config.get('model') or {}).get('resources')
    if declarations is None:
        if config.get('multi_agent'):
            raise ValueError('Multi-agent teams must declare model.resources')
        return [DatabricksServingEndpoint(endpoint_name=resolved_llm_endpoint)]
    if not isinstance(declarations, list):
        raise ValueError('model.resources must be a list')
    factories = {
        'serving_endpoint': (DatabricksServingEndpoint, 'endpoint_name'),
        'genie_space': (DatabricksGenieSpace, 'genie_space_id'),
        'vector_search_index': (DatabricksVectorSearchIndex, 'index_name'),
    }
    resources, seen = [], set()
    for entry in declarations:
        conditions = entry.get('when') or {}
        if not all(config_value(config, path) == expected for path, expected in conditions.items()):
            continue
        if any(config_value(config, path) == excluded for path, excluded in (entry.get('when_not') or {}).items()):
            continue
        kind = entry.get('type')
        if kind not in factories:
            raise ValueError('Unsupported model resource type: ' + str(kind))
        if ('name' in entry) == ('name_from' in entry):
            raise ValueError('Resource requires exactly one of name or name_from')
        name = config_value(config, entry['name_from']) if 'name_from' in entry else entry['name']
        if not isinstance(name, str) or not name.strip() or 'REQUIRED_' in name:
            raise ValueError('Unbound model resource: ' + str(entry.get('name_from', kind)))
        if (kind, name) not in seen:
            factory, argument = factories[kind]
            resources.append(factory(**{argument: name}))
            seen.add((kind, name))
    return resources


def model_dependencies(config, config_directory, defaults):
    """Read one team lock file or legacy inline packages, then extend/replace defaults."""
    model = config.get('model') or {}
    dependency_file = model.get('pip_dependencies_file')
    packages = model.get('pip_dependencies')
    if packages is None:
        packages = config.get('pip_dependencies', []) or []
    if dependency_file:
        if packages:
            raise ValueError('Configure pip_dependencies_file or pip_dependencies, not both')
        root = Path(config_directory).resolve()
        path = (root / dependency_file).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Dependency file must be inside the team configuration directory')
        packages = [line.strip() for line in path.read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith('#')]
        if any(line.startswith('-') for line in packages):
            raise ValueError('Dependency file must contain package specs, not pip options')
    if not isinstance(packages, list):
        raise ValueError('model.pip_dependencies must be a list')
    mode = str(model.get('pip_dependency_mode', 'extend')).strip().lower()
    if mode not in {'extend', 'replace'}:
        raise ValueError('model.pip_dependency_mode must be extend or replace')
    return list(dict.fromkeys(str(p).strip() for p in
                             ([] if mode == 'replace' else defaults) + packages if str(p).strip()))


def unique_model_resources(resources):
    """Deduplicate resources collected from legacy and explicit declarations."""
    import json
    result, seen = [], set()
    for resource in resources:
        key = json.dumps(resource.to_dict(), sort_keys=True)
        if key not in seen:
            result.append(resource)
            seen.add(key)
    return result
