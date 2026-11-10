"""Model resources for customer single-agent and multi-agent implementations."""


def language_model_resources(config, resolved_llm_endpoint):
    from mlflow.models.resources import DatabricksServingEndpoint, DatabricksGenieSpace
    multi = config.get('multi_agent') or {}
    if not multi:
        return [DatabricksServingEndpoint(endpoint_name=resolved_llm_endpoint)]
    if multi.get('pattern') != 'langgraph_supervisor':
        raise ValueError('Unsupported multi_agent.pattern')
    settings = multi.get('settings') or {}
    platform = settings.get('platform_endpoint')
    if not platform or platform.startswith('REQUIRED_'):
        raise ValueError('Bind an existing platform_endpoint before registration')
    resources = [DatabricksServingEndpoint(endpoint_name=platform)]
    cost_endpoint = settings.get('cost_endpoint')
    if cost_endpoint:
        resources.append(DatabricksServingEndpoint(endpoint_name=cost_endpoint))
    else:
        space = settings.get('genie_space_id')
        if not space or space.startswith('REQUIRED_'):
            raise ValueError('Bind an existing genie_space_id or cost_endpoint')
        resources.append(DatabricksGenieSpace(genie_space_id=space))
    transport = settings.get('model_transport', 'ai_gateway')
    if transport == 'model_serving':
        names = {settings.get('supervisor_model')}
        if not all(names):
            raise ValueError('Supervisor model binding is required')
        resources.extend(DatabricksServingEndpoint(endpoint_name=name) for name in sorted(names))
    elif transport != 'ai_gateway':
        raise ValueError('Unsupported multi-agent model transport')
    # UC model services need EXECUTE grants for the serving identity. They are
    # not DatabricksServingEndpoint resources; never declare them as such.
    return resources
