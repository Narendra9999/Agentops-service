"""One authenticated endpoint invocation policy for shared pipeline stages."""


def invoke_endpoint(endpoint, payload, *, timeout_seconds=240, workspace=None):
    if not endpoint or '/' in endpoint:
        raise ValueError('Serving endpoint must be a name')
    if not isinstance(timeout_seconds, (int, float)) or not 1 <= timeout_seconds <= 3600:
        raise ValueError('request_timeout_seconds must be between 1 and 3600')
    if workspace is None:
        from databricks.sdk import WorkspaceClient
        workspace = WorkspaceClient()
    import requests
    # Use SDK authentication, but issue one HTTP request. SDK API-client retries
    # can otherwise replay an uncertain inference that has already written memory.
    url = workspace.config.host.rstrip('/') + '/serving-endpoints/' + endpoint + '/invocations'
    with requests.post(url, headers=workspace.config.authenticate(), json=payload,
                       timeout=(20, timeout_seconds)) as response:
        response.raise_for_status()
        return response.json()
