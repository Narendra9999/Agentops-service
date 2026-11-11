"""Authenticated endpoint invocation with synchronous and asynchronous transports."""


def _request(endpoint, timeout_seconds, workspace):
    if not endpoint or '/' in endpoint:
        raise ValueError('Serving endpoint must be a name')
    if not isinstance(timeout_seconds, (int, float)) or not 1 <= timeout_seconds <= 3600:
        raise ValueError('request_timeout_seconds must be between 1 and 3600')
    if workspace is None:
        from databricks.sdk import WorkspaceClient
        workspace = WorkspaceClient()
    url = workspace.config.host.rstrip('/') + '/serving-endpoints/' + endpoint + '/invocations'
    return url, workspace.config.authenticate()


def invoke_endpoint(endpoint, payload, *, timeout_seconds=240, workspace=None):
    import requests
    url, headers = _request(endpoint, timeout_seconds, workspace)
    # One request: never automatically replay an inference that may write memory.
    with requests.post(url, headers=headers, json=payload,
                       timeout=(20, timeout_seconds)) as response:
        response.raise_for_status()
        return response.json()


async def ainvoke_endpoint(endpoint, payload, *, timeout_seconds=240, workspace=None):
    """Native async HTTP; credential refresh runs off the event loop; no retries."""
    import httpx
    from framework.inference.async_runtime import run_blocking
    url, headers = await run_blocking(_request, endpoint, timeout_seconds, workspace)
    timeout = httpx.Timeout(timeout_seconds, connect=20)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
