"""Conservative failure categories; raw exception messages never become tags."""

def classify_error(error):
    if isinstance(error, TimeoutError):
        return 'deadline_or_dependency_timeout'
    if isinstance(error, ConnectionError):
        return 'dependency_unavailable'
    return 'execution_failed'
