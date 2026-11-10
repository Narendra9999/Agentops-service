"""Application-owned execution status; never model reasoning or raw tool results."""
from contextlib import contextmanager
from contextvars import ContextVar

_sink = ContextVar('execution_progress', default=None)


@contextmanager
def progress_sink(sink):
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


def emit(kind, agent, message, **details):
    sink = _sink.get()
    if sink:
        sink({'kind': kind, 'agent': agent, 'message': message, **details})


@contextmanager
def operation(agent, tool, message):
    emit('tool_started', agent, message, tool=tool)
    try:
        yield
    except Exception:
        emit('tool_failed', agent, 'Tool did not complete successfully', tool=tool)
        raise
    else:
        emit('tool_completed', agent, 'Tool completed', tool=tool)
