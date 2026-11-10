"""Observe existing calls without changing inputs, outputs, exceptions or retries."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import uuid

_caller = ContextVar("agentops_observed_invocation", default=None)
logger = logging.getLogger(__name__)


class AgentObserver:
    def __init__(self, enabled=False, start_span=None):
        self.enabled = enabled
        self._start_span = start_span

    @contextmanager
    def agent(self, agent_id, execution_id, metadata=None, *, span_name=None, span_type="AGENT"):
        if not self.enabled:
            yield None
            return
        invocation_id = str(uuid.uuid4())
        attributes = {
            "agentops.agent_id": agent_id,
            "agentops.execution_id": execution_id,
            "agentops.invocation_id": invocation_id,
            "agentops.caller_invocation_id": _caller.get() or "",
        }
        # Metadata is explicitly supplied and must already satisfy caller capture policy.
        attributes.update({f"agentops.{k}": v for k, v in (metadata or {}).items()
                           if f"agentops.{k}" not in attributes})
        manager = None
        try:
            factory = self._start_span
            if factory is None:
                import mlflow
                factory = mlflow.start_span
            manager = factory(name=span_name or f"agent.{agent_id}", span_type=span_type, attributes=attributes)
            span = manager.__enter__()
        except Exception:
            logger.warning("AgentOPS observation unavailable", exc_info=False)
            manager, span = None, None
        token = _caller.set(invocation_id)
        error = (None, None, None)
        try:
            yield span
        except BaseException as exc:
            error = (type(exc), exc, exc.__traceback__)
            raise
        finally:
            _caller.reset(token)
            if manager is not None:
                try:
                    manager.__exit__(*error)
                except Exception:
                    logger.warning("AgentOPS observation export failed", exc_info=False)
