"""Single-use, explicit request state and resource ownership."""
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from framework.inference.async_runtime import request_deadline

@dataclass
class RequestContext:
    """Explicit state and async resources owned by one middleware invocation.

    A context is single-use. Configuration and client instances remain shared;
    callbacks put per-request facts in state, not on the middleware or target.
    """
    state: dict = field(default_factory=dict)
    resources: AsyncExitStack | None = field(default=None, init=False)
    _used: bool = field(default=False, init=False, repr=False)

    @asynccontextmanager
    async def scope(self, timeout_seconds=None):
        if self._used:
            raise ValueError('RequestContext is single-use; create one per request')
        self._used = True
        try:
            async with AsyncExitStack() as resources:
                self.resources = resources
                async with request_deadline(timeout_seconds):
                    yield self
        finally:
            self.resources = None
