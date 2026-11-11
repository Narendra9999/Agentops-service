"""Async serving interfaces for existing synchronous customer agents.

Blocking lifecycles run in an executor, preserving contextvars (including MLflow).
A stream is consumed by one worker so generator-owned trace contexts are never
entered and exited on different threads. Buffering is bounded to one event.
"""
import asyncio
from contextlib import aclosing
from functools import partial
from queue import Empty, Full, Queue
from threading import Event

from mlflow.pyfunc import ResponsesAgent


def run_sync(coroutine):
    """MLflow compatibility boundary. Async callers must use the async API.

    A synchronous caller inside a running loop still blocks that caller; execute
    on a separate thread to avoid nested asyncio.run / run_until_complete errors.
    """
    from contextvars import copy_context
    from concurrent.futures import ThreadPoolExecutor
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    context = copy_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(context.run, asyncio.run, coroutine).result()


async def run_blocking(function, /, *args, **kwargs):
    """Offload an SDK/database operation without blocking the caller's event loop.

    Cancelling the await does not roll back an already-running operation.
    """
    return await asyncio.to_thread(partial(function, *args, **kwargs))


async def call_async_method(target, async_name, sync_name, /, *args, **kwargs):
    """Prefer a native async integration, retaining existing sync integrations."""
    from inspect import iscoroutinefunction
    method = getattr(target, async_name, None)
    if iscoroutinefunction(method):
        return await method(*args, **kwargs)
    return await run_blocking(getattr(target, sync_name), *args, **kwargs)


async def iterate_blocking(factory, /, *args, **kwargs):
    """Bridge a blocking iterator; preserve order, exceptions and backpressure."""
    queue = Queue(maxsize=1)
    stopped = Event()
    loop = asyncio.get_running_loop()
    ready = asyncio.Event()

    def send(kind, value):
        while not stopped.is_set():
            try:
                queue.put((kind, value), timeout=0.05)
                loop.call_soon_threadsafe(ready.set)
                return
            except Full:
                pass

    def consume():
        iterator = None
        try:
            iterator = iter(factory(*args, **kwargs))
            while not stopped.is_set():
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                send('item', item)
        except Exception as exc:
            send('error', exc)
        finally:
            try:
                if iterator is not None and hasattr(iterator, 'close'):
                    iterator.close()
            except Exception as exc:
                send('error', exc)
            finally:
                send('done', None)

    worker = asyncio.create_task(run_blocking(consume))
    try:
        while True:
            ready.clear()
            try:
                kind, value = queue.get_nowait()
            except Empty:
                await ready.wait()
                continue
            if kind == 'done':
                break
            if kind == 'error':
                raise value
            yield value
    finally:
        stopped.set()
        # Do not wait indefinitely for an upstream blocking request on cancel.
        # The worker closes the generator when its current operation returns.
        if worker.done():
            await worker
        else:
            worker.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)


class AsyncResponsesAgent(ResponsesAgent):
    """Add awaitable APIs while retaining MLflow's synchronous serving contract."""

    def _async_request_agent(self):
        return self

    async def apredict(self, request):
        return await run_blocking(self._async_request_agent().predict, request)

    async def apredict_stream(self, request):
        async with aclosing(iterate_blocking(self._async_request_agent().predict_stream, request)) as stream:
            async for event in stream:
                yield event
