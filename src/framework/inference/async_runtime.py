"""Async serving interfaces for existing synchronous customer agents.

Blocking lifecycles run in an executor, preserving contextvars (including MLflow).
A stream is consumed by one worker so generator-owned trace contexts are never
entered and exited on different threads. Buffering is bounded to one event.
"""
import asyncio
from contextlib import aclosing, asynccontextmanager
from functools import partial
from queue import Empty, Full, Queue
from threading import Event

from mlflow.pyfunc import ResponsesAgent


class _StreamChannel:
    """Shared bounded delivery and cooperative stop for both stream bridges."""

    def __init__(self, notify=None):
        self.queue = Queue(maxsize=1)
        self.stopped = Event()
        self.notify = notify

    def send(self, kind, value):
        while not self.stopped.is_set():
            try:
                self.queue.put((kind, value), timeout=0.05)
                if self.notify is not None:
                    self.notify()
                return
            except Full:
                pass



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
    loop = asyncio.get_running_loop()
    ready = asyncio.Event()
    channel = _StreamChannel(lambda: loop.call_soon_threadsafe(ready.set))

    def consume():
        iterator = None
        try:
            iterator = iter(factory(*args, **kwargs))
            while not channel.stopped.is_set():
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                channel.send('item', item)
        except Exception as exc:
            channel.send('error', exc)
        finally:
            try:
                if iterator is not None and hasattr(iterator, 'close'):
                    iterator.close()
            except Exception as exc:
                channel.send('error', exc)
            finally:
                channel.send('done', None)

    worker = asyncio.create_task(run_blocking(consume))
    try:
        while True:
            ready.clear()
            try:
                kind, value = channel.queue.get_nowait()
            except Empty:
                await ready.wait()
                continue
            if kind == 'done':
                break
            if kind == 'error':
                raise value
            yield value
    finally:
        channel.stopped.set()
        # Do not wait indefinitely for an upstream blocking request on cancel.
        # The worker closes the generator when its current operation returns.
        if worker.done():
            await worker
        else:
            worker.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)


class AsyncResponsesAgent(ResponsesAgent):
    """Add awaitable APIs while retaining MLflow's synchronous serving contract."""

    async def apredict(self, request):
        return await run_blocking(self.predict, request)

    async def apredict_stream(self, request):
        async with aclosing(iterate_blocking(self.predict_stream, request)) as stream:
            async for event in stream:
                yield event


@asynccontextmanager
async def request_deadline(seconds=None):
    """Server-configured whole-request deadline; clients cannot extend it."""
    if seconds is None:
        yield
        return
    import math
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('middleware.timeout_seconds must be finite and positive')
    async with asyncio.timeout(seconds):
        yield


def iterate_async_sync(factory):
    """Consume an async stream from MLflow without buffering the whole response.

    A single producer task owns the async generator and its resource scopes.
    Closing the synchronous iterator cancels that task on its owning loop.
    """
    from contextvars import copy_context
    from threading import Thread
    channel = _StreamChannel()
    owner = {}

    async def produce():
        owner['loop'] = asyncio.get_running_loop()
        owner['task'] = asyncio.current_task()
        try:
            async with aclosing(factory()) as upstream:
                async for item in upstream:
                    if channel.stopped.is_set():
                        break
                    await run_blocking(channel.send, 'item', item)
        except BaseException as exc:
            if not channel.stopped.is_set():
                await run_blocking(channel.send, 'error', exc)
        finally:
            await run_blocking(channel.send, 'done', None)

    context = copy_context()
    def run_owner():
        try:
            context.run(asyncio.run, produce())
        except asyncio.CancelledError:
            pass  # Consumer closed the stream; resource scopes have unwound.

    thread = Thread(target=run_owner, daemon=True)
    thread.start()
    completed = False
    try:
        while True:
            kind, value = channel.queue.get()
            if kind == 'done':
                completed = True
                return
            if kind == 'error':
                raise value
            yield value
    finally:
        channel.stopped.set()
        loop, task = owner.get('loop'), owner.get('task')
        if not completed and loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass  # Owner finished between the check and notification.
        thread.join(timeout=1)
