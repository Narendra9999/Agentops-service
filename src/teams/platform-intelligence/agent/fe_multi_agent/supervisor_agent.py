import uuid
import json
import os
from dataclasses import replace
from contextlib import nullcontext, aclosing
from framework.inference.async_runtime import AsyncResponsesAgent, iterate_async_sync, run_blocking, run_sync
from pathlib import Path

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentResponse

from .memory import resolve_memory_context
from .config import Settings
from .graph import Execution, build_graph
from .prompts import load_prompts
from .usage import capture_request_usage, current_meter, install_native_cost_processor


class SupervisorAgent(AsyncResponsesAgent):
    """Customer-owned supervisor with cost intelligence and platform knowledge."""
    def __init__(self, settings=None):
        self.settings = settings or Settings.from_env()

    def load_context(self, context):
        # Databricks notebook import hooks need their loader thread. Resolve
        # optional team modules here, not on an async/executor request thread.
        from . import services, platform_worker, memory_graph, genie_agent
        if "runtime" in (context.artifacts or {}):
            self.settings = Settings(**json.loads(Path(context.artifacts["runtime"]).read_text()))
            self.settings = replace(self.settings, prompt_manifest=context.artifacts["prompts"])
        if self.settings.memory_enabled and not os.getenv('AGENT_CONTEXT_SIGNING_KEY'):
            raise ValueError('Memory-enabled serving requires AGENT_CONTEXT_SIGNING_KEY from a secret reference')
        # Public direct-export destination avoids the serving inference processor's
        # incompatibility with platform-injected OpenTelemetry 1.45. Request/response
        # inference logging remains configured independently on the outer endpoint.
        if self.settings.experiment_id:
            from mlflow.entities import MlflowExperimentLocation
            mlflow.set_tracking_uri("databricks")
            mlflow.tracing.set_destination(MlflowExperimentLocation(self.settings.experiment_id))
        install_native_cost_processor()

    def predict(self, request):
        return run_sync(self.apredict(request))

    @mlflow.trace(name="databricks_specialists.turn", span_type="AGENT")
    @capture_request_usage
    async def apredict(self, request):
        from databricks.sdk import WorkspaceClient
        from .progress import emit, operation
        emit('request_started', 'supervisor', 'Validating the request and preparing the agent team')
        self.settings.validate()
        workspace = await run_blocking(WorkspaceClient)
        actor = (await run_blocking(workspace.current_user.me)).user_name
        if (request.custom_inputs or {}).get("approved_mutation"):
            raise ValueError("Case operations are no longer supported")
        execution = Execution(str(uuid.uuid4()), self.settings.tenant, actor)
        observer = None
        if self.settings.observer_enabled:
            from framework.observability.agent_observer import AgentObserver
            observer = AgentObserver(enabled=True)
        prompts = await run_blocking(load_prompts, self.settings.prompt_manifest)
        messages = []
        for item in request.input:
            data = item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else item
            if data.get("role") not in {"user", "assistant"}:
                raise ValueError("Only user and assistant messages are accepted")
            content = data.get("content", "")
            if isinstance(content, list):
                content = "\n".join(p.get("text", "") for p in content if p.get("type") in {"input_text", "output_text", "text"})
            messages.append({"role": data["role"], "content": content})
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("A final user message is required")
        turn_settings, application_context, memory_mode = resolve_memory_context(
            self.settings, request.custom_inputs, messages)
        if memory_mode == 'stateless':
            prompts = {**prompts, 'supervisor': prompts['supervisor'] +
                '\nThis request has no authenticated memory context. Persistent memory and '
                'App history are unavailable. Use only the supplied conversation and worker '
                'results. Do not call memory tools or claim to recall, save, or delete memory. '
                'If asked to manage memory, explain that the authenticated App is required.'}
        graph = await run_blocking(build_graph,turn_settings,workspace,prompts,execution,observer,application_context)
        trace_tags = {"agentops.execution_id": execution.execution_id,
                                         "agentops.system_id": "fe-platform-engineering",
                                         "agentops.observer_enabled": str(self.settings.observer_enabled).lower(),
                                         "agentops.memory.mode": memory_mode}
        mlflow.update_current_trace(tags=trace_tags)
        scope = observer.agent("platform_team", execution.execution_id, span_name="workflow.execution", span_type="CHAIN") if observer else nullcontext()
        with scope:
            result = await graph.ainvoke({"messages": messages}, config={"recursion_limit": self.settings.max_steps, "callbacks": [current_meter()]})
            text = result["messages"][-1].content
        if isinstance(text, list):
            text = "\n".join(p.get("text", "") for p in text if isinstance(p, dict))
        active_span = mlflow.get_current_active_span()
        trace_id = active_span.trace_id if active_span else mlflow.get_last_active_trace_id()
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=str(text), id=str(uuid.uuid4()))],
            custom_outputs={"execution_id": execution.execution_id, "trace_id": trace_id,
                            "memory": {"mode": memory_mode, "persistent": memory_mode == 'persistent'},
                            "genie_result": result.get("genie_result"), "tags": trace_tags},
        )

    @mlflow.trace(
        name="supervisor.stream", span_type="AGENT",
        output_reducer=ResponsesAgent.responses_agent_output_reducer,
    )
    def predict_stream(self, request):
        yield from iterate_async_sync(lambda: self.apredict_stream(request))

    async def apredict_stream(self, request):
        async with aclosing(astream_prediction(self.apredict, request)) as stream:
            async for event in stream:
                yield event


def stream_prediction(predict, request):
    async def invoke(request):
        return await run_blocking(predict, request)
    yield from iterate_async_sync(lambda: astream_prediction(invoke, request))


async def astream_prediction(predict, request):
    """Stream public activity; release final output only after middleware accepts it."""
    import asyncio
    from .progress import progress_sink
    from mlflow.types.responses import ResponsesAgentStreamEvent
    loop = asyncio.get_running_loop()
    events = asyncio.Queue(maxsize=64)
    stopped = False
    activity_stream = (getattr(request, 'custom_inputs', None) or {}).get('stream_activity') is True
    response_id = str(uuid.uuid4())
    progress_response = {'id': response_id, 'object': 'response', 'status': 'in_progress', 'output': []}

    def publish(activity):
        if stopped:
            return
        # Only activity can be coalesced when a slow consumer fills the queue;
        # results and failures are delivered separately through the owner task.
        def enqueue():
            if not stopped and not events.full():
                events.put_nowait(activity)
        loop.call_soon_threadsafe(enqueue)

    async def execute():
        with progress_sink(publish):
            return await predict(request)

    task = asyncio.create_task(execute())
    pending = None
    try:
        while True:
            pending = asyncio.create_task(events.get())
            done, _ = await asyncio.wait({task, pending}, timeout=2,
                                          return_when=asyncio.FIRST_COMPLETED)
            if pending in done:
                if activity_stream:
                    yield ResponsesAgentStreamEvent(type='response.in_progress', response=progress_response,
                        custom_outputs={'activity': pending.result()})
            else:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            pending = None
            if task.done() and events.empty():
                break
            if not done and activity_stream:
                yield ResponsesAgentStreamEvent(type='response.in_progress', response=progress_response)
        response = await task
        for index, item in enumerate(response.output):
            yield ResponsesAgentStreamEvent(type='response.output_item.done', item=item, output_index=index,
                                             custom_outputs=response.custom_outputs)
        if activity_stream:
            yield ResponsesAgentStreamEvent(type='response.completed',
                response={'id': response_id, 'object': 'response', 'status': 'completed',
                          **response.model_dump(exclude_none=True)}, custom_outputs=response.custom_outputs)
    except Exception as exc:
        from .services import GenieWaitTimeout
        mlflow.update_current_trace(state='ERROR')
        if not activity_stream:
            raise
        details = {'code': 'execution_failed',
                   'message': 'The agent execution failed. Inspect the trace for the failed operation.'}
        if isinstance(exc, GenieWaitTimeout):
            details = {'code': 'genie_timeout', 'message': str(exc), **exc.details}
        active = mlflow.get_current_active_span()
        if active:
            details['trace_id'] = active.trace_id
        yield ResponsesAgentStreamEvent(type='error', message=details['message'],
                                         custom_outputs={'failure': details})
    finally:
        stopped = True
        if pending is not None:
            pending.cancel()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, *([pending] if pending is not None else []), return_exceptions=True)
