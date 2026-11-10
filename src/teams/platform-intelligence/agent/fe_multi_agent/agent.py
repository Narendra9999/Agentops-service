import uuid
import json
import os
from dataclasses import replace
from contextlib import nullcontext
from pathlib import Path

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentResponse

from .config import Settings
from .graph import Execution, build_graph
from .prompts import load_prompts
from .usage import capture_request_usage, current_meter, install_native_cost_processor


class PlatformEngineeringAgent(ResponsesAgent):
    """Customer-owned supervisor with cost intelligence and platform knowledge."""
    def __init__(self, settings=None):
        self.settings = settings or Settings.from_env()

    def load_context(self, context):
        if "runtime" in (context.artifacts or {}):
            self.settings = Settings(**json.loads(Path(context.artifacts["runtime"]).read_text()))
            self.settings = replace(self.settings, prompt_manifest=context.artifacts["prompts"])
        # Public direct-export destination avoids the serving inference processor's
        # incompatibility with platform-injected OpenTelemetry 1.45. Request/response
        # inference logging remains configured independently on the outer endpoint.
        if self.settings.experiment_id:
            from mlflow.entities import MlflowExperimentLocation
            mlflow.set_tracking_uri("databricks")
            mlflow.tracing.set_destination(MlflowExperimentLocation(self.settings.experiment_id))
        install_native_cost_processor()

    @mlflow.trace(name="databricks_specialists.turn", span_type="AGENT")
    @capture_request_usage
    def predict(self, request):
        from databricks.sdk import WorkspaceClient
        from .progress import emit, operation
        emit('request_started', 'supervisor', 'Validating the request and preparing the agent team')
        self.settings.validate()
        workspace = WorkspaceClient()
        actor = workspace.current_user.me().user_name
        if (request.custom_inputs or {}).get("approved_mutation"):
            raise ValueError("Case operations are no longer supported")
        execution = Execution(str(uuid.uuid4()), self.settings.tenant, actor)
        observer = None
        if self.settings.observer_enabled:
            from framework.observability.agent_observer import AgentObserver
            observer = AgentObserver(enabled=True)
        prompts = load_prompts(self.settings.prompt_manifest)
        graph = build_graph(self.settings, workspace, prompts, execution, observer)
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
        mlflow.update_current_trace(tags={"agentops.execution_id": execution.execution_id,
                                         "agentops.system_id": "fe-platform-engineering",
                                         "agentops.observer_enabled": str(self.settings.observer_enabled).lower()})
        scope = observer.agent("platform_team", execution.execution_id, span_name="workflow.execution", span_type="CHAIN") if observer else nullcontext()
        with scope:
            result = graph.invoke({"messages": messages}, config={"recursion_limit": self.settings.max_steps, "callbacks": [current_meter()]})
            text = result["messages"][-1].content
        if isinstance(text, list):
            text = "\n".join(p.get("text", "") for p in text if isinstance(p, dict))
        active_span = mlflow.get_current_active_span()
        trace_id = active_span.trace_id if active_span else mlflow.get_last_active_trace_id()
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=str(text), id=str(uuid.uuid4()))],
            custom_outputs={"execution_id": execution.execution_id, "trace_id": trace_id,
                            "genie_result": result.get("genie_result")},
        )

    def predict_stream(self, request):
        yield from stream_prediction(self.predict, request)


def stream_prediction(predict, request):
    import queue
    import threading
    from contextvars import copy_context
    from .progress import progress_sink
    from mlflow.types.responses import ResponsesAgentStreamEvent
    events = queue.Queue()
    stopped = threading.Event()

    def publish(activity):
        if stopped.is_set():
            raise RuntimeError('Execution stream closed; no further steps requested')
        events.put(('activity', activity))

    def execute():
        try:
            with progress_sink(publish):
                response = predict(request)
            events.put(('result', response))
        except Exception as exc:
            from .services import GenieWaitTimeout
            details = {'code': 'execution_failed', 'message': 'The agent execution failed. Inspect the trace for the failed operation.'}
            if isinstance(exc, GenieWaitTimeout):
                details = {'code': 'genie_timeout', 'message': str(exc), **exc.details}
            # The stream returns an error event, so mark its otherwise-successful
            # generator span explicitly. Inner execution spans retain the exception.
            active = mlflow.get_current_active_span()
            if active:
                details['trace_id'] = active.trace_id
            events.put(('error', details))

    context = copy_context()
    threading.Thread(target=lambda: context.run(execute), daemon=True).start()
    try:
        while True:
            try:
                kind, data = events.get(timeout=2)
            except queue.Empty:
                yield ResponsesAgentStreamEvent(type='response.in_progress')
                continue
            if kind == 'activity':
                yield ResponsesAgentStreamEvent(type='response.in_progress', custom_outputs={'activity': data})
            elif kind == 'result':
                for item in data.output:
                    yield ResponsesAgentStreamEvent(type='response.output_item.done', item=item,
                                                    custom_outputs=data.custom_outputs)
                yield ResponsesAgentStreamEvent(type='response.completed', response=data.model_dump(exclude_none=True), custom_outputs=data.custom_outputs)
                return
            else:
                mlflow.update_current_trace(state="ERROR")
                yield ResponsesAgentStreamEvent(type='error', message=data['message'],
                                                custom_outputs={'failure': data})
                return
    finally:
        # Cooperative boundary only: in-flight model/tool calls can still finish.
        stopped.set()
