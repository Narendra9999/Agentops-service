"""Team-owned LangGraph supervisor. AgentOPS wraps this ordinary ResponsesAgent."""
import asyncio
import hashlib
import json
from pathlib import Path
import uuid

from workers import call_genie, call_platform

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse, ResponsesAgentStreamEvent


class PlatformIntelligenceSupervisor(ResponsesAgent):
    def load_context(self, context):
        from databricks.sdk import WorkspaceClient
        from databricks_langchain import ChatDatabricks
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent, ToolNode

        config = context.model_config or {}
        settings = config['supervisor']
        self.timeout = settings.get('timeout_seconds', 210)
        self.max_steps = settings.get('max_steps', 12)
        workers = settings['workers']
        endpoint = workers['platform_engineering']['endpoint_name']
        space_id = workers['cost_intelligence']['space_id']
        for label, value in [('endpoint_name', endpoint), ('space_id', space_id)]:
            if not isinstance(value, str) or not value.strip() or '/' in value:
                raise ValueError(f'{label} must be an existing resource name or ID')
        self.workspace = WorkspaceClient()
        manifest = json.loads((Path(__file__).resolve().parent.parent / 'configs/prompts.json').read_text())
        prompt = manifest['prompts']['supervisor']
        if hashlib.sha256(prompt['template'].encode()).hexdigest() != prompt['sha256']:
            raise ValueError('Supervisor prompt fingerprint mismatch')

        @tool
        async def platform_engineering(task: str) -> str:
            """Ask the platform engineering agent for Databricks guidance and documentation citations."""
            return await call_platform(self.workspace, endpoint, task)

        @tool
        async def cost_intelligence(task: str) -> str:
            """Ask the existing Genie agent about costs, estimates, pricing, or missing workload inputs."""
            return await call_genie(self.workspace, space_id, task)

        model = ChatDatabricks(model=config['llm']['endpoint'], workspace_client=self.workspace,
                               temperature=0, max_retries=0)
        if config['llm'].get('api') == 'ai_gateway':
            model.client.base_url = self.workspace.config.host.rstrip('/') + '/ai-gateway/mlflow/v1/'
        # Routing, tools and graph are owned by this team, not the AgentOPS wrapper.
        tools = ToolNode([platform_engineering, cost_intelligence], handle_tool_errors=False)
        self.graph = create_react_agent(model, tools, prompt=prompt['template'])

    async def _run(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        messages = []
        for item in request.input:
            data = item.model_dump(exclude_none=True) if hasattr(item, 'model_dump') else item
            if data.get('role') not in {'user', 'assistant'}:
                raise ValueError('Only user and assistant conversation messages are accepted')
            content = data.get('content', '')
            if isinstance(content, list):
                content = '\n'.join(p.get('text', '') for p in content
                                    if p.get('type') in {'input_text', 'output_text', 'text'})
            messages.append({'role': data['role'], 'content': content})
        if not messages or messages[-1]['role'] != 'user':
            raise ValueError('A final user message is required')
        async with asyncio.timeout(self.timeout):
            result = await self.graph.ainvoke({'messages': messages},
                                             config={'recursion_limit': self.max_steps})
        final = result['messages'][-1]
        if final.type != 'ai' or getattr(final, 'tool_calls', None):
            raise RuntimeError('Supervisor did not finish with an assistant answer')
        content = final.content
        if isinstance(content, list):
            content = '\n'.join(p.get('text', '') for p in content if isinstance(p, dict))
        if not str(content).strip():
            raise RuntimeError('Supervisor returned no answer')
        return ResponsesAgentResponse(output=[self.create_text_output_item(
            text=str(content), id=str(uuid.uuid4()))])

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        return asyncio.run(self._run(request))

    def predict_stream(self, request: ResponsesAgentRequest):
        response = self.predict(request)
        for item in response.output:
            yield ResponsesAgentStreamEvent(type='response.output_item.done', item=item)
