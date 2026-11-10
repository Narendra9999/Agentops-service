from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class Settings:
    supervisor_model: str = "system.ai.claude-sonnet-4-6"
    model_transport: str = "ai_gateway"
    genie_space_id: str = ""
    platform_endpoint: str = ""
    platform_mode: str = "remote"
    platform_model: str = "system.ai.gpt-oss-120b"
    docs_index: str = ""
    docs_text_column: str = "chunk_text"
    docs_url_column: str = "url"
    memory_enabled: bool = False
    unsigned_context_policy: str = "reject"
    memory_endpoint: str = ""
    memory_ttl_days: int = 90
    cost_endpoint: str = ""
    experiment_id: str = ""
    prompt_manifest: str = ""
    observer_enabled: bool = False
    tenant: str = "fe-demo"
    max_steps: int = 30
    # Existing AgentOPS rate-card assumptions, USD / million tokens; estimates only.
    usage_rates_per_million: dict = field(default_factory=lambda: {
        "system.ai.claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cached_input": 1.5},
        "system.ai.gpt-oss-120b": {"input": 1.0, "output": 2.0, "cached_input": 0.5},
    })

    @classmethod
    def from_env(cls):
        values = {}
        for name, field in cls.__dataclass_fields__.items():
            value = os.getenv("FE_MULTI_" + name.upper())
            if value is not None:
                if isinstance(field.default, bool):
                    value = value.lower() == "true"
                elif isinstance(field.default, int):
                    value = int(value)
                elif name == "usage_rates_per_million":
                    import json
                    value = json.loads(value)
                values[name] = value
        return cls(**values)

    def validate(self):
        if self.unsigned_context_policy not in {'reject', 'stateless'}:
            raise ValueError('unsigned_context_policy must be reject or stateless')
        if self.model_transport not in {"ai_gateway", "model_serving"}:
            raise ValueError("model_transport must be ai_gateway or model_serving")
        if self.model_transport == "ai_gateway":
            models=[self.supervisor_model]+([self.platform_model] if self.platform_mode=='in_process' else [])
            for model in models:
                if len(model.split('.')) != 3 or '/' in model:
                    raise ValueError("AI Gateway models must use catalog.schema.model_service names")
        if self.platform_mode not in {'remote','in_process'}:
            raise ValueError('platform_mode must be remote or in_process')
        if self.platform_mode == 'in_process' and not self.docs_index:
            raise ValueError('Configure docs_index for in-process retrieval')
        if self.memory_enabled and (not self.memory_endpoint or not 1 <= self.memory_ttl_days <= 365):
            raise ValueError('Configure memory_endpoint and memory_ttl_days (1-365)')
        for name in (("platform_endpoint", "prompt_manifest") if self.platform_mode == 'remote' else ("prompt_manifest",)):
            value = getattr(self, name)
            if not value or "REQUIRED" in value:
                raise ValueError(f"Configure FE_MULTI_{name.upper()}")
        if not self.cost_endpoint and (not self.genie_space_id or "REQUIRED" in self.genie_space_id):
            raise ValueError("Configure FE_MULTI_GENIE_SPACE_ID or FE_MULTI_COST_ENDPOINT")
        for endpoint in (self.platform_endpoint, self.cost_endpoint):
            if endpoint and ("/" in endpoint or endpoint.startswith("http")):
                raise ValueError("Worker bindings must be endpoint names, not URLs")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
