"""Three read-only paired scenarios: cost, platform knowledge, and combined routing."""
from dataclasses import replace
import json
from pathlib import Path


def response_text(response):
    data = response.model_dump() if hasattr(response, "model_dump") else response
    return "\n".join(part["text"] for item in data.get("output", [])
                     for part in item.get("content", []) if part.get("type") == "output_text")


def scenarios():
    return [
        {"inputs": {"query": "A scheduled batch takes 2 hours per run. What inputs do you still need before estimating monthly cost? Do not assume rates or missing workload facts."},
         "expectations": {"agents": ["cost_estimation"], "expected_facts": ["asks for missing workload inputs and run frequency", "does not invent a monetary estimate from incomplete inputs"]}},
        {"inputs": {"query": "Explain how MLflow tracing helps debug a LangGraph supervisor with specialist agents. Cite Databricks documentation."},
         "expectations": {"agents": ["platform_docs"], "expected_facts": ["agent calls and tool operations", "documentation source URLs"]}},
        {"inputs": {"query": "Use Genie to identify the information needed for a monthly batch workload estimate, then retrieve Databricks documentation about serverless jobs. Cite sources and do not invent a quote."},
         "expectations": {"agents": ["cost_estimation", "platform_docs"], "expected_facts": ["missing cost inputs are explicit", "platform guidance is supported by documentation URLs"]}},
    ]


def routing(trace, expectations, outputs=None):
    """Require exactly the expected workers; standalone for scorer serialization."""
    reference=(outputs or {}).get('execution_trace_id')
    if reference and reference!=getattr(getattr(trace,'info',None),'trace_id',None):
        # Serving writes its own trace. Resolve that explicit reference instead
        # of mistaking the caller's HTTP span for the remote execution graph.
        import mlflow
        import time
        remote=None
        for attempt in range(6):
            remote=mlflow.get_trace(reference)
            if remote is not None:break
            if attempt<5:time.sleep(2)
        if remote is None:raise RuntimeError('Serving execution trace is unavailable: '+reference)
        trace=remote
    expected=set(expectations.get('agents',[]))
    observed={s.attributes.get('agentops.agent_id') for s in trace.search_spans()}
    return expected == (observed & {'cost_estimation','platform_docs'})


def scorers(judge_model):
    from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety, scorer
    model = f"databricks:/{judge_model}"
    return [scorer(routing), Correctness(model=model), RelevanceToQuery(model=model), Safety(model=model)]


def run_pair(settings, before_manifest, after_manifest, judge_model, output_dir):
    import mlflow
    import os, time, hashlib
    from mlflow.types.responses import ResponsesAgentRequest
    from .agent import PlatformEngineeringAgent

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, manifest in (("before", before_manifest), ("after", after_manifest)):
        agent = PlatformEngineeringAgent(replace(settings, prompt_manifest=manifest, observer_enabled=True))

        @mlflow.trace(name=f"paired_eval.{label}", span_type="AGENT")
        def predict_fn(query):
            mlflow.update_current_trace(tags={"fe.eval_phase": label})
            messages=[{"role":"user","content":query}]
            custom={}
            if settings.memory_enabled:
                from .memory import sign_context
                owner=os.environ.get('FE_MULTI_EVALUATION_OWNER')
                if not owner or not owner.startswith('evaluation:'):
                    raise ValueError('Memory evaluation requires FE_MULTI_EVALUATION_OWNER=evaluation:<isolated-run-id> and the context signing key')
                context={'owner_id':owner+':'+label,'conversation_id':'paired-evaluation',
                    'expires_at':int(time.time())+300,'summary':'','preferences':{},
                    'messages_sha256':hashlib.sha256(json.dumps(messages,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
                custom={'application_context':sign_context(context)}
            response = agent.predict(ResponsesAgentRequest(input=messages,custom_inputs=custom))
            return {"response": response_text(response)}

        with mlflow.start_run(run_name=f"multi-agent-{label}"):
            mlflow.log_params({"phase": label, "prompt_manifest": Path(manifest).read_text(),
                               "supervisor_model": settings.supervisor_model, "worker_endpoint": settings.platform_endpoint,
                               "genie_space": settings.genie_space_id, "platform_endpoint": settings.platform_endpoint})
            result = mlflow.genai.evaluate(data=scenarios(), predict_fn=predict_fn, scorers=scorers(judge_model))
            results[label] = {"run_id": result.run_id, "metrics": result.metrics}
    (root / "paired-evaluation.json").write_text(json.dumps(results, indent=2, default=str))
    return results
