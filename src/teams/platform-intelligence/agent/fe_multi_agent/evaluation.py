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


def scorers(judge_model):
    from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety, scorer
    from mlflow.entities import Feedback

    @scorer
    def routing(trace, expectations):
        expected = set(expectations.get("agents", []))
        actual = {s.attributes.get("agentops.agent_id") for s in trace.search_spans()}
        return Feedback(name="routing", value=expected.issubset(actual),
                        rationale=f"Expected={sorted(expected)}, observed={sorted(x for x in actual if x)}")

    model = f"databricks:/{judge_model}"
    return [routing, Correctness(model=model), RelevanceToQuery(model=model), Safety(model=model)]


def run_pair(settings, before_manifest, after_manifest, judge_model, output_dir):
    import mlflow
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
            response = agent.predict(ResponsesAgentRequest(input=[{"role": "user", "content": query}]))
            return {"response": response_text(response)}

        with mlflow.start_run(run_name=f"multi-agent-{label}"):
            mlflow.log_params({"phase": label, "prompt_manifest": Path(manifest).read_text(),
                               "supervisor_model": settings.supervisor_model, "worker_endpoint": settings.platform_endpoint,
                               "genie_space": settings.genie_space_id, "platform_endpoint": settings.platform_endpoint})
            result = mlflow.genai.evaluate(data=scenarios(), predict_fn=predict_fn, scorers=scorers(judge_model))
            results[label] = {"run_id": result.run_id, "metrics": result.metrics}
    (root / "paired-evaluation.json").write_text(json.dumps(results, indent=2, default=str))
    return results
