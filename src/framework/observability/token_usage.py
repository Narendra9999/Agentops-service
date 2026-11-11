"""Provider-reported usage normalization and AgentOPS tags; no token estimation."""
import json
import math
from collections.abc import Mapping


def as_dict(value):
    if isinstance(value, Mapping):
        return dict(value)
    for method in ('model_dump', 'as_dict'):
        if hasattr(value, method):
            return getattr(value, method)()
    return dict(vars(value)) if hasattr(value, "__dict__") else {}


def normalize_usage(value):
    usage = as_dict(value)
    if not any(k in usage for k in ('input_tokens', 'output_tokens', 'prompt_tokens', 'completion_tokens', 'total_tokens')):
        return None
    def count(value):
        try:
            n = int(value)
            return n if n >= 0 and float(value) == n else 0
        except (ValueError, TypeError, OverflowError):
            return 0
    prompt = count(usage.get('input_tokens', usage.get('prompt_tokens', 0)))
    completion = count(usage.get('output_tokens', usage.get('completion_tokens', 0)))
    inputs = as_dict(usage.get('input_tokens_details') or usage.get('prompt_tokens_details') or usage.get('input_token_details'))
    outputs = as_dict(usage.get('output_tokens_details') or usage.get('completion_tokens_details') or usage.get('output_token_details'))
    cached = count(usage.get('cached_tokens', usage.get('cache_read_input_tokens', inputs.get('cached_tokens', inputs.get('cache_read', 0)))))
    reasoning = count(usage.get('reasoning_tokens', usage.get('thinking_tokens', outputs.get('reasoning_tokens', outputs.get('reasoning', 0)))))
    return {'input_tokens': prompt, 'output_tokens': completion,
            'total_tokens': count(usage.get('total_tokens')) or prompt + completion,
            'cached_tokens': min(cached, prompt), 'reasoning_tokens': reasoning}


def response_usage(response):
    value = as_dict(response)
    for source in (value, as_dict(value.get('response')), as_dict(value.get('response_metadata'))):
        for key in ('usage', 'token_usage', 'usage_metadata'):
            normalized = normalize_usage(source.get(key))
            if normalized is not None:
                return normalized
    return None


def usage_tags(usage, *, source='responses_agent', scope='response_aggregate'):
    if usage is None:
        return {}
    return {'agentops.tokens.prompt': str(usage['input_tokens']),
            'agentops.tokens.completion': str(usage['output_tokens']),
            'agentops.tokens.total': str(usage['total_tokens']),
            'agentops.tokens.cached': str(usage['cached_tokens']),
            'agentops.tokens.reasoning': str(usage['reasoning_tokens']),
            'agentops.usage.source': source, 'agentops.usage.scope': scope}


def estimate_cost(usage, rates):
    """Rates are configured USD per million tokens, not a billing assertion."""
    cost = cost_breakdown(usage, rates)
    return cost['total_cost'] if cost is not None else None


def cost_breakdown(usage, rates):
    """MLflow's native USD cost fields, calculated from an explicit rate card."""
    if usage is None or not isinstance(rates, dict):
        return None
    required = ['input', 'output'] + (['cached_input'] if usage['cached_tokens'] else [])
    try:
        if any(not math.isfinite(float(rates[k])) or float(rates[k]) < 0 for k in required):
            return None
        input_cost = ((usage['input_tokens'] - usage['cached_tokens']) * float(rates['input'])
                      + usage['cached_tokens'] * float(rates.get('cached_input', 0))) / 1_000_000
        output_cost = usage['output_tokens'] * float(rates['output']) / 1_000_000
        return {'input_cost': input_cost, 'output_cost': output_cost,
                'total_cost': input_cost + output_cost}
    except (KeyError, ValueError, TypeError):
        return None


def set_native_cost_fallback(span, rates_by_model):
    """Fill missing per-call cost without changing routing or pricing parent spans."""
    if span.span_type not in {'LLM', 'CHAT_MODEL'}:
        return
    if span.get_attribute('mlflow.llm.cost') is not None:
        return
    model = span.get_attribute('mlflow.llm.model')
    usage = normalize_usage(span.get_attribute('mlflow.chat.tokenUsage'))
    cost = cost_breakdown(usage, rates_by_model.get(model))
    if cost is not None:
        span.set_attribute('mlflow.llm.cost', cost)
        span.set_attribute('agentops.cost.source', 'configured_rate_estimate')
        span.set_attribute('agentops.cost.currency', 'USD')


def emit_response_usage(response):
    """Copy a response aggregate once; never add it to child-call totals."""
    usage = response_usage(response)
    if usage is not None:
        import mlflow
        value = as_dict(response)
        summary = as_dict(as_dict(value.get('custom_outputs')).get('usage_metrics'))
        mlflow.update_current_trace(tags=usage_tags(usage, source=summary.get('source', 'responses_agent'), scope=summary.get('scope', 'response_aggregate')))
    return usage
