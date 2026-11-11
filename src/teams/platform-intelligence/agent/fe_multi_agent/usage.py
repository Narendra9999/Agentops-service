"""Request-scoped callback aggregation without altering customer routing."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import logging
import threading

from langchain_core.callbacks.base import BaseCallbackHandler
from framework.observability.token_usage import as_dict, normalize_usage, estimate_cost, usage_tags, set_native_cost_fallback

_agent = ContextVar('agentops_usage_agent', default='unattributed')
_meter = ContextVar('agentops_usage_meter', default=None)
logger = logging.getLogger(__name__)
_processor_lock = threading.Lock()


def native_cost_processor(span):
    meter = current_meter()
    if meter is not None:
        set_native_cost_fallback(span, meter.rates)


def install_native_cost_processor():
    """Install once, preserving existing processors and their redaction policies."""
    import mlflow
    from mlflow.tracing.config import get_config
    with _processor_lock:
        processors = list(get_config().span_processors)
        if native_cost_processor not in processors:
            mlflow.tracing.configure(span_processors=[*processors, native_cost_processor])


@contextmanager
def agent_usage_scope(agent):
    token = _agent.set(agent)
    try:
        yield
    finally:
        _agent.reset(token)


def mark_unmetered(service):
    meter = _meter.get()
    if meter:
        with meter.lock:
            meter.unmetered.add(service)


def extract_llm_usage(response):
    output = as_dict(getattr(response, 'llm_output', None))
    candidates = [output.get('usage'), output.get('token_usage'), output]
    for group in getattr(response, 'generations', []) or []:
        for generation in group if isinstance(group, list) else [group]:
            message = getattr(generation, 'message', None)
            metadata = as_dict(getattr(message, 'response_metadata', None))
            candidates.extend([getattr(message, 'usage_metadata', None), metadata.get('usage'), metadata.get('token_usage'), metadata])
    # Provider totals and generation metadata describe the same call; take one.
    return next((usage for candidate in candidates if (usage := normalize_usage(candidate)) is not None), None)


class UsageMeter(BaseCallbackHandler):
    run_inline = True
    def __init__(self, rates=None):
        self.rates = rates or {}
        self.lock = threading.Lock()
        self.calls, self.started, self.unmetered = {}, {}, set()
        self.remote_workers = []

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        serialized = serialized or {}
        params = kwargs.get('invocation_params') or {}
        metadata = kwargs.get('metadata') or {}
        model = params.get('model') or params.get('model_name') or metadata.get('ls_model_name') or serialized.get('kwargs', {}).get('model') or 'unknown'
        with self.lock:
            self.started[str(run_id)] = (_agent.get(), str(model))

    def on_llm_end(self, response, *, run_id, **kwargs):
        usage = extract_llm_usage(response)
        with self.lock:
            agent, model = self.started.pop(str(run_id), ('unattributed', 'unknown'))
            self.calls.setdefault(str(run_id), {'agent': agent, 'model': model, 'usage': usage})

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self.lock:
            agent, model = self.started.pop(str(run_id), ('unattributed', 'unknown'))
            self.calls.setdefault(str(run_id), {'agent': agent, 'model': model, 'usage': None})

    def observe_remote(self, agent, endpoint, response):
        from framework.observability.token_usage import response_usage
        custom = as_dict(response.get('custom_outputs'))
        reported = response_usage(response)
        with self.lock:
            self.remote_workers.append({'agent':agent,'endpoint':endpoint,
                                        'trace_id':custom.get('trace_id'), 'usage':reported,
                                        'usage_metrics':custom.get('usage_metrics')})
            self.unmetered.add('remote_worker:' + agent)

    def summary(self):
        with self.lock:
            calls, unmetered = list(self.calls.values()), sorted(self.unmetered)
        totals = dict.fromkeys(('input_tokens','output_tokens','total_tokens','cached_tokens','reasoning_tokens'), 0)
        groups, missing, unpriced, cost = {}, 0, 0, 0.0
        for call in calls:
            key = (call['agent'], call['model'])
            group = groups.setdefault(key, {'agent': key[0], 'model': key[1], 'requests': 0, **dict.fromkeys(totals, 0), 'estimated_cost_usd': 0.0, 'usage_missing': 0})
            group['requests'] += 1
            usage = call['usage']
            if usage is None:
                missing += 1
                group['usage_missing'] += 1
                continue
            for name in totals:
                totals[name] += usage[name]
                group[name] += usage[name]
            value = estimate_cost(usage, self.rates.get(call['model']))
            if value is None:
                unpriced += 1
            else:
                cost += value
                group['estimated_cost_usd'] += value
        # A subtotal is useful, but must never be labeled complete when Genie or
        # another call has no reported tokens/rates. Missing is not zero usage.
        partial = bool(missing or unpriced or unmetered)
        no_priced_usage = bool(calls or unmetered) and len(calls) == missing + unpriced
        return {**totals, 'llm_requests': len(calls), 'usage_missing_calls': missing,
                'usage_status': 'partial' if missing or unmetered else 'complete',
                'cost_status': 'unavailable' if no_priced_usage else 'partial_estimate' if partial else 'estimated',
                'estimated_cost_usd': None if no_priced_usage else round(cost, 10), 'unpriced_calls': unpriced,
                'unmetered_services': unmetered, 'by_agent_model': list(groups.values()),
                'source': 'provider_reported', 'scope': 'observed_model_calls',
                'remote_workers': list(self.remote_workers)}


def current_meter():
    return _meter.get()


def summary_tags(summary):
    """Compact request totals; detailed agent/model accounting belongs in the response."""
    tags = usage_tags(summary, source='provider_reported', scope='observed_model_calls')
    tags.update({'agentops.llm_requests': str(summary['llm_requests']),
                 'agentops.cost.kind': 'configured_rate_estimate',
                 'agentops.cost.status': summary['cost_status'],
                 'agentops.usage.status': summary['usage_status'],
                 'agentops.usage.missing_calls': str(summary['usage_missing_calls'])})
    if summary['estimated_cost_usd'] is not None:
        tags['agentops.cost_usd'] = f"{summary['estimated_cost_usd']:.8f}"
    return tags


@contextmanager
def _request_usage(settings):
    import mlflow
    install_native_cost_processor()
    meter = UsageMeter(getattr(settings, 'usage_rates_per_million', {}))
    token = _meter.set(meter)
    try:
        yield meter
    finally:
        try:
            mlflow.update_current_trace(tags=summary_tags(meter.summary()))
        except Exception:
            logger.warning('Usage tag export failed', exc_info=False)
        finally:
            _meter.reset(token)


def _response_with_usage(response, meter):
    summary = meter.summary()
    payload = response.model_dump(exclude_none=True)
    payload['usage'] = {k: summary[k] for k in ('input_tokens', 'output_tokens', 'total_tokens')}
    payload['usage']['input_tokens_details'] = {'cached_tokens': summary['cached_tokens']}
    payload['usage']['output_tokens_details'] = {'reasoning_tokens': summary['reasoning_tokens']}
    custom = payload.get('custom_outputs') or {}
    tags = {**custom.get('tags', {}), **summary_tags(summary)}
    payload['custom_outputs'] = {**custom, 'usage_metrics': summary, 'tags': tags}
    return type(response)(**payload)


def capture_request_usage(function):
    from inspect import iscoroutinefunction
    if iscoroutinefunction(function):
        @wraps(function)
        async def wrapped(self, *args, **kwargs):
            with _request_usage(self.settings) as meter:
                return _response_with_usage(await function(self, *args, **kwargs), meter)
    else:
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            with _request_usage(self.settings) as meter:
                return _response_with_usage(function(self, *args, **kwargs), meter)
    return wrapped
