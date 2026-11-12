"""Remote response accounting, separate from native model-span aggregation."""
import math
from framework.observability.token_usage import as_dict, response_usage, normalize_usage


def remote_usage_record(agent, endpoint, response, invocation_id):
    value = as_dict(response); custom = as_dict(value.get('custom_outputs'))
    summary = as_dict(custom.get('usage_metrics'))
    tags = as_dict(custom.get('tags'))
    if 'estimated_cost_usd' not in summary and tags.get('agentops.cost_usd') is not None:
        try:
            summary['estimated_cost_usd'] = float(tags['agentops.cost_usd'])
            summary['cost_status'] = tags.get('agentops.cost.status', 'estimated')
        except (ValueError, TypeError): pass
    if 'usage_status' not in summary and tags.get('agentops.usage.status'):
        summary['usage_status'] = tags['agentops.usage.status']
    if 'llm_requests' not in summary and tags.get('agentops.llm_requests') is not None:
        try: summary['llm_requests'] = int(tags['agentops.llm_requests'])
        except (ValueError, TypeError): pass
    return {'agent': agent, 'endpoint': endpoint, 'invocation_id': str(invocation_id),
            'execution_id': custom.get('execution_id') or tags.get('agentops.execution_id'),
            'trace_id': custom.get('trace_id'),
            'usage': response_usage(value) or normalize_usage(summary), 'usage_metrics': summary}


def rollup_remote_usage(local, workers, *, local_call_ids=()):
    """Add each worker response aggregate once; never add its nested breakdown too.

    Local callback accounting and remote invocation accounting own disjoint
    execution domains. Span-level metrics are untouched: exporting a remote
    aggregate as a model span would double count joined worker spans in MLflow.
    Overlapping reported call IDs are excluded and marked partial instead of
    guessing how much of an aggregate has already been counted.
    """
    result = {**local, 'by_agent_model': list(local.get('by_agent_model', []))}
    missing = set(local.get('unmetered_services', []))
    seen, covered, records = set(), set(map(str, local_call_ids)), []
    cost = float(local.get('estimated_cost_usd') or 0)
    priced = local.get('estimated_cost_usd') is not None and (local.get('llm_requests', 0) > 0 or not workers)
    cost_partial = local.get('cost_status') in {'partial_estimate', 'unavailable'}
    usage_partial = local.get('usage_status') != 'complete'
    for worker in workers:
        metric = as_dict(worker.get('usage_metrics'))
        identity = str(worker.get('execution_id') or worker['invocation_id'])
        key = (worker['endpoint'], identity)
        if key in seen: continue
        seen.add(key)
        record = dict(worker); records.append(record)
        call_ids = set(map(str, metric.get('model_call_ids') or []))
        if call_ids & covered:
            record['accounting_status'] = 'overlap_excluded'
            missing.add('remote_worker_overlap:' + worker['agent'])
            usage_partial = cost_partial = True
            continue
        covered.update(call_ids)
        usage = worker.get('usage')
        if usage is None:
            record['accounting_status'] = 'usage_missing'
            missing.add('remote_worker:' + worker['agent'])
            usage_partial = cost_partial = True
            continue
        record['accounting_status'] = 'included'
        for field in ('input_tokens', 'output_tokens', 'total_tokens', 'cached_tokens', 'reasoning_tokens'):
            result[field] += usage[field]
        if metric.get('usage_status', 'complete') != 'complete' or metric.get('unmetered_services'):
            missing.add('remote_worker_partial:' + worker['agent']); usage_partial = True
        # Usage alone does not establish the number of underlying model calls.
        count = metric.get('llm_requests')
        if isinstance(count, int) and count >= 0:
            result['llm_requests'] += count
        else:
            result['remote_model_requests_unknown'] = result.get('remote_model_requests_unknown', 0) + 1
        value = metric.get('estimated_cost_usd')
        valid_cost = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
        if valid_cost:
            cost += value; priced = True
            if metric.get('cost_status') not in {'estimated', 'complete', 'actual'}: cost_partial = True
        else:
            cost_partial = True; result['unpriced_calls'] += 1
        result['by_agent_model'].append({'agent': worker['agent'], 'model': 'remote_aggregate',
            **usage, 'requests': count, 'estimated_cost_usd': value if valid_cost else None,
            'scope': 'worker_response_aggregate', 'invocation_id': identity})
    result.update(remote_workers=records, unmetered_services=sorted(missing),
                  usage_status='partial' if usage_partial or missing else 'complete',
                  cost_status=('partial_estimate' if cost_partial or missing else 'estimated') if priced else 'unavailable',
                  estimated_cost_usd=round(cost, 10) if priced else None)
    result['model_call_ids'] = sorted(covered)
    if records:
        result['scope'] = 'observed_model_calls_and_remote_workers'
        result['cost_kind'] = 'local_and_worker_reported_estimate'
    return result
