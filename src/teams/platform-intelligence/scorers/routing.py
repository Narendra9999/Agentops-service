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
    expected=set((expectations.get('expected_output') or {}).get('agents',[]))
    if not expected:
        raise ValueError('Routing requires expected_output.agents')
    observed={s.attributes.get('agentops.agent_id') for s in trace.search_spans()}
    return expected == (observed & {'cost_estimation','platform_docs'})


def build_scorer():
    from mlflow.genai.scorers import scorer
    return scorer(routing)
