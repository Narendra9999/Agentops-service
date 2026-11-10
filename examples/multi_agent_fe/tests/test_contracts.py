from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import uuid

import pytest
from fe_multi_agent.config import Settings
from framework.observability.agent_observer import AgentObserver


def test_observer_preserves_original_exception_even_when_export_fails():
    original = RuntimeError("application failure")

    @contextmanager
    def broken_export(**kwargs):
        try:
            yield object()
        finally:
            raise ValueError("export failure")

    observer = AgentObserver(True, broken_export)
    with pytest.raises(RuntimeError) as exc:
        with observer.agent("docs", "E1"):
            raise original
    assert exc.value is original


def test_observer_failure_does_not_retry_application():
    def broken(**kw):
        raise OSError("telemetry down")
    calls = []
    with AgentObserver(True, broken).agent("docs", "E1"):
        calls.append("called once")
    assert calls == ["called once"]


def test_disabled_observer_never_initializes_telemetry():
    def forbidden(**kw):
        raise AssertionError("Must not initialize telemetry")
    with AgentObserver(False, forbidden).agent("docs", "E1") as span:
        assert span is None


def test_caller_lineage_is_nested_and_thread_local():
    events = []

    @contextmanager
    def capture(**kw):
        events.append(kw["attributes"])
        yield object()

    observer = AgentObserver(True, capture)
    def run(execution):
        with observer.agent("supervisor", execution):
            with observer.agent("docs", execution):
                pass
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, ["E1", "E2"]))
    for execution in ["E1", "E2"]:
        parent, child = [x for x in events if x["agentops.execution_id"] == execution]
        assert parent["agentops.caller_invocation_id"] == ""
        assert child["agentops.caller_invocation_id"] == parent["agentops.invocation_id"]


def test_config_requires_explicit_resource_bindings():
    with pytest.raises(ValueError, match="PLATFORM_ENDPOINT"):
        Settings().validate()
