"""Serving regressions observed with OAuth retrieval and injected OTel versions."""
from types import SimpleNamespace as NS
import importlib.util
from pathlib import Path

import pytest


def test_retrieval_uses_sdk_identity_and_manifest_columns(monkeypatch):
    from databricks import sdk
    spec = importlib.util.spec_from_file_location('search_docs_under_test',
        Path('src/agent_development/agent/tools/search_docs.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def query_index(**kwargs):
        calls.append(kwargs)
        return NS(as_dict=lambda: {'manifest': {'columns': [
            {'name': 'url'}, {'name': 'chunk_text'}, {'name': 'score'}]},
            'result': {'data_array': [['https://docs.databricks.com/example', 'evidence', 0.9]]}})

    # No PAT or separately supplied OAuth credentials: WorkspaceClient owns auth.
    monkeypatch.setattr(sdk, 'WorkspaceClient', lambda: NS(
        vector_search_indexes=NS(query_index=query_index)))
    docs = module.search_docs('question', 'catalog.schema.docs', columns=['chunk_text', 'url'])
    assert docs == [{'chunk_text': 'evidence', 'url': 'https://docs.databricks.com/example'}]
    assert len(calls) == 1 and calls[0]['index_name'] == 'catalog.schema.docs'


def test_experiment_export_is_explicit_and_uses_runtime_experiment(monkeypatch):
    import mlflow
    from framework.observability.tracing import configure_trace_destination
    destinations = []
    monkeypatch.setattr(mlflow, 'set_tracking_uri', lambda uri: None)
    monkeypatch.setattr(mlflow.tracing, 'set_destination', destinations.append)
    monkeypatch.setenv('MLFLOW_EXPERIMENT_ID', 'current-run')
    configure_trace_destination({})
    assert not destinations
    configure_trace_destination({'tracing': {'exporter': 'experiment', 'experiment_id': 'stale'}})
    assert destinations[0].experiment_id == 'current-run'
    monkeypatch.delenv('MLFLOW_EXPERIMENT_ID')
    with pytest.raises(ValueError, match='resolved experiment ID'):
        configure_trace_destination({'tracing': {'exporter': 'experiment'}})
    with pytest.raises(ValueError, match='tracing.exporter'):
        configure_trace_destination({'tracing': {'exporter': 'unknown'}})
