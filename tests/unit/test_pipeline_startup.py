"""Startup policy is independent of team orchestration and safe across restarts."""
from types import SimpleNamespace as NS
import os
import sys
import pytest
from framework import notebook_bootstrap as bootstrap


def dbutils_for(path, source):
    widgets = NS(text=lambda *a: None, get=lambda _: source)
    context = NS(notebookPath=lambda: NS(get=lambda: path))
    notebook = NS(getContext=lambda: context)
    entry = NS(getDbutils=lambda: NS(notebook=lambda: notebook))
    return NS(widgets=widgets, notebook=NS(entry_point=entry))


@pytest.mark.parametrize('source', ['volume', 'environment'])
@pytest.mark.parametrize('prefix', ['', '/Workspace'])
def test_startup_is_repeatable_after_restart(monkeypatch, source, prefix):
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setenv('AGENTOPS_DEPENDENCY_SOURCE', 'stale')
    dbutils = dbutils_for(prefix + '/Shared/project/src/agent_development/agent/notebooks/RegisterModel', source)
    first = bootstrap.initialize_notebook(dbutils)
    assert first['src_root'] == '/Workspace/Shared/project/src'
    assert first['bundle_root'] == '/Workspace/Shared/project'
    assert os.environ['AGENTOPS_DEPENDENCY_SOURCE'] == source
    sys.path.remove(first['src_root'])  # restart clears Python import setup
    assert bootstrap.initialize_notebook(dbutils) == first
    assert sys.path.count(first['src_root']) == 1


def test_invalid_dependency_source_is_rejected(monkeypatch):
    monkeypatch.setattr(sys, 'path', list(sys.path))
    with pytest.raises(ValueError, match='dependency_source'):
        bootstrap.initialize_notebook(dbutils_for('/Shared/p/src/a/b/notebooks/n', 'internet'))


def test_environment_policy_never_looks_up_volume_or_installs(monkeypatch):
    monkeypatch.setenv('AGENTOPS_DEPENDENCY_SOURCE', 'environment')
    def forbidden(*args, **kwargs):
        pytest.fail('environment-managed dependencies must not be reinstalled')
    monkeypatch.setattr(bootstrap, 'get_pip_find_links', forbidden)
    monkeypatch.setattr(bootstrap.subprocess, 'check_call', forbidden)
    assert bootstrap.install_air_gapped_packages(['mlflow']) == ''


def test_volume_remains_default_and_offline(monkeypatch):
    monkeypatch.delenv('AGENTOPS_DEPENDENCY_SOURCE', raising=False)
    monkeypatch.setattr(bootstrap, 'get_pip_find_links', lambda _: '/Volumes/offline')
    calls = []
    monkeypatch.setattr(bootstrap.subprocess, 'check_call', calls.append)
    assert bootstrap.install_air_gapped_packages(['mlflow']) == '/Volumes/offline'
    assert calls[0][-4:] == ['--find-links', '/Volumes/offline', '--no-index', '-q']


def test_framework_wheel_built_once_from_isolated_sources(monkeypatch, tmp_path):
    import framework.mlops_utils as utils
    monkeypatch.setattr(utils, 'stage_framework_source', lambda _: str(tmp_path))
    calls = []
    def build(command, **kwargs):
        calls.append(command)
        (tmp_path / 'wheels/agentops_framework-2.1.8-py3-none-any.whl').touch()
    monkeypatch.setattr('subprocess.run', build)
    result = utils.build_framework_wheel('bundle')
    assert result.endswith('.whl') and len(calls) == 1
    assert '--no-build-isolation' in calls[0]
