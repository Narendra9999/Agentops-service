"""Connection boundary checks; durable persistence is tested against live FEVM Lakebase."""
from types import SimpleNamespace as NS
import pytest
from fe_multi_agent.memory import lakebase_connection


@pytest.mark.parametrize('endpoint',['localhost','postgresql://localhost/example','example.com',''])
def test_memory_rejects_non_lakebase_connection_targets(endpoint):
    with pytest.raises(ValueError,match='Lakebase project endpoint'):
        with lakebase_connection(NS(),endpoint):
            pytest.fail('An external database connection must never be attempted')


def test_lakebase_uses_endpoint_oauth_tls_and_transaction_cleanup(monkeypatch):
    import pg8000.dbapi
    calls=[]
    class Connection:
        def commit(self):calls.append('commit')
        def rollback(self):calls.append('rollback')
        def close(self):calls.append('close')
    def connect(**kwargs):
        assert kwargs['host']=='lakebase.example'
        assert kwargs['password']=='test-ephemeral-oauth'
        assert kwargs['ssl_context'].check_hostname
        assert kwargs['user']=='runtime-identity'
        return Connection()
    monkeypatch.setattr(pg8000.dbapi,'connect',connect)
    def endpoint(**kwargs):
        assert kwargs['name'].startswith('projects/')
        return NS(status=NS(hosts=NS(host='lakebase.example')))
    def credential(**kwargs):
        assert kwargs['endpoint'].startswith('projects/')
        return NS(token='test-ephemeral-oauth')
    w=NS(postgres=NS(get_endpoint=endpoint,generate_database_credential=credential))
    resource='projects/demo/branches/production/endpoints/primary'
    with lakebase_connection(w,resource,user='runtime-identity'):pass
    assert calls==['commit','close']
    calls.clear()
    with pytest.raises(RuntimeError):
        with lakebase_connection(w,resource,user='runtime-identity'):raise RuntimeError('failed')
    assert calls==['rollback','close']
