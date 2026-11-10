"""Real PostgreSQL persistence tests, restricted to a disposable local Unix socket."""
import os
from contextlib import contextmanager
import pytest
from fe_multi_agent.memory import DDL,MemoryStore,MemoryWorkspace


@pytest.fixture
def database():
    dsn=os.getenv('FE_MULTI_MEMORY_TEST_DSN','')
    if not dsn:pytest.skip('Disposable local PostgreSQL DSN not supplied')
    if '/private/tmp/agentops-memory-test-' not in dsn:
        pytest.fail('Memory persistence tests require a disposable local test database')
    import psycopg
    @contextmanager
    def connection():
        with psycopg.connect(dsn,client_encoding='utf8') as conn:yield conn
    with connection() as conn:
        conn.execute('CREATE SCHEMA IF NOT EXISTS app_history')
        for sql in DDL:conn.execute(sql)
        conn.execute('TRUNCATE agent_memory.items,agent_memory.audit,agent_memory.files,agent_memory.file_versions')
    return connection


def test_cross_session_recall_approval_expiry_deletion_and_user_isolation(database):
    alice=MemoryStore(database,'alice','conversation-one','trace-one')
    bob=MemoryStore(database,'bob','conversation-other','trace-other')
    fact='Our scheduled batch runs in eu-west-1.'
    saved=alice.store('user_fact',fact,fact,'Remember this: '+fact,'')
    assert saved['stored'] and saved['status']=='active'
    later=MemoryStore(database,'alice','conversation-two','trace-two')
    assert later.search('batch region')[0]['content']==fact
    assert bob.search('batch region')==[]
    assert not bob.change(saved['id'],'forget','forget memory '+saved['id'])['changed']
    pattern='Collect run frequency before multiplying hours by verified rates.'
    proposed=alice.store('cost_pattern',pattern,pattern,'Estimate costs',pattern)
    assert proposed['status']=='pending'
    assert all(m['id']!=proposed['id'] for m in later.search('run frequency'))
    assert not bob.change(proposed['id'],'approve','approve pattern '+proposed['id'])['changed']
    assert later.change(proposed['id'],'approve','approve pattern '+proposed['id'])['changed']
    assert any(m['id']==proposed['id'] for m in later.search('run frequency'))
    with database() as conn:
        conn.execute("UPDATE agent_memory.items SET expires_at=now()-interval '1 second' WHERE id=%s",(proposed['id'],))
    assert all(m['id']!=proposed['id'] for m in later.search('run frequency'))
    assert later.change(saved['id'],'forget','forget memory '+saved['id'])['changed']
    assert later.search('batch region')==[]
    with database() as conn:
        actions=[r[0] for r in conn.execute('SELECT action FROM agent_memory.audit WHERE owner_id=%s ORDER BY id',('alice',)).fetchall()]
        assert 'store' in actions and 'approve' in actions and 'forget' in actions


def test_concurrent_duplicate_memory_writes_have_one_durable_record(database):
    from concurrent.futures import ThreadPoolExecutor
    content='Batch jobs run daily.'
    def save(i):return MemoryStore(database,'alice',str(i)).store('user_fact',content,content,content,'')
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(save,range(4)))
    assert all(r['stored'] for r in results)
    assert len({r['id'] for r in results})==1
    assert len(MemoryStore(database,'alice','new').search('batch'))==1


def test_markdown_workspace_persistence_conflict_history_and_isolation(database):
    first=MemoryWorkspace(database,'alice','one')
    later=MemoryWorkspace(database,'alice','two')
    other=MemoryWorkspace(database,'bob','three')
    path='/memories/workloads/batch.md'
    content='# Batch\nRegion: eu-west-1'
    assert first.write_file(path,content,0,content,'')['revision']==1
    assert later.read_file(path)['content']==content
    assert other.files()==[] and other.read_file(path)['not_found']
    assert not other.delete_file(path,1)['written']
    assert later.write_file(path,content+'\nFrequency: daily',1,'','')['revision']==2
    assert not first.write_file(path,content,1,'','')['written']
    assert [v['revision'] for v in later.history(path)]==[2,1]
    assert later.delete_file(path,2)['deleted']
    assert later.read_file(path)['not_found']
    assert later.write_file(path,content,3,'','')['revision']==4
    assert later.read_file(path)['content']==content
    with pytest.raises(ValueError):later.read_file('/memories/../bob.md')
    with database() as conn:
        conn.execute("UPDATE agent_memory.files SET expires_at=now()-interval '1 second' WHERE owner_id='alice'")
    assert later.files()==[]


def test_concurrent_file_creation_uses_revision_check(database):
    from concurrent.futures import ThreadPoolExecutor
    def write(i):
        return MemoryWorkspace(database,'alice',str(i)).write_file('/memories/profile.md','# Workload\nDaily batch',0,'','')
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(write,range(4)))
    assert sum(r['written'] for r in results)==1
