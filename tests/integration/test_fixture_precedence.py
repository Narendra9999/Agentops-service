"""Team fixtures override shared defaults and stale UC evaluation rows."""
from pathlib import Path

from framework.notebook_bootstrap import resolve_fixture_path


def test_fixture_resolution_is_team_first_per_file(tmp_path):
    notebook_dir = tmp_path / 'src/agent_development/agent_evaluation/notebooks'
    shared = notebook_dir.parent / 'evaluation'
    shared.mkdir(parents=True)
    team = tmp_path / 'src/teams/example/fixtures'
    team.mkdir(parents=True)
    for filename in ('golden_dataset.json', 'adversarial_dataset.json'):
        (shared / filename).write_text('[]')
    (team / 'golden_dataset.json').write_text('[{"query":"team"}]')
    assert resolve_fixture_path(str(notebook_dir), str(tmp_path), 'example', 'golden_dataset.json') == str(team / 'golden_dataset.json')
    assert resolve_fixture_path(str(notebook_dir), str(tmp_path), 'example', 'adversarial_dataset.json') == str(shared / 'adversarial_dataset.json')
    assert resolve_fixture_path(str(notebook_dir), str(tmp_path), '', 'golden_dataset.json') == str(shared / 'golden_dataset.json')


def test_predeployment_loads_resolved_fixtures_before_persisting_tables(tmp_path):
    import json
    import pandas as pd
    from unittest.mock import MagicMock
    notebook_dir = tmp_path / 'src/agent_development/agent_evaluation/notebooks'
    shared = notebook_dir.parent / 'evaluation'
    team = tmp_path / 'src/teams/example/fixtures'
    shared.mkdir(parents=True)
    team.mkdir(parents=True)
    for root, query in ((shared, 'shared'), (team, 'team')):
        (root / 'golden_dataset.json').write_text(json.dumps([{'request': query, 'expected_response': 'answer'}]))
    (shared / 'adversarial_dataset.json').write_text(json.dumps([{'query': 'shared attack'}]))
    source = Path('src/agent_development/agent_evaluation/notebooks/PreDeploymentEval.py').read_text()
    block = source[source.index('# Resolve team fixtures'):source.index('# MAGIC ## 2. Load the agent')]
    spark = MagicMock()
    spark.table.side_effect = AssertionError('stale table must not override shipped fixtures')
    scope = {'pd': pd, 'spark': spark, 'catalog': 'catalog', 'schema': 'schema',
             'eval_golden_table': 'golden', 'eval_adversarial_table': 'adversarial',
             'resolve_fixture': lambda filename: resolve_fixture_path(str(notebook_dir), str(tmp_path), 'example', filename)}
    exec(compile(block, '<predeployment-fixtures>', 'exec'), scope)
    assert scope['golden_pd'].iloc[0]['request'] == 'team'
    assert scope['adversarial_pd'].iloc[0]['query'] == 'shared attack'
    assert spark.createDataFrame.call_count == 2
    spark.table.assert_not_called()
