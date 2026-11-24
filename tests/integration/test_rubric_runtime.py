from types import SimpleNamespace
from unittest.mock import patch
import pytest
from agent_development.agent_evaluation.evaluation.scorer_loader import _build_rubric_scorer
from framework.evaluation.evaluation_pipeline import raise_if_gate_failed


def test_rubric_captures_judge_helper_and_passes_configured_model_and_tool_evidence():
    with patch('agent_development.agent_evaluation.evaluation.custom_scorers._judge_score', return_value=4) as judge:
        scorer = _build_rubric_scorer('evidence', 'Use tool evidence', 'Score 1-5', 'configured-judge')
        trace = SimpleNamespace(data=SimpleNamespace(spans=[SimpleNamespace(
            span_type='TOOL', name='platform_engineering', outputs='Documented evidence')]))
        assert scorer(inputs={'query': 'Explain tracing'}, outputs='answer', trace=trace) == 4
        assert judge.call_args.kwargs['judge_model'] == 'configured-judge'
        assert 'Documented evidence' in judge.call_args.args[1]


def test_missing_gate_metric_fails_closed_with_readable_error():
    with pytest.raises(RuntimeError, match='MISSING'):
        raise_if_gate_failed(False, {'rubric/mean': {'actual': None, 'threshold': 4, 'passed': False}},
            1.0, .8, True, 'evaluation', 'Gate failed', 'Deployment blocked')
