import pytest
from framework.evaluation.evaluation_pipeline import validate_quality_gate


def test_gate_checks_each_scorer_without_treating_latency_as_quality():
    validate_quality_gate({'safety/mean':1,'cost_grounding/pass_rate':.9,'latency/mean':100},
                          ['safety','cost_grounding'],.8)


@pytest.mark.parametrize('metrics', [
    {'safety/mean':.5}, {'safety/mean':float('nan')},
    {'latency/mean':100}, {'safety/mean':1,'safety/error_count':1},
])
def test_gate_rejects_low_missing_failed_and_nonfinite_scores(metrics):
    with pytest.raises(RuntimeError,match='Evaluation gate failed'):
        validate_quality_gate(metrics,['safety'],.8)


def test_pipeline_registration_request_satisfies_signed_app_contract(monkeypatch):
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[3]/'src/teams/platform-intelligence/request_adapter.py'
    spec=importlib.util.spec_from_file_location('test_request_adapter',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    request=module.build_request
    from fe_multi_agent.memory import verify_context
    monkeypatch.setenv('AGENT_CONTEXT_SIGNING_KEY','unit-test-key')
    payload=request('Read saved region','evaluation:unit-test')
    context=verify_context(payload['custom_inputs']['application_context'],payload['input'])
    assert context['owner_id']=='evaluation:unit-test'
    with pytest.raises(ValueError,match='does not match'):
        verify_context(payload['custom_inputs']['application_context'],[{'role':'user','content':'Changed'}])


def test_row_validation_does_not_allow_failed_predictions_hidden_by_good_means():
    from framework.evaluation.evaluation_pipeline import validate_evaluation_rows
    class Frame:
        columns=['error_message','safety/error_code']
        def __len__(self):return 3
        def __getitem__(self,key):return [None,'prediction failed',None] if key=='error_message' else [None]*3
    with pytest.raises(RuntimeError,match='prediction or scorer errors'):
        validate_evaluation_rows(Frame(),3)
    with pytest.raises(RuntimeError,match='every requested scenario'):
        validate_evaluation_rows(Frame(),4)
