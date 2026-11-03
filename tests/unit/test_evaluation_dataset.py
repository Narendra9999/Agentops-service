from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from framework.evaluation.evaluation_pipeline import (
    _build_endpoint_inputs,
    _build_expectations,
    _build_inputs,
    prepare_golden_dataset_for_spark,
    restore_golden_dataset_from_spark,
    run_evaluation,
    validate_golden_dataset,
)


def test_validate_golden_dataset_accepts_canonical_schema():
    dataset = pd.DataFrame([
        {
            "request": "What is the total transaction amount?",
            "expected_response": "SELECT SUM(total_spend) FROM transactions",
            "context": "merchant_analysis",
        }
    ])

    validate_golden_dataset(dataset)


def test_validate_golden_dataset_accepts_expected_output():
    dataset = pd.DataFrame([
        {
            "request": "What is the total transaction amount?",
            "expected_response": "SELECT SUM(total_spend) FROM transactions",
            "expected_output": 81.01,
            "context": "merchant_analysis",
        }
    ])

    validate_golden_dataset(dataset)
    assert _build_expectations(dataset) == [{
        "expected_response": "SELECT SUM(total_spend) FROM transactions",
        "expected_output": 81.01,
    }]
    assert _build_inputs(dataset) == [{
        "query": "What is the total transaction amount?",
        "context": "merchant_analysis",
    }]
    assert _build_endpoint_inputs(dataset) == [{
        "input": [{
            "role": "user",
            "content": "What is the total transaction amount?",
        }],
    }]


def test_endpoint_evaluation_passes_native_responses_inputs_to_mlflow():
    dataset = pd.DataFrame([
        {"request": "Show monthly transaction volume", "expected_response": "Monthly volume summary"},
        {"request": "Show quarterly spend", "expected_response": "Quarterly spend summary"},
    ])
    with patch("framework.evaluation.evaluation_pipeline.mlflow.genai.to_predict_fn") as factory, patch(
        "framework.evaluation.evaluation_pipeline.mlflow.genai.evaluate",
        return_value=SimpleNamespace(metrics={}, tables={}),
    ) as evaluate:
        run_evaluation(
            eval_dataset=dataset, model_endpoint="analytics-responses", scorers=[], use_default_thresholds=False,
        )
    factory.assert_called_once_with("endpoints:/analytics-responses")
    assert evaluate.call_args.kwargs["predict_fn"] is factory.return_value
    assert evaluate.call_args.kwargs["data"]["inputs"].tolist() == [
        {"input": [{"role": "user", "content": query}]}
        for query in dataset["request"]
    ]


def test_spark_round_trip_preserves_mixed_expected_output_types():
    dataset = pd.DataFrame([
        {"request": "amount", "expected_response": "SQL 1", "expected_output": 81.01},
        {"request": "count", "expected_response": "SQL 2", "expected_output": 9800},
        {"request": "status", "expected_response": "SQL 3", "expected_output": "approved"},
        {"request": "rows", "expected_response": "SQL 4", "expected_output": [{"count": 1}]},
    ], dtype=object)

    normalized = prepare_golden_dataset_for_spark(dataset)
    restored = restore_golden_dataset_from_spark(normalized)

    assert normalized["expected_output"].tolist() == [
        "81.01",
        "9800",
        '"approved"',
        '[{"count": 1}]',
    ]
    assert all(
        isinstance(record["expected_output"], str)
        for record in normalized.to_dict("records")
    )
    assert restored["expected_output"].tolist() == [
        81.01,
        9800,
        "approved",
        [{"count": 1}],
    ]


@pytest.mark.parametrize("column", ["request", "expected_response"])
def test_validate_golden_dataset_rejects_empty_required_values(column):
    dataset = pd.DataFrame([
        {"request": "question", "expected_response": "answer", "context": "test"}
    ])
    dataset.loc[0, column] = "  "

    with pytest.raises(ValueError, match=rf"'{column}' is empty at row indexes: \[0\]"):
        validate_golden_dataset(dataset)


def test_validate_golden_dataset_rejects_empty_expected_output():
    dataset = pd.DataFrame([{
        "request": "question",
        "expected_response": "answer",
        "expected_output": None,
    }])

    with pytest.raises(ValueError, match=r"'expected_output' is empty at row indexes: \[0\]"):
        validate_golden_dataset(dataset)
