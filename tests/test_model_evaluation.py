
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)

from src.model_evaluation import (
    load_test_data,
    load_best_model,
    predict_churn,
    calculate_metrics,
    create_classification_report,
    calculate_permutation_importance,
    sha256_file,
)

from src.evaluate_model import FIGURES_DIR


@pytest.fixture(scope="module")
def test_data():
    return load_test_data()


@pytest.fixture(scope="module")
def saved_model(test_data):
    X, _ = test_data
    model, metadata = load_best_model(X=X)
    return model, metadata


@pytest.fixture(scope="module")
def predictions(test_data, saved_model):
    X, _ = test_data
    model, _ = saved_model
    return predict_churn(model, X)


@pytest.fixture(scope="module")
def metrics(test_data, predictions):
    _, y = test_data
    predicted, probabilities = predictions

    return calculate_metrics(
        y, predicted, probabilities
    )


def test_test_customer_count(test_data):
    X, y = test_data
    assert len(X) == 1409
    assert len(y) == 1409


def test_test_feature_count(test_data):
    X, _ = test_data
    assert X.shape[1] == 19


def test_no_target_leakage(test_data):
    X, _ = test_data

    assert {
        "customerID",
        "Churn",
        "ChurnBinary",
    }.isdisjoint(X.columns)


def test_valid_test_labels(test_data):
    _, y = test_data
    assert set(y.unique()) == {0, 1}


def test_saved_model_is_pipeline(saved_model):
    model, _ = saved_model

    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps


def test_model_metadata(saved_model):
    _, metadata = saved_model

    assert metadata["test_set_used"] is False
    assert metadata["training_customers"] == 5634


def test_predictions_length(predictions):
    predicted, probabilities = predictions

    assert len(predicted) == 1409
    assert len(probabilities) == 1409


def test_prediction_values(predictions):
    predicted, _ = predictions

    assert set(np.unique(predicted)).issubset(
        {0, 1}
    )


def test_probability_range(predictions):
    _, probabilities = predictions

    assert np.isfinite(probabilities).all()

    assert (
        (probabilities >= 0)
        & (probabilities <= 1)
    ).all()


def test_metrics_in_valid_range(metrics):
    for name in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "brier_score",
    ]:
        assert 0 <= metrics[name] <= 1


def test_confusion_matrix_total(metrics):
    total = sum(
        metrics[name]
        for name in [
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        ]
    )

    assert total == 1409


def test_classification_report(
    test_data,
    predictions,
):
    _, y = test_data
    predicted, _ = predictions

    report = create_classification_report(
        y, predicted
    )

    assert "Retained" in report
    assert "Churned" in report
    assert "precision" in report


def test_permutation_importance(
    test_data,
    saved_model,
):
    X, y = test_data
    model, _ = saved_model

    # Small diagnostic sample for faster testing.
    # The production report uses the full test set.
    sample = X.iloc[:100]
    sample_y = y.iloc[:100]

    if sample_y.nunique() != 2:
        pytest.skip(
            "Diagnostic sample has only one class."
        )

    result = calculate_permutation_importance(
        model,
        sample,
        sample_y,
        n_repeats=2,
    )

    assert len(result) == X.shape[1]
    assert result["feature"].is_unique

    assert np.isfinite(
        result["importance_mean"]
    ).all()


def test_saved_metrics_exist():
    path = REPORTS_DIR / "evaluation_metrics.json"

    assert path.exists()

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert data["test_customers"] == 1409
    assert "roc_auc" in data


def test_saved_predictions_exist():
    path = REPORTS_DIR / "test_predictions.csv"

    assert path.exists()

    df = pd.read_csv(path)

    assert len(df) == 1409
    assert "churn_probability" in df.columns


def test_importance_csv_exists():
    path = REPORTS_DIR / "permutation_importance.csv"

    assert path.exists()

    df = pd.read_csv(path)

    assert len(df) == 19
    assert "importance_mean" in df.columns


def test_all_figures_exist():
    filenames = [
        "confusion_matrix.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "permutation_importance.png",
    ]

    for filename in filenames:
        path = FIGURES_DIR / filename
        assert path.exists()
        assert path.stat().st_size > 0


def test_evaluation_reports_exist():
    filenames = [
        "model_evaluation_report.txt",
        "classification_report.txt",
    ]

    for filename in filenames:
        assert (REPORTS_DIR / filename).exists()


def test_saved_model_hash():
    metadata_path = (
        MODELS_DIR / "best_model_metadata.json"
    )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert metadata["best_model_sha256"] == (
        sha256_file(
            MODELS_DIR / "best_model.joblib"
        )
    )


def test_model_predictions_reproducible(
    test_data,
    predictions,
):
    X, _ = test_data

    model = joblib.load(
        MODELS_DIR / "best_model.joblib"
    )

    repeated_predictions, repeated_probs = (
        predict_churn(model, X)
    )

    original_predictions, original_probs = (
        predictions
    )

    np.testing.assert_array_equal(
        original_predictions,
        repeated_predictions,
    )

    np.testing.assert_allclose(
        original_probs,
        repeated_probs,
    )
