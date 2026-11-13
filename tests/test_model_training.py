
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    PROCESSED_DATA_DIR,
)
from src.model_training import (
    MODEL_ORDER,
    load_training_data,
    get_model_definitions,
    build_model_pipeline,
    evaluate_model,
    compare_models,
    select_best_model,
    fit_final_models,
)
from src.train_models import sha256_file


@pytest.fixture(scope="module")
def training_data():
    return load_training_data()


@pytest.fixture(scope="module")
def small_data(training_data):
    X, y = training_data

    # Deterministic subset for faster tests
    rng = np.random.default_rng(42)

    selected = rng.choice(
        len(X),
        size=300,
        replace=False,
    )

    return (
        X.iloc[selected].reset_index(drop=True),
        y.iloc[selected].reset_index(drop=True),
    )


@pytest.fixture(scope="module")
def small_cv_result(small_data):
    X, y = small_data

    pipeline = build_model_pipeline(
        X,
        LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
    )

    return evaluate_model(
        pipeline,
        X,
        y,
        n_splits=2,
    )


def test_training_customer_count(training_data):
    X, y = training_data

    assert len(X) == 5634
    assert len(y) == 5634


def test_training_feature_count(training_data):
    X, _ = training_data

    assert X.shape[1] == 19


def test_no_target_leakage(training_data):
    X, _ = training_data

    forbidden = {
        "customerID",
        "Churn",
        "ChurnBinary",
    }

    assert forbidden.isdisjoint(X.columns)


def test_valid_training_labels(training_data):
    _, y = training_data

    assert set(y.unique()) == {0, 1}


def test_three_models_defined():
    models = get_model_definitions()

    assert set(models.keys()) == set(MODEL_ORDER)


def test_pipeline_structure(small_data):
    X, _ = small_data

    pipeline = build_model_pipeline(
        X,
        LogisticRegression(max_iter=1000),
    )

    assert isinstance(pipeline, Pipeline)

    assert list(pipeline.named_steps) == [
        "preprocessor",
        "classifier",
    ]


def test_pipeline_is_initially_unfitted(small_data):
    X, _ = small_data

    pipeline = build_model_pipeline(
        X,
        LogisticRegression(max_iter=1000),
    )

    assert not hasattr(
        pipeline.named_steps["classifier"],
        "coef_",
    )


def test_cross_validation_metrics(small_cv_result):
    for metric in [
        "roc_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]:
        assert f"mean_{metric}" in small_cv_result
        assert f"std_{metric}" in small_cv_result


def test_cross_validation_score_range(
    small_cv_result
):
    for metric in [
        "roc_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]:
        value = small_cv_result[f"mean_{metric}"]

        assert 0 <= value <= 1


def test_model_selection():
    comparison = pd.DataFrame({
        "model": [
            "logistic_regression",
            "random_forest",
        ],
        "mean_roc_auc": [0.82, 0.85],
        "mean_f1": [0.70, 0.68],
    })

    comparison = comparison.sort_values(
        ["mean_roc_auc", "mean_f1", "model"],
        ascending=[False, False, True],
    )

    assert select_best_model(comparison) == (
        "random_forest"
    )


def test_small_model_comparison(small_data):
    X, y = small_data

    models = {
        "logistic_regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            )
    }

    result = compare_models(
        X,
        y,
        model_definitions=models,
        n_splits=2,
    )

    assert len(result) == 1
    assert "mean_roc_auc" in result.columns


def test_final_pipeline_fit(small_data):
    X, y = small_data

    models = {
        "logistic_regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            )
    }

    fitted = fit_final_models(
        X,
        y,
        model_definitions=models,
    )

    pipeline = fitted["logistic_regression"]

    check_is_fitted(
        pipeline.named_steps["classifier"]
    )

    predictions = pipeline.predict(X.iloc[:10])

    assert len(predictions) == 10
    assert set(predictions).issubset({0, 1})


def test_saved_models_exist():
    filenames = [
        "logistic_regression.joblib",
        "random_forest.joblib",
        "hist_gradient_boosting.joblib",
        "best_model.joblib",
    ]

    for filename in filenames:
        path = MODELS_DIR / filename

        assert path.exists()
        assert path.stat().st_size > 0


def test_saved_best_model_predicts(training_data):
    X, _ = training_data

    model = joblib.load(
        MODELS_DIR / "best_model.joblib"
    )

    predictions = model.predict(
        X.iloc[:10]
    )

    probabilities = model.predict_proba(
        X.iloc[:10]
    )

    assert len(predictions) == 10
    assert probabilities.shape == (10, 2)

    assert np.isfinite(probabilities).all()

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(10),
    )


def test_comparison_file_exists():
    path = REPORTS_DIR / "model_comparison.csv"

    assert path.exists()

    comparison = pd.read_csv(path)

    assert len(comparison) == 3
    assert "mean_roc_auc" in comparison.columns


def test_metadata_integrity():
    path = (
        MODELS_DIR / "best_model_metadata.json"
    )

    assert path.exists()

    metadata = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert metadata["training_customers"] == 5634
    assert metadata["input_feature_count"] == 19
    assert metadata["test_set_used"] is False

    assert metadata["best_model"] in MODEL_ORDER

    assert metadata["best_model_sha256"] == (
        sha256_file(
            MODELS_DIR / "best_model.joblib"
        )
    )

    assert metadata["training_files_sha256"][
        "X_train.csv"
    ] == sha256_file(
        PROCESSED_DATA_DIR / "X_train.csv"
    )

    assert metadata["training_files_sha256"][
        "y_train.csv"
    ] == sha256_file(
        PROCESSED_DATA_DIR / "y_train.csv"
    )
