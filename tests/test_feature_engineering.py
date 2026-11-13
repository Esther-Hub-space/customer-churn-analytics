
import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import (
    CLEAN_DATA_FILE,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
)

from src.feature_engineering import (
    select_features,
    split_dataset,
    build_preprocessor,
    prepare_training_data,
)


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CLEAN_DATA_FILE)


@pytest.fixture(scope="module")
def prepared(df):
    return prepare_training_data(df)


def test_total_customers(df):
    assert len(df) == 7043


def test_target_separation(df):
    X, y = select_features(df)

    assert "ChurnBinary" not in X.columns
    assert "Churn" not in X.columns
    assert "customerID" not in X.columns

    assert len(X) == len(y)


def test_original_feature_count(df):
    X, _ = select_features(df)
    assert X.shape[1] == 19


def test_training_size(prepared):
    assert len(prepared["X_train"]) == 5634


def test_testing_size(prepared):
    assert len(prepared["X_test"]) == 1409


def test_no_split_overlap(prepared):
    train_ids = set(
        prepared["X_train"].index
    )
    test_ids = set(
        prepared["X_test"].index
    )

    assert train_ids.isdisjoint(test_ids)


def test_target_values(prepared):
    assert set(
        prepared["y_train"].unique()
    ).issubset({0, 1})

    assert set(
        prepared["y_test"].unique()
    ).issubset({0, 1})


def test_stratified_split(prepared):
    train_rate = prepared["y_train"].mean()
    test_rate = prepared["y_test"].mean()

    assert abs(train_rate - test_rate) < 0.01


def test_processed_feature_dimensions(prepared):
    train = prepared["X_train_processed"]
    test = prepared["X_test_processed"]

    assert train.shape[0] == 5634
    assert test.shape[0] == 1409
    assert train.shape[1] == test.shape[1]


def test_no_missing_processed_values(prepared):
    assert np.isfinite(
        prepared["X_train_processed"]
    ).all()

    assert np.isfinite(
        prepared["X_test_processed"]
    ).all()


def test_preprocessor_is_fitted(prepared):
    names = (
        prepared["preprocessor"]
        .get_feature_names_out()
    )

    assert len(names) > 19


def test_saved_preprocessor_exists():
    path = MODELS_DIR / "preprocessor.joblib"
    assert path.exists()


def test_saved_training_files_exist():
    for name in [
        "X_train.csv",
        "X_test.csv",
        "y_train.csv",
        "y_test.csv",
    ]:
        assert (
            PROCESSED_DATA_DIR / name
        ).exists()


def test_saved_preprocessor_transform(prepared):
    path = MODELS_DIR / "preprocessor.joblib"

    loaded = joblib.load(path)

    transformed = loaded.transform(
        prepared["X_test"]
    )

    assert transformed.shape == (
        prepared["X_test_processed"].shape
    )


def test_unknown_category_handling(df):
    X, y = select_features(df)

    X_train, X_test, _, _ = split_dataset(
        X, y
    )

    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    sample = X_test.iloc[[0]].copy()

    sample["Contract"] = (
        "Previously Unseen Contract"
    )

    result = preprocessor.transform(sample)

    assert np.isfinite(result).all()


def test_training_only_median(df):
    X, y = select_features(df)

    X_train, X_test, _, _ = split_dataset(
        X, y
    )

    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    numeric_pipeline = (
        preprocessor.named_transformers_["numeric"]
    )

    imputer = numeric_pipeline.named_steps[
        "imputer"
    ]

    numeric_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    expected = (
        X_train[numeric_columns]
        .median()
        .to_numpy()
    )

    np.testing.assert_allclose(
        imputer.statistics_,
        expected
    )
