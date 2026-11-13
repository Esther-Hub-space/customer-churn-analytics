
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from src.config import RANDOM_STATE, TEST_SIZE


TARGET_COLUMN = "ChurnBinary"

EXCLUDED_COLUMNS = [
    "customerID",
    "Churn",
    "ChurnBinary",
]

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


def select_features(df):
    """Separate predictors from the churn target."""

    required = set(
        EXCLUDED_COLUMNS + NUMERIC_COLUMNS
    )

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["customerID"].isna().any():
        raise ValueError("Missing customer IDs.")

    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customer IDs.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Missing churn labels.")

    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Invalid churn labels.")

    X = df.drop(
        columns=EXCLUDED_COLUMNS
    ).copy()

    y = df[TARGET_COLUMN].astype(int).copy()

    for column in NUMERIC_COLUMNS:
        X[column] = pd.to_numeric(
            X[column],
            errors="raise"
        )

    return X, y


def split_dataset(X, y):
    """Create a stratified train/test split."""

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_preprocessor(X):
    """
    Create an unfitted preprocessing pipeline.

    Fit this pipeline only on training data.
    """

    numeric_columns = [
        col for col in NUMERIC_COLUMNS
        if col in X.columns
    ]

    categorical_columns = [
        col for col in X.columns
        if col not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def prepare_training_data(df):
    """
    Split data and fit preprocessing on training only.

    Returns original feature splits, target splits,
    fitted preprocessor and transformed arrays.
    """

    X, y = select_features(df)

    X_train, X_test, y_train, y_test = (
        split_dataset(X, y)
    )

    preprocessor = build_preprocessor(X_train)

    X_train_processed = (
        preprocessor.fit_transform(X_train)
    )

    X_test_processed = (
        preprocessor.transform(X_test)
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
    }
