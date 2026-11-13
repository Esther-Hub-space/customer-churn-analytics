
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from src.config import (
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)
from src.feature_engineering import (
    build_preprocessor,
    NUMERIC_COLUMNS,
)

CV_FOLDS = 5

MODEL_ORDER = [
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
]

SCORING = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


def load_training_data(
    directory=PROCESSED_DATA_DIR,
):
    """
    Load only Phase 6 training files.

    Held-out test files are never read here.
    """
    x_path = directory / "X_train.csv"
    y_path = directory / "y_train.csv"

    for path in [x_path, y_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}\n"
                "Run: python -m src.prepare_features"
            )

    X = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)

    if "ChurnBinary" not in y_df.columns:
        raise ValueError(
            "Missing ChurnBinary target."
        )

    y = y_df["ChurnBinary"]

    if X.empty or len(X) != len(y):
        raise ValueError(
            "Invalid training dataset dimensions."
        )

    forbidden = {
        "customerID",
        "Churn",
        "ChurnBinary",
    }

    if forbidden.intersection(X.columns):
        raise ValueError(
            "Target or identifier leakage detected."
        )

    if X.columns.duplicated().any():
        raise ValueError(
            "Duplicate feature column names."
        )

    missing_numeric = (
        set(NUMERIC_COLUMNS) - set(X.columns)
    )

    if missing_numeric:
        raise ValueError(
            f"Missing numeric features: "
            f"{sorted(missing_numeric)}"
        )

    for column in NUMERIC_COLUMNS:
        X[column] = pd.to_numeric(
            X[column],
            errors="raise",
        )

    if y.isna().any():
        raise ValueError("Missing training labels.")

    if not y.isin([0, 1]).all():
        raise ValueError("Invalid target labels.")

    if y.nunique() != 2:
        raise ValueError(
            "Both churn classes are required."
        )

    return X, y.astype(int)


def get_model_definitions():
    """Define three reproducible classifiers."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "hist_gradient_boosting":
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            ),
    }


def build_model_pipeline(X, estimator):
    """
    Create an unfitted end-to-end pipeline.

    Preprocessing is fitted within each CV fold.
    """
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(X),
            ),
            (
                "classifier",
                clone(estimator),
            ),
        ]
    )


def evaluate_model(
    pipeline,
    X,
    y,
    n_splits=CV_FOLDS,
):
    """Evaluate using stratified cross-validation."""

    if y.value_counts().min() < n_splits:
        raise ValueError(
            "Not enough samples per class "
            "for cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = cross_validate(
        estimator=pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring=SCORING,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )

    result = {}

    for metric in SCORING:
        values = scores[f"test_{metric}"]

        result[f"mean_{metric}"] = float(
            np.mean(values)
        )

        result[f"std_{metric}"] = float(
            np.std(values)
        )

    return result


def compare_models(
    X,
    y,
    model_definitions=None,
    n_splits=CV_FOLDS,
):
    """Cross-validate and compare all models."""

    if model_definitions is None:
        model_definitions = get_model_definitions()

    results = []

    for name, estimator in model_definitions.items():
        print(f"\nEvaluating: {name}")

        pipeline = build_model_pipeline(
            X,
            estimator,
        )

        metrics = evaluate_model(
            pipeline,
            X,
            y,
            n_splits=n_splits,
        )

        results.append({
            "model": name,
            **metrics,
        })

        print(
            "Mean ROC-AUC:",
            f"{metrics['mean_roc_auc']:.4f}",
        )

        print(
            "Mean F1:",
            f"{metrics['mean_f1']:.4f}",
        )

    comparison = pd.DataFrame(results)

    # ROC-AUC is the primary selection metric.
    # F1 and model name provide deterministic ties.
    comparison = comparison.sort_values(
        by=["mean_roc_auc", "mean_f1", "model"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return comparison


def select_best_model(comparison):
    """Return the top model based on CV results."""

    if comparison.empty:
        raise ValueError(
            "Model comparison is empty."
        )

    return str(comparison.iloc[0]["model"])


def fit_final_models(
    X,
    y,
    model_definitions=None,
):
    """
    Fit final pipelines on all training customers.

    The held-out test set is not used.
    """

    if model_definitions is None:
        model_definitions = get_model_definitions()

    fitted = {}

    for name, estimator in model_definitions.items():
        print(f"Fitting final model: {name}")

        pipeline = build_model_pipeline(
            X,
            estimator,
        )

        pipeline.fit(X, y)

        fitted[name] = pipeline

    return fitted
