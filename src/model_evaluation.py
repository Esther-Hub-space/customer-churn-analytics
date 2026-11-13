
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)

from src.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)

THRESHOLD = 0.50


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_test_data(directory=PROCESSED_DATA_DIR):
    """Load the untouched Phase 6 test split."""

    directory = Path(directory)

    x_path = directory / "X_test.csv"
    y_path = directory / "y_test.csv"

    for path in (x_path, y_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing test file: {path}"
            )

    X = pd.read_csv(x_path)
    labels = pd.read_csv(y_path)

    if "ChurnBinary" not in labels.columns:
        raise ValueError("ChurnBinary target missing.")

    if labels.shape[1] != 1:
        raise ValueError(
            "Target file must contain one column."
        )

    y = labels["ChurnBinary"]

    if X.empty or len(X) != len(y):
        raise ValueError("Invalid test data dimensions.")

    if X.columns.duplicated().any():
        raise ValueError("Duplicate feature columns.")

    forbidden = {
        "customerID",
        "Churn",
        "ChurnBinary",
    }

    if forbidden.intersection(X.columns):
        raise ValueError(
            "Target or identifier leakage detected."
        )

    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("Invalid test labels.")

    if y.nunique() != 2:
        raise ValueError(
            "Both classes required for ROC-AUC."
        )

    return X, y.astype(int)


def load_best_model(
    model_dir=MODELS_DIR,
    X=None,
):
    """Load and verify the trusted saved model."""

    model_dir = Path(model_dir)

    model_path = model_dir / "best_model.joblib"
    metadata_path = (
        model_dir / "best_model_metadata.json"
    )

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    if metadata.get("test_set_used") is not False:
        raise ValueError(
            "Training metadata does not confirm "
            "test-set isolation."
        )

    if metadata.get("selection_source") != (
        "training_cross_validation"
    ):
        raise ValueError(
            "Unexpected model selection source."
        )

    expected_hash = metadata.get(
        "best_model_sha256"
    )

    if sha256_file(model_path) != expected_hash:
        raise ValueError(
            "Saved model hash does not match metadata."
        )

    expected_version = metadata.get(
        "sklearn_version"
    )

    if expected_version != sklearn.__version__:
        raise RuntimeError(
            "scikit-learn version mismatch: "
            f"saved={expected_version}, "
            f"installed={sklearn.__version__}. "
            "Use the matching environment or "
            "retrain the model."
        )

    if X is not None:
        expected_features = metadata.get(
            "input_features"
        )

        if list(X.columns) != expected_features:
            raise ValueError(
                "Test feature columns do not match "
                "training feature columns."
            )

    # Only load artifacts produced by your trusted code.
    model = joblib.load(model_path)

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Model must support predict_proba."
        )

    if not hasattr(model, "named_steps"):
        raise TypeError(
            "Expected a complete sklearn pipeline."
        )

    if not {
        "preprocessor",
        "classifier",
    }.issubset(model.named_steps):
        raise ValueError(
            "Saved pipeline is missing required steps."
        )

    return model, metadata


def predict_churn(model, X, threshold=THRESHOLD):
    """Generate probabilities and binary predictions."""

    if not 0 < threshold < 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    classes = list(model.classes_)

    if set(classes) != {0, 1}:
        raise ValueError(
            "Model classes must be 0 and 1."
        )

    positive_index = classes.index(1)

    probabilities = model.predict_proba(X)[
        :, positive_index
    ]

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Non-finite churn probabilities."
        )

    if not (
        (probabilities >= 0)
        & (probabilities <= 1)
    ).all():
        raise ValueError(
            "Probabilities outside [0, 1]."
        )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return predictions, probabilities


def calculate_metrics(
    y_true,
    predictions,
    probabilities,
):
    """Calculate classification and ranking metrics."""

    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)

    if not (
        len(y_true)
        == len(predictions)
        == len(probabilities)
    ):
        raise ValueError(
            "Prediction lengths do not match."
        )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "test_customers": int(len(y_true)),
        "actual_churned": int(y_true.sum()),
        "predicted_churned": int(
            predictions.sum()
        ),
        "threshold": THRESHOLD,
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true, predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true, predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "average_precision": float(
            average_precision_score(
                y_true, probabilities
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true, probabilities
            )
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def create_classification_report(
    y_true,
    predictions,
):
    """Return sklearn's full classification report."""

    return classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=[
            "Retained",
            "Churned",
        ],
        digits=4,
        zero_division=0,
    )


def calculate_permutation_importance(
    model,
    X,
    y,
    n_repeats=5,
):
    """
    Measure feature importance using ROC-AUC.

    This is a post-selection explanation of the
    locked model, not a model selection procedure.
    """

    result = permutation_importance(
        estimator=model,
        X=X,
        y=y,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance_mean":
            result.importances_mean,
        "importance_std":
            result.importances_std,
    })

    return importance.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)
