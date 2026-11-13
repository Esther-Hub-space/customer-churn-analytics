
import hashlib
import json
import shutil
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)

from src.model_training import (
    CV_FOLDS,
    load_training_data,
    get_model_definitions,
    compare_models,
    select_best_model,
    fit_final_models,
)


def sha256_file(path):
    """Calculate SHA-256 of a file."""
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def save_models(fitted_models, best_name):
    """Save each fitted pipeline and best model."""

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {}

    for name, pipeline in fitted_models.items():
        path = MODELS_DIR / f"{name}.joblib"

        joblib.dump(
            pipeline,
            path,
            compress=3,
        )

        paths[name] = path

    best_path = MODELS_DIR / "best_model.joblib"

    # Copy the already fitted winning pipeline.
    # No additional fitting occurs here.
    shutil.copyfile(
        paths[best_name],
        best_path,
    )

    return paths, best_path


def create_metadata(
    X,
    y,
    comparison,
    best_name,
    best_path,
):
    """Record reproducibility information."""

    train_x_path = (
        PROCESSED_DATA_DIR / "X_train.csv"
    )

    train_y_path = (
        PROCESSED_DATA_DIR / "y_train.csv"
    )

    best_row = comparison.iloc[0]

    metadata = {
        "project": "Customer Churn Analytics",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "best_model": best_name,
        "selection_metric": "mean_roc_auc",
        "selection_source": "training_cross_validation",
        "cross_validation_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "training_customers": int(len(X)),
        "training_churned": int(y.sum()),
        "training_churn_rate": float(y.mean()),
        "input_features": list(X.columns),
        "input_feature_count": int(X.shape[1]),
        "best_cv_roc_auc": float(
            best_row["mean_roc_auc"]
        ),
        "best_cv_f1": float(
            best_row["mean_f1"]
        ),
        "sklearn_version": sklearn.__version__,
        "training_files_sha256": {
            "X_train.csv": sha256_file(
                train_x_path
            ),
            "y_train.csv": sha256_file(
                train_y_path
            ),
        },
        "best_model_sha256": sha256_file(
            best_path
        ),
        "test_set_used": False,
    }

    path = (
        MODELS_DIR / "best_model_metadata.json"
    )

    path.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    return path


def create_training_report(
    comparison,
    best_name,
    X,
    y,
):
    """Generate a human-readable model report."""

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_row = comparison.iloc[0]

    lines = [
        "CUSTOMER CHURN - MODEL TRAINING REPORT",
        "=" * 60,
        "",
        "1. TRAINING DATA",
        f"Training customers: {len(X):,}",
        f"Input features: {X.shape[1]}",
        f"Churned customers: {int(y.sum()):,}",
        f"Churn rate: {y.mean() * 100:.2f}%",
        "",
        "2. TRAINING METHODOLOGY",
        f"Cross-validation: {CV_FOLDS}-fold stratified",
        "Preprocessing: fitted separately in each fold",
        "Model selection: mean cross-validation ROC-AUC",
        "Held-out test data: not used",
        "",
        "3. MODEL COMPARISON",
        comparison.to_string(index=False),
        "",
        "4. SELECTED MODEL",
        f"Best model: {best_name}",
        (
            "Cross-validation ROC-AUC: "
            f"{best_row['mean_roc_auc']:.4f}"
        ),
        (
            "Cross-validation F1: "
            f"{best_row['mean_f1']:.4f}"
        ),
        "",
        "5. IMPORTANT LIMITATIONS",
        "- CV scores estimate validation performance.",
        "- These are not held-out test scores.",
        "- No hyperparameter search was performed.",
        "- Models were trained on a customer snapshot.",
        "- Churn associations do not prove causation.",
        "- Final evaluation occurs in Phase 8.",
    ]

    report_path = (
        REPORTS_DIR / "model_training_report.txt"
    )

    report_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return report_path


def main():
    print("=" * 65)
    print("CUSTOMER CHURN - MACHINE LEARNING TRAINING")
    print("=" * 65)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X, y = load_training_data()

    print(f"\nTraining customers: {len(X):,}")
    print(f"Input features: {X.shape[1]}")
    print(f"Training churn rate: {y.mean() * 100:.2f}%")

    definitions = get_model_definitions()

    print("\nSTEP 1: CROSS-VALIDATION")

    comparison = compare_models(
        X,
        y,
        model_definitions=definitions,
    )

    comparison_path = (
        REPORTS_DIR / "model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print("\nMODEL COMPARISON")
    print(
        comparison[
            [
                "model",
                "mean_roc_auc",
                "mean_f1",
                "mean_recall",
                "mean_precision",
                "mean_accuracy",
            ]
        ].round(4).to_string(index=False)
    )

    best_name = select_best_model(comparison)

    print(f"\nSelected model: {best_name}")

    print("\nSTEP 2: FIT FINAL TRAINING MODELS")

    fitted_models = fit_final_models(
        X,
        y,
        model_definitions=definitions,
    )

    model_paths, best_path = save_models(
        fitted_models,
        best_name,
    )

    metadata_path = create_metadata(
        X,
        y,
        comparison,
        best_name,
        best_path,
    )

    report_path = create_training_report(
        comparison,
        best_name,
        X,
        y,
    )

    print("\nSAVED MODELS")

    for name, path in model_paths.items():
        print(f"{name}: {path}")

    print(f"\nBest model: {best_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Comparison: {comparison_path}")
    print(f"Report: {report_path}")

    print("\nTEST DATA WAS NOT USED.")
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
