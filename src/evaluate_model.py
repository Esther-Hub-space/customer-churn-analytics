
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from src.config import (
    REPORTS_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
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

FIGURES_DIR = REPORTS_DIR / "figures"


def save_figure(filename):
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = FIGURES_DIR / filename

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    return path


def plot_confusion_matrix(
    y_true,
    predictions,
):
    fig, ax = plt.subplots(figsize=(7, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=[0, 1],
        display_labels=[
            "Retained",
            "Churned",
        ],
        cmap="Blues",
        values_format="d",
        ax=ax,
    )

    ax.set_title(
        "Customer Churn - Confusion Matrix"
    )

    return save_figure("confusion_matrix.png")


def plot_roc_curve(
    y_true,
    probabilities,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    RocCurveDisplay.from_predictions(
        y_true,
        probabilities,
        name="Selected model",
        ax=ax,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random ranking",
    )

    ax.set_title("ROC Curve - Test Dataset")
    ax.legend(loc="lower right")

    return save_figure("roc_curve.png")


def plot_precision_recall_curve(
    y_true,
    probabilities,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    PrecisionRecallDisplay.from_predictions(
        y_true,
        probabilities,
        name="Selected model",
        ax=ax,
    )

    prevalence = float(
        np.mean(y_true)
    )

    ax.axhline(
        prevalence,
        linestyle="--",
        color="gray",
        label="Churn prevalence",
    )

    ax.set_title(
        "Precision-Recall Curve - Test Dataset"
    )

    ax.legend(loc="best")

    return save_figure(
        "precision_recall_curve.png"
    )


def plot_feature_importance(importance):
    top = importance.head(15).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        color="#247BA0",
        capsize=3,
    )

    ax.axvline(
        0,
        color="black",
        linewidth=0.8,
    )

    ax.set_xlabel(
        "Mean decrease in ROC-AUC when shuffled"
    )

    ax.set_title(
        "Top 15 Permutation Feature Importances"
    )

    return save_figure(
        "permutation_importance.png"
    )


def save_predictions(
    X,
    y,
    predictions,
    probabilities,
):
    """
    Save predictions for analysis.

    Row numbers are local to the held-out split.
    """

    output = pd.DataFrame({
        "test_row": np.arange(len(X)),
        "actual_churn": y.to_numpy(),
        "predicted_churn": predictions,
        "churn_probability": probabilities,
        "correct_prediction":
            predictions == y.to_numpy(),
    })

    path = REPORTS_DIR / "test_predictions.csv"

    output.to_csv(path, index=False)

    return path


def create_evaluation_report(
    metrics,
    metadata,
    importance,
):
    lines = [
        "CUSTOMER CHURN - MODEL EVALUATION REPORT",
        "=" * 60,
        "",
        "1. MODEL INFORMATION",
        f"Selected model: {metadata['best_model']}",
        "Selection: Phase 7 training cross-validation",
        "Preprocessing: saved fitted pipeline",
        "Held-out test set: evaluated without fitting",
        "",
        "2. TEST DATASET",
        f"Customers: {metrics['test_customers']:,}",
        f"Actual churned: {metrics['actual_churned']:,}",
        f"Predicted churned: {metrics['predicted_churned']:,}",
        f"Decision threshold: {metrics['threshold']:.2f}",
        "",
        "3. PERFORMANCE METRICS",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"F1-score: {metrics['f1']:.4f}",
        f"ROC-AUC: {metrics['roc_auc']:.4f}",
        (
            "Average precision: "
            f"{metrics['average_precision']:.4f}"
        ),
        (
            "Brier score: "
            f"{metrics['brier_score']:.4f}"
        ),
        "",
        "4. CONFUSION MATRIX",
        f"True negatives: {metrics['true_negative']}",
        f"False positives: {metrics['false_positive']}",
        f"False negatives: {metrics['false_negative']}",
        f"True positives: {metrics['true_positive']}",
        "",
        "5. TOP PERMUTATION IMPORTANCES",
        importance.head(10).to_string(
            index=False
        ),
        "",
        "6. BUSINESS INTERPRETATION",
        (
            "- Recall measures the proportion of "
            "actual churners identified."
        ),
        (
            "- Precision measures the proportion "
            "of flagged customers who churned."
        ),
        (
            "- False negatives are churned customers "
            "the model did not flag."
        ),
        (
            "- ROC-AUC measures ranking quality."
        ),
        (
            "- Average precision summarizes "
            "precision-recall performance."
        ),
        (
            "- Lower Brier scores indicate better "
            "probability accuracy."
        ),
        "",
        "7. LIMITATIONS",
        "- Test metrics are snapshot-based estimates.",
        "- No threshold tuning was performed on test data.",
        "- Feature importance is not causal evidence.",
        "- Correlated features can distort importance.",
        "- Test results must not drive model selection.",
        "- Business deployment needs external validation.",
    ]

    path = (
        REPORTS_DIR / "model_evaluation_report.txt"
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return path


def main():
    print("=" * 65)
    print("CUSTOMER CHURN - MODEL EVALUATION")
    print("=" * 65)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    x_path = PROCESSED_DATA_DIR / "X_test.csv"
    y_path = PROCESSED_DATA_DIR / "y_test.csv"
    model_path = MODELS_DIR / "best_model.joblib"

    hashes_before = {
        "X_test": sha256_file(x_path),
        "y_test": sha256_file(y_path),
        "model": sha256_file(model_path),
    }

    X, y = load_test_data()
    model, metadata = load_best_model(X=X)

    print(
        f"\nSelected model: {metadata['best_model']}"
    )

    print(f"Test customers: {len(X):,}")

    predictions, probabilities = predict_churn(
        model, X
    )

    metrics = calculate_metrics(
        y, predictions, probabilities
    )

    classification_text = (
        create_classification_report(
            y, predictions
        )
    )

    print("\nTEST PERFORMANCE")

    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "brier_score",
    ]:
        print(f"{key}: {metrics[key]:.4f}")

    metrics_path = (
        REPORTS_DIR / "evaluation_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            {
                "model": metadata["best_model"],
                **metrics,
            },
            indent=4,
        ),
        encoding="utf-8",
    )

    classification_path = (
        REPORTS_DIR / "classification_report.txt"
    )

    classification_path.write_text(
        classification_text,
        encoding="utf-8",
    )

    predictions_path = save_predictions(
        X, y, predictions, probabilities
    )

    print("\nGenerating evaluation charts...")

    plot_confusion_matrix(y, predictions)
    plot_roc_curve(y, probabilities)
    plot_precision_recall_curve(
        y, probabilities
    )

    print("Calculating permutation importance...")

    importance = calculate_permutation_importance(
        model,
        X,
        y,
        n_repeats=5,
    )

    importance_path = (
        REPORTS_DIR / "permutation_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    plot_feature_importance(importance)

    report_path = create_evaluation_report(
        metrics,
        metadata,
        importance,
    )

    hashes_after = {
        "X_test": sha256_file(x_path),
        "y_test": sha256_file(y_path),
        "model": sha256_file(model_path),
    }

    if hashes_before != hashes_after:
        raise RuntimeError(
            "A protected input file changed!"
        )

    print("\nCLASSIFICATION REPORT")
    print(classification_text)

    print("\nTOP FIVE FEATURES")
    print(
        importance.head(5).to_string(
            index=False
        )
    )

    print(f"\nPredictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Importance: {importance_path}")
    print(f"Report: {report_path}")
    print(f"Charts: {FIGURES_DIR}")

    print("\nTEST DATA AND MODEL PRESERVED.")
    print("MODEL EVALUATION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
