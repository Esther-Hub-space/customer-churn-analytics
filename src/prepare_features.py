
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_FILE,
    CLEAN_DATA_FILE,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
)

from src.feature_engineering import (
    prepare_training_data,
    TARGET_COLUMN,
)


def file_hash(path):
    """Calculate SHA-256 for file integrity."""
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_dataset():
    if not CLEAN_DATA_FILE.exists():
        raise FileNotFoundError(
            "Cleaned dataset missing. "
            "Run: python -m src.data_preprocessing"
        )

    df = pd.read_csv(CLEAN_DATA_FILE)

    if df.empty:
        raise ValueError("Cleaned dataset is empty.")

    return df


def save_training_files(data):
    PROCESSED_DATA_DIR.mkdir(
        parents=True, exist_ok=True
    )
    MODELS_DIR.mkdir(
        parents=True, exist_ok=True
    )

    for name in ["X_train", "X_test"]:
        data[name].to_csv(
            PROCESSED_DATA_DIR / f"{name}.csv",
            index=False,
        )

    for name in ["y_train", "y_test"]:
        data[name].rename(
            TARGET_COLUMN
        ).to_frame().to_csv(
            PROCESSED_DATA_DIR / f"{name}.csv",
            index=False,
        )

    preprocessor_path = (
        MODELS_DIR / "preprocessor.joblib"
    )

    joblib.dump(
        data["preprocessor"],
        preprocessor_path
    )

    return preprocessor_path


def create_report(data, raw_hash, clean_hash):
    REPORTS_DIR.mkdir(
        parents=True, exist_ok=True
    )

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    train_processed = data["X_train_processed"]
    test_processed = data["X_test_processed"]

    feature_names = (
        data["preprocessor"].get_feature_names_out()
    )

    lines = [
        "CUSTOMER CHURN - FEATURE ENGINEERING REPORT",
        "=" * 55,
        f"Training customers: {len(X_train):,}",
        f"Testing customers: {len(X_test):,}",
        f"Original predictor columns: {X_train.shape[1]}",
        f"Encoded feature columns: {train_processed.shape[1]}",
        "",
        f"Training churn rate: {y_train.mean() * 100:.2f}%",
        f"Testing churn rate: {y_test.mean() * 100:.2f}%",
        "",
        "PREPROCESSING",
        "- Numeric median imputation",
        "- Numeric standardization",
        "- Categorical most-frequent imputation",
        "- One-hot encoding",
        "- Unknown categories ignored",
        "- Preprocessor fitted on training data only",
        "",
        "DATA INTEGRITY",
        f"Raw dataset SHA-256: {raw_hash}",
        f"Cleaned dataset SHA-256: {clean_hash}",
        "",
        "ENCODED FEATURE NAMES",
        *feature_names.tolist(),
    ]

    report_path = (
        REPORTS_DIR / "feature_engineering_report.txt"
    )

    report_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    return report_path


def main():
    print("=" * 60)
    print("CUSTOMER CHURN - FEATURE ENGINEERING")
    print("=" * 60)

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_FILE}"
        )

    raw_before = file_hash(RAW_DATA_FILE)
    clean_before = file_hash(CLEAN_DATA_FILE)

    df = load_dataset()
    data = prepare_training_data(df)

    # Check transformed features are numeric and finite
    for name in [
        "X_train_processed",
        "X_test_processed",
    ]:
        array = data[name]

        if not np.isfinite(array).all():
            raise ValueError(
                f"Invalid transformed values in {name}"
            )

    preprocessor_path = save_training_files(data)

    # Verify original datasets remain unchanged
    raw_after = file_hash(RAW_DATA_FILE)
    clean_after = file_hash(CLEAN_DATA_FILE)

    if raw_before != raw_after:
        raise RuntimeError("Raw dataset was modified!")

    if clean_before != clean_after:
        raise RuntimeError(
            "Cleaned dataset was modified!"
        )

    report = create_report(
        data,
        raw_after,
        clean_after
    )

    print(f"\nTotal customers: {len(df):,}")
    print(
        f"Training customers: "
        f"{len(data['X_train']):,}"
    )
    print(
        f"Testing customers: "
        f"{len(data['X_test']):,}"
    )

    print(
        "Original predictor columns:",
        data["X_train"].shape[1]
    )

    print(
        "Encoded feature columns:",
        data["X_train_processed"].shape[1]
    )

    print(
        f"Training churn rate: "
        f"{data['y_train'].mean() * 100:.2f}%"
    )
    print(
        f"Testing churn rate: "
        f"{data['y_test'].mean() * 100:.2f}%"
    )

    print(f"\nPreprocessor: {preprocessor_path}")
    print(f"Report: {report}")

    print("\nRaw and cleaned datasets preserved.")
    print("FEATURE ENGINEERING COMPLETED!")


if __name__ == "__main__":
    main()
