
from pathlib import Path
import hashlib

import pandas as pd

from src.config import (
    RAW_DATA_FILE,
    CLEAN_DATA_FILE,
    REPORTS_DIR,
    create_directories,
)
from src.data_loader import load_data


def file_sha256(path):
    """Calculate a file hash to verify it is unchanged."""
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def clean_data(df):
    """
    Clean a copy of the customer churn dataset.

    The original DataFrame is not modified.
    """
    cleaned = df.copy(deep=True)

    # 1. Remove accidental whitespace from text
    text_columns = cleaned.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        cleaned[column] = (
            cleaned[column]
            .astype("string")
            .str.strip()
        )

    # 2. Replace empty strings with missing values
    for column in text_columns:
        cleaned[column] = (
            cleaned[column].replace("", pd.NA)
        )

    # 3. Validate customer identifiers
    if cleaned["customerID"].isna().any():
        raise ValueError(
            "Customer IDs cannot be missing."
        )

    # 4. Remove exact duplicate rows
    cleaned = cleaned.drop_duplicates().copy()

    # 5. Reject conflicting duplicate customer IDs
    if cleaned["customerID"].duplicated().any():
        raise ValueError(
            "Conflicting duplicate customer IDs found."
        )

    # 6. Convert numeric columns
    numeric_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce"
        )

    # 7. Standardize important categories
    cleaned["Churn"] = (
        cleaned["Churn"]
        .str.title()
    )

    cleaned["gender"] = (
        cleaned["gender"]
        .str.title()
    )

    # 8. Validate churn values before encoding
    valid_churn = {"Yes", "No"}

    if cleaned["Churn"].isna().any():
        raise ValueError(
            "Missing churn labels found."
        )

    if not set(cleaned["Churn"]).issubset(
        valid_churn
    ):
        raise ValueError(
            "Unexpected churn labels found."
        )

    # 9. Encode churn target
    cleaned["ChurnBinary"] = (
        cleaned["Churn"]
        .map({"Yes": 1, "No": 0})
        .astype("int64")
    )

    # 10. Validate business rules
    if cleaned["tenure"].isna().any():
        raise ValueError(
            "Missing or invalid tenure values."
        )

    if (cleaned["tenure"] < 0).any():
        raise ValueError(
            "Negative tenure values found."
        )

    if cleaned["MonthlyCharges"].isna().any():
        raise ValueError(
            "Missing or invalid monthly charges."
        )

    if (cleaned["MonthlyCharges"] < 0).any():
        raise ValueError(
            "Negative monthly charges found."
        )

    if (
        cleaned["TotalCharges"].dropna() < 0
    ).any():
        raise ValueError(
            "Negative total charges found."
        )

    # 11. Reset row index
    cleaned = cleaned.reset_index(drop=True)

    return cleaned


def generate_cleaning_report(
    original,
    cleaned,
    raw_hash,
):
    """Create a human-readable cleaning report."""

    original_rows = len(original)
    cleaned_rows = len(cleaned)

    missing_total = int(
        cleaned["TotalCharges"].isna().sum()
    )

    churned = int(
        cleaned["ChurnBinary"].sum()
    )

    churn_rate = (
        churned / cleaned_rows * 100
        if cleaned_rows else 0
    )

    lines = [
        "CUSTOMER CHURN - DATA CLEANING REPORT",
        "=" * 50,
        f"Original rows: {original_rows:,}",
        f"Cleaned rows: {cleaned_rows:,}",
        f"Rows removed: {original_rows - cleaned_rows}",
        f"Cleaned columns: {len(cleaned.columns)}",
        f"Missing TotalCharges: {missing_total}",
        f"Churned customers: {churned:,}",
        f"Churn rate: {churn_rate:.2f}%",
        "",
        "Cleaning actions:",
        "- Trimmed whitespace from text columns",
        "- Converted blank strings to missing values",
        "- Removed exact duplicate records",
        "- Checked for duplicate customer IDs",
        "- Converted numeric columns",
        "- Standardized churn and gender values",
        "- Created ChurnBinary target",
        "- Validated tenure and charge values",
        "- Preserved missing TotalCharges values",
        "",
        "Raw dataset SHA-256:",
        raw_hash,
    ]

    report_path = (
        REPORTS_DIR / "cleaning_report.txt"
    )

    report_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    return report_path


def main():
    print("=" * 60)
    print("CUSTOMER CHURN - DATA PREPROCESSING")
    print("=" * 60)

    create_directories()

    # Verify raw file before processing
    hash_before = file_sha256(RAW_DATA_FILE)

    original = load_data()
    cleaned = clean_data(original)

    # Save cleaned data separately
    cleaned.to_csv(
        CLEAN_DATA_FILE,
        index=False
    )

    # Verify original file remains unchanged
    hash_after = file_sha256(RAW_DATA_FILE)

    if hash_before != hash_after:
        raise RuntimeError(
            "Raw dataset was unexpectedly modified!"
        )

    report_path = generate_cleaning_report(
        original,
        cleaned,
        hash_after,
    )

    print(f"\nOriginal records: {len(original):,}")
    print(f"Cleaned records: {len(cleaned):,}")
    print(f"Cleaned columns: {len(cleaned.columns)}")

    print(
        "Missing TotalCharges:",
        cleaned["TotalCharges"].isna().sum()
    )

    print(
        "Churned customers:",
        cleaned["ChurnBinary"].sum()
    )

    print("\nCleaned dataset:")
    print(CLEAN_DATA_FILE)

    print("\nCleaning report:")
    print(report_path)

    print("\nRaw dataset preserved successfully!")
    print("DATA PREPROCESSING COMPLETED!")


if __name__ == "__main__":
    main()
