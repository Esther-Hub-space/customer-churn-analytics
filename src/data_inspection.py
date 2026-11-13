
import pandas as pd

from src.config import REPORTS_DIR, create_directories
from src.data_loader import load_data


def inspect_data():
    create_directories()
    df = load_data()

    print("=" * 60)
    print("CUSTOMER CHURN - DATA INSPECTION")
    print("=" * 60)

    total_customers = len(df)
    total_columns = len(df.columns)

    # Identify churn status
    churn = df["Churn"].astype(str).str.strip()

    churned = int((churn == "Yes").sum())
    retained = int((churn == "No").sum())

    churn_rate = (
        churned / total_customers * 100
        if total_customers else 0
    )

    # Identify duplicate records and IDs
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(
        df["customerID"].duplicated().sum()
    )

    # Detect blank strings as well as actual missing values
    missing_counts = {}

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_object_dtype(series):
            blank_mask = (
                series.astype("string")
                .str.strip()
                .eq("")
                .fillna(False)
            )
            missing_counts[column] = int(
                (series.isna() | blank_mask).sum()
            )
        else:
            missing_counts[column] = int(
                series.isna().sum()
            )

    # Create column-level profiling report
    profile = pd.DataFrame({
        "column": df.columns,
        "data_type": [
            str(dtype) for dtype in df.dtypes
        ],
        "missing_count": [
            missing_counts[col] for col in df.columns
        ],
        "unique_values": [
            df[col].nunique(dropna=True)
            for col in df.columns
        ],
    })

    profile["missing_percent"] = (
        profile["missing_count"]
        / total_customers * 100
    ).round(2)

    # Monthly charges statistics
    monthly = pd.to_numeric(
        df["MonthlyCharges"],
        errors="coerce"
    )

    average_monthly = monthly.mean()

    print(f"\nTotal customers: {total_customers:,}")
    print(f"Total columns: {total_columns}")
    print(f"Churned customers: {churned:,}")
    print(f"Retained customers: {retained:,}")
    print(f"Churn rate: {churn_rate:.2f}%")
    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Duplicate customer IDs: {duplicate_ids}")

    print("\nMissing values:")
    print(
        profile.loc[
            profile["missing_count"] > 0,
            ["column", "missing_count"]
        ].to_string(index=False)
    )

    print("\nContract distribution:")
    print(df["Contract"].value_counts().to_string())

    print("\nMonthly charges:")
    print(f"Average: {average_monthly:.2f}")
    print(f"Minimum: {monthly.min():.2f}")
    print(f"Maximum: {monthly.max():.2f}")

    # Save profiling report
    profile_path = REPORTS_DIR / "dataset_profile.csv"
    profile.to_csv(profile_path, index=False)

    # Save summary report
    summary_path = REPORTS_DIR / "dataset_summary.txt"

    summary = [
        "CUSTOMER CHURN DATASET SUMMARY",
        "=" * 40,
        f"Total customers: {total_customers:,}",
        f"Total columns: {total_columns}",
        f"Churned customers: {churned:,}",
        f"Retained customers: {retained:,}",
        f"Churn rate: {churn_rate:.2f}%",
        f"Duplicate rows: {duplicate_rows}",
        f"Duplicate customer IDs: {duplicate_ids}",
        f"Average monthly charges: {average_monthly:.2f}",
        "",
        "Missing values:",
    ]

    for column, count in missing_counts.items():
        if count > 0:
            summary.append(f"{column}: {count}")

    summary_path.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8"
    )

    print("\nReports generated successfully!")
    print(f"Profile: {profile_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    inspect_data()
