
import pandas as pd

from src.config import RAW_DATA_FILE

REQUIRED_COLUMNS = {
    "customerID",
    "tenure",
    "Contract",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}


def load_data(file_path=RAW_DATA_FILE):
    """Load and validate the customer churn dataset."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Run: python -m src.download_data"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("Dataset contains no records.")

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Required columns missing: {sorted(missing)}"
        )

    return df


def main():
    df = load_data()

    print("=" * 55)
    print("CUSTOMER CHURN - DATA LOADING")
    print("=" * 55)

    print(f"\nTotal customers: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    print("\nFirst five customer records:")
    print(df.head().to_string(index=False))

    print("\nColumn names:")
    for column in df.columns:
        print(f"- {column}")

    print("\nData loaded successfully!")


if __name__ == "__main__":
    main()
