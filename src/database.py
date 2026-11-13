
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    create_engine,
    text,
    inspect,
)

from src.config import (
    BASE_DIR,
    CLEAN_DATA_FILE,
)

DATABASE_DIR = BASE_DIR / "data" / "database"
DATABASE_FILE = DATABASE_DIR / "churn_analytics.db"


def get_engine(database_path=DATABASE_FILE):
    """Create a SQLAlchemy SQLite connection engine."""

    database_path = Path(database_path).resolve()
    database_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    return create_engine(
        f"sqlite:///{database_path.as_posix()}",
        future=True,
    )


def load_clean_dataset(path=CLEAN_DATA_FILE):
    """Read and validate the cleaned CSV."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing cleaned dataset: {path}\n"
            "Run: python -m src.data_preprocessing"
        )

    df = pd.read_csv(path)

    required = {
        "customerID",
        "tenure",
        "Contract",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
        "ChurnBinary",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Cleaned dataset is empty.")

    if df["customerID"].isna().any():
        raise ValueError("Missing customer IDs.")

    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customer IDs.")

    if df["ChurnBinary"].isna().any():
        raise ValueError("Missing churn labels.")

    if not df["ChurnBinary"].isin([0, 1]).all():
        raise ValueError("Invalid churn labels.")

    if not (
        df["ChurnBinary"]
        == df["Churn"].map({"Yes": 1, "No": 0})
    ).all():
        raise ValueError(
            "ChurnBinary does not match Churn."
        )

    for column in [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

    return df


def create_database(
    engine=None,
    csv_path=CLEAN_DATA_FILE,
):
    """
    Import cleaned customer data into SQLite.

    Existing customers table is replaced on reruns.
    """

    if engine is None:
        engine = get_engine()

    df = load_clean_dataset(csv_path)

    # Explicit database types for important fields
    from sqlalchemy import (
        String,
        Integer,
        Float,
    )

    dtype_mapping = {
        "customerID": String(30),
        "tenure": Integer(),
        "MonthlyCharges": Float(),
        "TotalCharges": Float(),
        "Churn": String(10),
        "ChurnBinary": Integer(),
    }

    with engine.begin() as connection:
        df.to_sql(
            name="customers",
            con=connection,
            if_exists="replace",
            index=False,
            dtype=dtype_mapping,
        )

        # Customer IDs must be unique
        connection.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS
            idx_customers_customer_id
            ON customers(customerID)
        """))

        # Index frequently analyzed attributes
        indexes = {
            "idx_customers_churn": "ChurnBinary",
            "idx_customers_contract": "Contract",
            "idx_customers_tenure": "tenure",
            "idx_customers_internet": "InternetService",
            "idx_customers_payment": "PaymentMethod",
        }

        for index_name, column in indexes.items():
            connection.execute(text(
                f'CREATE INDEX IF NOT EXISTS '
                f'"{index_name}" '
                f'ON customers("{column}")'
            ))

        count = connection.execute(
            text("SELECT COUNT(*) FROM customers")
        ).scalar_one()

        if count != len(df):
            raise RuntimeError(
                "Database import count mismatch."
            )

    return len(df)


def check_database(engine=None):
    """Print basic database information."""

    if engine is None:
        engine = get_engine()

    if not inspect(engine).has_table("customers"):
        raise RuntimeError(
            "Customers table does not exist."
        )

    with engine.connect() as connection:
        result = connection.execute(text("""
            SELECT
                COUNT(*) AS total_customers,
                SUM(ChurnBinary) AS churned,
                COUNT(DISTINCT customerID)
                    AS unique_customers
            FROM customers
        """)).mappings().one()

    return dict(result)


def main():
    print("=" * 55)
    print("CUSTOMER CHURN - SQLITE DATABASE SETUP")
    print("=" * 55)

    engine = get_engine()

    count = create_database(engine)
    result = check_database(engine)

    print(f"\nDatabase: {DATABASE_FILE}")
    print("Table: customers")
    print(f"Imported records: {count:,}")
    print(
        "Unique customers:",
        f"{result['unique_customers']:,}"
    )
    print(
        "Churned customers:",
        f"{result['churned']:,}"
    )

    print("\nDATABASE CREATED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
