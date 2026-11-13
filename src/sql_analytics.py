
from pathlib import Path
import re

import pandas as pd
from sqlalchemy import text

from src.config import BASE_DIR, REPORTS_DIR
from src.database import (
    get_engine,
    check_database,
    load_clean_dataset,
)

SQL_FILE = BASE_DIR / "sql" / "analytics_queries.sql"
SQL_REPORTS_DIR = REPORTS_DIR / "sql"

EXPECTED_QUERIES = [
    "customer_overview",
    "churn_by_contract",
    "churn_by_tenure",
    "churn_by_internet_service",
    "churn_by_payment_method",
    "churn_by_gender",
    "churn_by_senior_citizen",
    "revenue_exposure",
    "revenue_exposure_by_contract",
    "top_churned_customers_by_charges",
    "contract_payment_segments",
    "data_quality_checks",
]


def load_sql_queries(path=SQL_FILE):
    """Read named SQL statements from a .sql file."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"SQL file not found: {path}"
        )

    content = path.read_text(encoding="utf-8")

    pattern = re.compile(
        r"^-- name: ([a-z][a-z0-9_]*)\s*$",
        re.MULTILINE
    )

    matches = list(pattern.finditer(content))

    if not matches:
        raise ValueError("No named SQL queries found.")

    queries = {}

    for index, match in enumerate(matches):
        name = match.group(1)

        start = match.end()

        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(content)
        )

        sql = content[start:end].strip()

        if sql.endswith(";"):
            sql = sql[:-1].strip()

        if not sql:
            raise ValueError(
                f"Empty SQL query: {name}"
            )

        if name in queries:
            raise ValueError(
                f"Duplicate SQL query name: {name}"
            )

        # Only SELECT and CTE-based SELECT statements
        # are supported by this analytics runner.
        first_keyword = sql.split(None, 1)[0].upper()

        if first_keyword not in {"SELECT", "WITH"}:
            raise ValueError(
                f"Unsupported SQL statement: {name}"
            )

        queries[name] = sql

    missing = set(EXPECTED_QUERIES) - set(queries)

    if missing:
        raise ValueError(
            f"Missing SQL queries: {sorted(missing)}"
        )

    return queries


def run_query(engine, sql):
    """Execute a SQL SELECT query and return a DataFrame."""

    with engine.connect() as connection:
        return pd.read_sql_query(
            text(sql),
            connection
        )


def run_all_queries(engine=None):
    """Execute and export every analytics query."""

    if engine is None:
        engine = get_engine()

    check_database(engine)

    queries = load_sql_queries()

    SQL_REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {}

    for name, sql in queries.items():
        df = run_query(engine, sql)

        output = SQL_REPORTS_DIR / f"{name}.csv"
        df.to_csv(output, index=False)

        results[name] = df

        print(
            f"[OK] {name}: "
            f"{len(df)} result row(s)"
        )

    return results


def validate_results(results, cleaned_data=None):
    """Verify SQL totals against the cleaned CSV."""

    if cleaned_data is None:
        cleaned_data = load_clean_dataset()

    overview = results["customer_overview"].iloc[0]
    quality = results["data_quality_checks"].iloc[0]

    expected_total = len(cleaned_data)
    expected_churned = int(
        cleaned_data["ChurnBinary"].sum()
    )

    assert int(overview["total_customers"]) == (
        expected_total
    )

    assert int(overview["churned_customers"]) == (
        expected_churned
    )

    assert int(quality["unique_customers"]) == (
        expected_total
    )

    assert int(quality["invalid_churn_labels"]) == 0
    assert int(quality["invalid_monthly_charges"]) == 0

    return True


def generate_report(results):
    """Create a readable SQL analytics report."""

    overview = results["customer_overview"].iloc[0]
    revenue = results["revenue_exposure"].iloc[0]
    quality = results["data_quality_checks"].iloc[0]

    lines = [
        "CUSTOMER CHURN - SQL ANALYTICS REPORT",
        "=" * 60,
        "",
        "1. CUSTOMER OVERVIEW",
        f"Total customers: {int(overview['total_customers']):,}",
        f"Churned customers: {int(overview['churned_customers']):,}",
        f"Non-churned customers: {int(overview['retained_customers']):,}",
        f"Churn rate: {overview['churn_rate_pct']:.2f}%",
        f"Non-churned share: {overview['retained_share_pct']:.2f}%",
        "",
        "2. MONTHLY CHARGE EXPOSURE",
        (
            "Total monthly charges: "
            f"{revenue['total_monthly_charges']:,.2f}"
        ),
        (
            "Churned customers' monthly charges: "
            f"{revenue['churned_monthly_charges']:,.2f}"
        ),
        (
            "Exposure share: "
            f"{revenue['exposure_share_pct']:.2f}%"
        ),
        "",
        "These are dataset monetary units.",
        "The dataset does not specify a currency.",
        "Exposure is not verified financial loss.",
        "",
        "3. CHURN BY CONTRACT",
        results["churn_by_contract"].to_string(
            index=False
        ),
        "",
        "4. CHURN BY TENURE",
        results["churn_by_tenure"].to_string(
            index=False
        ),
        "",
        "5. CHURN BY INTERNET SERVICE",
        results["churn_by_internet_service"].to_string(
            index=False
        ),
        "",
        "6. CHURN BY PAYMENT METHOD",
        results["churn_by_payment_method"].to_string(
            index=False
        ),
        "",
        "7. REVENUE EXPOSURE BY CONTRACT",
        results["revenue_exposure_by_contract"].to_string(
            index=False
        ),
        "",
        "8. DATA QUALITY",
        quality.to_string(),
        "",
        "9. BUSINESS INTERPRETATION",
        "- Compare churn across contract segments.",
        "- Investigate early-tenure customer churn.",
        "- Review service and payment friction.",
        "- Prioritize retention experiments.",
        "",
        "10. LIMITATIONS",
        "- Segment associations are not causal effects.",
        "- This is a customer-level snapshot.",
        "- Cohort retention requires historical data.",
        "- No currency should be assumed.",
    ]

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        REPORTS_DIR / "sql_analytics_report.txt"
    )

    report_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    return report_path


def main():
    print("=" * 60)
    print("CUSTOMER CHURN - SQL ANALYTICS")
    print("=" * 60)

    engine = get_engine()

    results = run_all_queries(engine)

    validate_results(results)

    report = generate_report(results)

    overview = results["customer_overview"]

    print("\nCUSTOMER OVERVIEW")
    print(overview.to_string(index=False))

    print("\nCHURN BY CONTRACT")
    print(
        results["churn_by_contract"]
        .to_string(index=False)
    )

    print(f"\nCSV exports: {SQL_REPORTS_DIR}")
    print(f"Report: {report}")

    print("\nSQL ANALYTICS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
