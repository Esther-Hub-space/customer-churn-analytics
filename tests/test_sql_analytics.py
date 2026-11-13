
import pandas as pd
import pytest

from sqlalchemy import inspect, text

from src.database import (
    get_engine,
    create_database,
    check_database,
    load_clean_dataset,
)

from src.sql_analytics import (
    load_sql_queries,
    run_query,
    validate_results,
    EXPECTED_QUERIES,
)


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    path = (
        tmp_path_factory.mktemp("sqlite")
        / "test_churn.db"
    )

    db_engine = get_engine(path)
    create_database(db_engine)

    yield db_engine

    db_engine.dispose()


@pytest.fixture(scope="module")
def queries():
    return load_sql_queries()


def test_clean_dataset_count():
    df = load_clean_dataset()
    assert len(df) == 7043


def test_database_table_exists(engine):
    assert inspect(engine).has_table("customers")


def test_database_customer_count(engine):
    result = check_database(engine)
    assert result["total_customers"] == 7043


def test_database_churn_count(engine):
    result = check_database(engine)
    assert result["churned"] == 1869


def test_unique_customers(engine):
    result = check_database(engine)
    assert result["unique_customers"] == 7043


def test_all_named_queries_exist(queries):
    assert set(EXPECTED_QUERIES).issubset(
        set(queries)
    )


def test_overview_churn_rate(engine, queries):
    df = run_query(
        engine,
        queries["customer_overview"]
    )

    assert df.iloc[0]["churn_rate_pct"] == (
        pytest.approx(26.54, abs=0.01)
    )


def test_overview_retention_share(engine, queries):
    df = run_query(
        engine,
        queries["customer_overview"]
    )

    assert df.iloc[0]["retained_share_pct"] == (
        pytest.approx(73.46, abs=0.01)
    )


def test_contract_segments(engine, queries):
    df = run_query(
        engine,
        queries["churn_by_contract"]
    )

    assert len(df) == 3
    assert df["total_customers"].sum() == 7043


def test_tenure_segments(engine, queries):
    df = run_query(
        engine,
        queries["churn_by_tenure"]
    )

    assert len(df) == 4
    assert df["total_customers"].sum() == 7043


def test_internet_service_segments(engine, queries):
    df = run_query(
        engine,
        queries["churn_by_internet_service"]
    )

    assert df["total_customers"].sum() == 7043


def test_payment_method_segments(engine, queries):
    df = run_query(
        engine,
        queries["churn_by_payment_method"]
    )

    assert df["total_customers"].sum() == 7043


def test_revenue_exposure(engine, queries):
    df = run_query(
        engine,
        queries["revenue_exposure"]
    )

    row = df.iloc[0]

    assert row["total_monthly_charges"] > 0
    assert row["churned_monthly_charges"] > 0

    assert (
        row["churned_monthly_charges"]
        <= row["total_monthly_charges"]
    )


def test_missing_total_charges(engine, queries):
    df = run_query(
        engine,
        queries["data_quality_checks"]
    )

    assert (
        df.iloc[0]["missing_total_charges"]
        == 11
    )


def test_no_invalid_churn_labels(engine, queries):
    df = run_query(
        engine,
        queries["data_quality_checks"]
    )

    assert (
        df.iloc[0]["invalid_churn_labels"]
        == 0
    )


def test_sql_matches_pandas(engine, queries):
    results = {}

    for name in [
        "customer_overview",
        "data_quality_checks",
    ]:
        results[name] = run_query(
            engine,
            queries[name]
        )

    assert validate_results(
        results,
        load_clean_dataset()
    )


def test_database_indexes(engine):
    indexes = inspect(engine).get_indexes(
        "customers"
    )

    names = {index["name"] for index in indexes}

    assert "idx_customers_customer_id" in names
    assert "idx_customers_churn" in names
    assert "idx_customers_contract" in names


def test_reimport_is_repeatable(engine):
    create_database(engine)

    result = check_database(engine)

    assert result["total_customers"] == 7043
    assert result["unique_customers"] == 7043
