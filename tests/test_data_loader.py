
import pandas as pd
import pytest

from src.data_loader import load_data


@pytest.fixture
def customer_data():
    return load_data()


def test_dataset_not_empty(customer_data):
    assert not customer_data.empty


def test_customer_count(customer_data):
    assert len(customer_data) == 7043


def test_column_count(customer_data):
    assert len(customer_data.columns) == 21


def test_required_columns(customer_data):
    required = {
        "customerID",
        "tenure",
        "Contract",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    }

    assert required.issubset(
        set(customer_data.columns)
    )


def test_unique_customer_ids(customer_data):
    assert customer_data["customerID"].is_unique


def test_churn_values(customer_data):
    values = set(
        customer_data["Churn"].dropna().unique()
    )

    assert values == {"Yes", "No"}


def test_churn_count(customer_data):
    churned = (
        customer_data["Churn"] == "Yes"
    ).sum()

    assert churned == 1869


def test_monthly_charges_numeric(customer_data):
    assert pd.api.types.is_numeric_dtype(
        customer_data["MonthlyCharges"]
    )
