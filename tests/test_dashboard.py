
import hashlib

import numpy as np
import pandas as pd
import pytest

from dashboard.data_access import (
    REQUIRED_FILES,
    validate_files,
    load_dashboard_data,
    calculate_executive_kpis,
    churn_by_category,
    churn_by_tenure,
    filter_customers,
    filter_risk_customers,
    simulate_campaign,
    dataframe_to_csv,
)

from src.retention_campaign import file_hashes


@pytest.fixture(scope="module")
def dashboard_data():
    return load_dashboard_data()


@pytest.fixture
def sample_customers():
    return pd.DataFrame({
        "customerID": ["A", "B", "C", "D"],
        "ChurnBinary": [0, 1, 0, 1],
        "Contract": [
            "One year",
            "Month-to-month",
            "Two year",
            "Month-to-month",
        ],
        "InternetService": [
            "DSL", "Fiber optic", "DSL", "DSL"
        ],
        "PaymentMethod": [
            "Bank transfer",
            "Electronic check",
            "Bank transfer",
            "Electronic check",
        ],
        "tenure": [12, 5, 40, 60],
        "MonthlyCharges": [50, 100, 70, 150],
    })


@pytest.fixture
def sample_risk():
    return pd.DataFrame({
        "test_row": [10, 11, 12, 13],
        "contract": [
            "One year",
            "Month-to-month",
            "One year",
            "Month-to-month",
        ],
        "tenure": [2, 5, 20, 40],
        "monthly_charges": [100, 200, 50, 150],
        "churn_probability": [
            0.10, 0.40, 0.60, 0.90
        ],
        "risk_segment": [
            "Low", "Medium", "High", "Critical"
        ],
    })


def test_required_files_exist():
    assert validate_files() is True


def test_missing_file_detected(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_files({
            "missing": tmp_path / "missing.csv"
        })


def test_dashboard_data_loaded(dashboard_data):
    assert "customers" in dashboard_data
    assert "metrics" in dashboard_data
    assert "risk" in dashboard_data


def test_real_customer_count(dashboard_data):
    assert len(
        dashboard_data["customers"]
    ) == 7043


def test_real_risk_count(dashboard_data):
    assert len(
        dashboard_data["risk"]
    ) == 1409


def test_kpi_customer_counts(sample_customers):
    kpis = calculate_executive_kpis(
        sample_customers
    )

    assert kpis["total_customers"] == 4
    assert kpis["churned_customers"] == 2
    assert kpis["retained_customers"] == 2


def test_kpi_churn_rate(sample_customers):
    kpis = calculate_executive_kpis(
        sample_customers
    )

    assert kpis["churn_rate_pct"] == 50.0


def test_kpi_monthly_exposure(sample_customers):
    kpis = calculate_executive_kpis(
        sample_customers
    )

    assert kpis["churned_monthly_charges"] == (
        250.0
    )


def test_empty_kpi_rejected():
    with pytest.raises(ValueError):
        calculate_executive_kpis(
            pd.DataFrame()
        )


def test_contract_grouping(sample_customers):
    result = churn_by_category(
        sample_customers,
        "Contract",
    )

    assert result["customers"].sum() == 4
    assert result["churned"].sum() == 2


def test_invalid_grouping(sample_customers):
    with pytest.raises(ValueError):
        churn_by_category(
            sample_customers,
            "customerID",
        )


def test_tenure_grouping(sample_customers):
    result = churn_by_tenure(
        sample_customers
    )

    assert result["customers"].sum() == 4
    assert len(result) == 4


def test_contract_filter(sample_customers):
    result = filter_customers(
        sample_customers,
        contracts=["Month-to-month"],
    )

    assert len(result) == 2


def test_internet_filter(sample_customers):
    result = filter_customers(
        sample_customers,
        internet_services=["Fiber optic"],
    )

    assert len(result) == 1


def test_risk_segment_filter(sample_risk):
    result = filter_risk_customers(
        sample_risk,
        segments=["Critical"],
    )

    assert len(result) == 1
    assert result.iloc[0]["test_row"] == 13


def test_campaign_budget_limit(sample_risk):
    result = simulate_campaign(
        sample_risk,
        budget=40,
        contact_cost=20,
        horizon_months=6,
    )

    priorities = result["priorities"]

    assert priorities["selected"].sum() == 2
    assert priorities["contact_cost"].sum() <= 40


def test_campaign_exposure(sample_risk):
    result = simulate_campaign(
        sample_risk,
        budget=40,
        contact_cost=20,
        horizon_months=6,
    )

    selected = result["priorities"].loc[
        result["priorities"]["selected"]
    ]

    assert selected[
        "planning_exposure"
    ].sum() == pytest.approx(1290)


def test_campaign_scenarios(sample_risk):
    result = simulate_campaign(
        sample_risk,
        budget=40,
        contact_cost=20,
        horizon_months=6,
        scenario_rates={"Test": 0.10},
    )

    scenario = result["scenarios"].iloc[0]

    assert scenario["potential_benefit"] == (
        pytest.approx(129)
    )

    assert scenario["net_benefit"] == (
        pytest.approx(89)
    )


def test_campaign_zero_budget(sample_risk):
    result = simulate_campaign(
        sample_risk,
        budget=0,
        contact_cost=20,
    )

    assert result["priorities"][
        "selected"
    ].sum() == 0


def test_csv_export(sample_customers):
    csv_bytes = dataframe_to_csv(
        sample_customers
    )

    assert isinstance(csv_bytes, bytes)
    assert b"customerID" in csv_bytes
    assert b"ChurnBinary" in csv_bytes


def test_protected_files_unchanged():
    before = file_hashes()

    # Dashboard reads existing data only.
    load_dashboard_data()

    after = file_hashes()

    assert before == after


def test_all_required_paths_are_read_only_sources():
    for path in REQUIRED_FILES.values():
        assert path.is_file()
        assert path.stat().st_size > 0
