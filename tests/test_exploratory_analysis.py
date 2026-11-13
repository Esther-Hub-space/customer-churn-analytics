
import pytest

from src.config import REPORTS_DIR
from src.exploratory_analysis import (
    FIGURES_DIR,
    load_clean_data,
    churn_rate,
    churn_by_group,
    main as run_eda,
)
from src.business_insights import (
    calculate_revenue_exposure,
    create_business_report,
)


@pytest.fixture(scope="module")
def df():
    return load_clean_data()


def test_dataset_size(df):
    assert len(df) == 7043


def test_churn_count(df):
    assert int(df["ChurnBinary"].sum()) == 1869


def test_churn_rate(df):
    assert churn_rate(df) == pytest.approx(
        26.54, abs=0.01
    )


def test_churn_rate_range(df):
    assert 0 <= churn_rate(df) <= 100


def test_contract_group_counts(df):
    result = churn_by_group(df, "Contract")
    assert result["customers"].sum() == len(df)


def test_segment_churn_rates(df):
    result = churn_by_group(df, "InternetService")
    assert result["churn_rate"].between(0, 100).all()


def test_revenue_exposure(df):
    exposure = calculate_revenue_exposure(df)
    assert exposure >= 0
    assert exposure <= df["MonthlyCharges"].sum()


def test_charts_generated():
    run_eda()

    expected = [
        "churn_distribution.png",
        "contract_churn.png",
        "tenure_churn.png",
        "monthly_charges_churn.png",
        "internet_service_churn.png",
        "payment_method_churn.png",
    ]

    for filename in expected:
        path = FIGURES_DIR / filename
        assert path.exists()
        assert path.stat().st_size > 0


def test_eda_summary_generated():
    assert (REPORTS_DIR / "eda_summary.txt").exists()


def test_business_report_generated(df):
    path = create_business_report(df)
    assert path.exists()
    assert "EXECUTIVE SUMMARY" in path.read_text(
        encoding="utf-8"
    )
