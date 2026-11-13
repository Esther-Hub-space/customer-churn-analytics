
import json

import numpy as np
import pandas as pd
import pytest

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
)

from src.model_evaluation import sha256_file

from src.retention_analytics import (
    assign_risk_segment,
    create_risk_segments,
    summarize_risk_segments,
    prioritize_customers,
    calculate_campaign_scenarios,
)

from src.retention_campaign import (
    RETENTION_DIR,
    FIGURES_DIR,
    load_verified_probabilities,
    file_hashes,
)


@pytest.fixture(scope="module")
def sample_customers():
    features = pd.DataFrame({
        "MonthlyCharges": [100, 200, 50, 150],
        "tenure": [2, 12, 24, 36],
        "Contract": [
            "Month-to-month",
            "Month-to-month",
            "One year",
            "Two year",
        ],
    })

    probabilities = np.array([
        0.10, 0.40, 0.60, 0.90
    ])

    return create_risk_segments(
        features,
        probabilities,
        horizon_months=6,
    )


def test_low_risk_boundary():
    assert assign_risk_segment(0.0) == "Low"
    assert assign_risk_segment(0.249) == "Low"


def test_medium_risk_boundary():
    assert assign_risk_segment(0.25) == "Medium"
    assert assign_risk_segment(0.499) == "Medium"


def test_high_risk_boundary():
    assert assign_risk_segment(0.50) == "High"
    assert assign_risk_segment(0.749) == "High"


def test_critical_risk_boundary():
    assert assign_risk_segment(0.75) == "Critical"
    assert assign_risk_segment(1.0) == "Critical"


def test_invalid_probability():
    with pytest.raises(ValueError):
        assign_risk_segment(1.5)


def test_invalid_horizon():
    features = pd.DataFrame({
        "MonthlyCharges": [100],
        "tenure": [2],
        "Contract": ["Month-to-month"],
    })

    with pytest.raises(ValueError):
        create_risk_segments(
            features,
            [0.5],
            horizon_months=0,
        )


def test_risk_segment_count(sample_customers):
    assert len(sample_customers) == 4

    assert set(
        sample_customers["risk_segment"]
    ) == {
        "Low", "Medium", "High", "Critical"
    }


def test_exposure_calculation(sample_customers):
    expected = np.array([
        60.0,
        480.0,
        180.0,
        810.0,
    ])

    np.testing.assert_allclose(
        sample_customers[
            "planning_exposure"
        ].to_numpy(),
        expected,
    )


def test_risk_summary(sample_customers):
    summary = summarize_risk_segments(
        sample_customers
    )

    assert summary["customers"].sum() == 4
    assert len(summary) == 4

    assert summary[
        "customer_share_pct"
    ].sum() == pytest.approx(100.0)


def test_priority_budget_limit(sample_customers):
    priorities = prioritize_customers(
        sample_customers,
        budget=40,
        contact_cost=20,
    )

    assert priorities["selected"].sum() == 2

    assert (
        priorities["contact_cost"].sum()
        <= 40
    )


def test_priority_order(sample_customers):
    priorities = prioritize_customers(
        sample_customers,
        budget=40,
        contact_cost=20,
    )

    selected = priorities.loc[
        priorities["selected"]
    ]

    assert selected.iloc[0][
        "planning_exposure"
    ] >= selected.iloc[1][
        "planning_exposure"
    ]


def test_zero_budget(sample_customers):
    priorities = prioritize_customers(
        sample_customers,
        budget=0,
        contact_cost=20,
    )

    assert priorities["selected"].sum() == 0


def test_invalid_contact_cost(sample_customers):
    with pytest.raises(ValueError):
        prioritize_customers(
            sample_customers,
            contact_cost=0,
        )


def test_campaign_scenario_calculation(
    sample_customers
):
    priorities = prioritize_customers(
        sample_customers,
        budget=40,
        contact_cost=20,
    )

    scenarios = calculate_campaign_scenarios(
        priorities,
        scenario_rates={"Test": 0.10},
    )

    row = scenarios.iloc[0]

    # Selected exposures: 810 + 480 = 1290
    assert row["planning_exposure"] == (
        pytest.approx(1290)
    )

    assert row["potential_benefit"] == (
        pytest.approx(129)
    )

    assert row["campaign_cost"] == (
        pytest.approx(40)
    )

    assert row["net_benefit"] == (
        pytest.approx(89)
    )

    assert row["roi_pct"] == (
        pytest.approx(222.5)
    )


def test_three_scenarios(sample_customers):
    priorities = prioritize_customers(
        sample_customers,
        budget=40,
        contact_cost=20,
    )

    scenarios = calculate_campaign_scenarios(
        priorities
    )

    assert len(scenarios) == 3

    assert set(scenarios["scenario"]) == {
        "Conservative",
        "Base",
        "Optimistic",
    }


def test_verified_predictions():
    X, probabilities, metadata = (
        load_verified_probabilities()
    )

    assert len(X) == 1409
    assert len(probabilities) == 1409
    assert metadata["test_set_used"] is False


def test_retention_exports_exist():
    filenames = [
        "customer_risk_segments.csv",
        "risk_summary.csv",
        "campaign_priorities.csv",
        "campaign_scenarios.csv",
    ]

    for filename in filenames:
        path = RETENTION_DIR / filename
        assert path.exists()
        assert path.stat().st_size > 0


def test_retention_charts_exist():
    filenames = [
        "retention_risk_distribution.png",
        "retention_risk_exposure.png",
        "retention_scenario_roi.png",
        "retention_budget_allocation.png",
    ]

    for filename in filenames:
        path = FIGURES_DIR / filename
        assert path.exists()
        assert path.stat().st_size > 0


def test_business_report_exists():
    path = (
        RETENTION_DIR
        / "retention_strategy_report.txt"
    )

    assert path.exists()

    content = path.read_text(
        encoding="utf-8"
    )

    assert "CAMPAIGN SCENARIOS" in content
    assert "Model retrained: No" in content


def test_model_integrity():
    metadata_path = (
        MODELS_DIR / "best_model_metadata.json"
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    assert metadata[
        "best_model_sha256"
    ] == sha256_file(
        MODELS_DIR / "best_model.joblib"
    )


def test_protected_file_hashes_stable():
    first = file_hashes()
    second = file_hashes()

    assert first == second
