
import hashlib
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)

from src.model_evaluation import (
    load_best_model,
    load_test_data,
    predict_churn,
    sha256_file,
)

from src.retention_analytics import (
    HORIZON_MONTHS,
    CAMPAIGN_BUDGET,
    CONTACT_COST,
    create_risk_segments,
    summarize_risk_segments,
    prioritize_customers,
    calculate_campaign_scenarios,
)

RETENTION_DIR = REPORTS_DIR / "retention"
FIGURES_DIR = REPORTS_DIR / "figures"


def protected_paths():
    """Files that Phase 9 must not modify."""

    return {
        "model": MODELS_DIR / "best_model.joblib",
        "metadata": (
            MODELS_DIR / "best_model_metadata.json"
        ),
        "X_test": (
            PROCESSED_DATA_DIR / "X_test.csv"
        ),
        "y_test": (
            PROCESSED_DATA_DIR / "y_test.csv"
        ),
        "predictions": (
            REPORTS_DIR / "test_predictions.csv"
        ),
        "metrics": (
            REPORTS_DIR / "evaluation_metrics.json"
        ),
    }


def file_hashes():
    return {
        name: sha256_file(path)
        for name, path in protected_paths().items()
    }


def load_verified_probabilities():
    """
    Reuse Phase 8 predictions and verify alignment.

    Test labels are not used to prioritize customers.
    """

    X, y = load_test_data()

    model, metadata = load_best_model(X=X)

    prediction_path = (
        REPORTS_DIR / "test_predictions.csv"
    )

    metrics_path = (
        REPORTS_DIR / "evaluation_metrics.json"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)

    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    predictions = pd.read_csv(prediction_path)

    metrics = json.loads(
        metrics_path.read_text(encoding="utf-8")
    )

    if len(predictions) != len(X):
        raise ValueError(
            "Prediction row count mismatch."
        )

    expected_rows = np.arange(len(X))

    if not np.array_equal(
        predictions["test_row"].to_numpy(),
        expected_rows,
    ):
        raise ValueError(
            "Prediction row order mismatch."
        )

    if not np.array_equal(
        predictions["actual_churn"].to_numpy(),
        y.to_numpy(),
    ):
        raise ValueError(
            "Prediction labels do not align."
        )

    if metrics["model"] != metadata["best_model"]:
        raise ValueError(
            "Evaluation model mismatch."
        )

    if metrics["test_customers"] != len(X):
        raise ValueError(
            "Evaluation customer count mismatch."
        )

    predicted, probabilities = predict_churn(
        model, X
    )

    np.testing.assert_allclose(
        predictions[
            "churn_probability"
        ].to_numpy(),
        probabilities,
        rtol=1e-9,
        atol=1e-9,
    )

    np.testing.assert_array_equal(
        predictions[
            "predicted_churn"
        ].to_numpy(),
        predicted,
    )

    return X, probabilities, metadata


def save_chart(filename):
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = FIGURES_DIR / filename

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    return path


def plot_risk_distribution(summary):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        summary["risk_segment"],
        summary["customers"],
        color=[
            "#247BA0",
            "#E9C46A",
            "#F4A261",
            "#E76F51",
        ],
    )

    ax.set_title("Customers by Churn Risk")
    ax.set_xlabel("Risk segment")
    ax.set_ylabel("Number of customers")

    return save_chart(
        "retention_risk_distribution.png"
    )


def plot_risk_exposure(summary):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        summary["risk_segment"],
        summary["planning_exposure"],
        color="#247BA0",
    )

    ax.set_title(
        "Illustrative Exposure by Risk Segment"
    )

    ax.set_xlabel("Risk segment")
    ax.set_ylabel("Planning exposure (dataset units)")

    return save_chart(
        "retention_risk_exposure.png"
    )


def plot_scenario_roi(scenarios):
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [
        "#E76F51" if value < 0
        else "#247BA0"
        for value in scenarios["roi_pct"]
    ]

    ax.bar(
        scenarios["scenario"],
        scenarios["roi_pct"],
        color=colors,
    )

    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_title(
        "Illustrative Campaign ROI Scenarios"
    )

    ax.set_ylabel("Scenario ROI (%)")

    return save_chart(
        "retention_scenario_roi.png"
    )


def plot_budget_allocation(
    priorities,
    budget=CAMPAIGN_BUDGET,
):
    spent = float(
        priorities["contact_cost"].sum()
    )

    remaining = max(budget - spent, 0)

    fig, ax = plt.subplots(figsize=(7, 5))

    if budget > 0:
        ax.barh(
            ["Campaign budget"],
            [spent],
            color="#247BA0",
            label="Allocated",
        )

        ax.barh(
            ["Campaign budget"],
            [remaining],
            left=[spent],
            color="#D9DEE5",
            label="Unallocated",
        )

        ax.set_xlim(0, budget)
        ax.legend()
    else:
        ax.text(
            0.5, 0.5,
            "No campaign budget",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title("Campaign Budget Allocation")
    ax.set_xlabel("Dataset monetary units")

    return save_chart(
        "retention_budget_allocation.png"
    )


def create_business_report(
    customers,
    summary,
    priorities,
    scenarios,
    metadata,
):
    selected = priorities.loc[
        priorities["selected"]
    ]

    lines = [
        "CUSTOMER CHURN - RETENTION STRATEGY",
        "=" * 60,
        "",
        "1. MODEL AND DATA",
        f"Model: {metadata['best_model']}",
        f"Customers analyzed: {len(customers):,}",
        "Source: held-out Phase 8 predictions",
        "Model retrained: No",
        "Test labels used for targeting: No",
        "",
        "2. CAMPAIGN ASSUMPTIONS",
        f"Planning horizon: {HORIZON_MONTHS} months",
        f"Budget: {CAMPAIGN_BUDGET:,.2f}",
        f"Contact cost: {CONTACT_COST:,.2f}",
        "Currency: not specified by dataset",
        "",
        "3. RISK DISTRIBUTION",
        summary.to_string(index=False),
        "",
        "4. PRIORITIZATION",
        f"Eligible customers: {len(priorities):,}",
        f"Selected contacts: {len(selected):,}",
        (
            "Allocated budget: "
            f"{selected['contact_cost'].sum():,.2f}"
        ),
        (
            "Selected planning exposure: "
            f"{selected['planning_exposure'].sum():,.2f}"
        ),
        "",
        "5. CAMPAIGN SCENARIOS",
        scenarios.to_string(index=False),
        "",
        "6. BUSINESS RECOMMENDATIONS",
        "- Prioritize customers with material risk and value.",
        "- Test outreach on a randomized eligible sample.",
        "- Include a comparable no-offer control group.",
        "- Measure incremental retention, not just churn.",
        "- Track actual campaign costs and margins.",
        "- Obtain appropriate contact consent.",
        "",
        "7. IMPORTANT LIMITATIONS",
        "- This is a retrospective test-set demonstration.",
        "- No campaign treatment effect was estimated.",
        "- Scenario effectiveness rates are assumptions.",
        "- ROI values are illustrative, not forecasts.",
        "- Monthly charges are not profit margins.",
        "- The dataset does not specify currency.",
        "- Model scores may require calibration.",
        "- Do not tune models using these test outcomes.",
    ]

    path = (
        RETENTION_DIR / "retention_strategy_report.txt"
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return path


def main():
    print("=" * 65)
    print("CUSTOMER CHURN - RETENTION ANALYTICS")
    print("=" * 65)

    before = file_hashes()

    X, probabilities, metadata = (
        load_verified_probabilities()
    )

    customers = create_risk_segments(
        X,
        probabilities,
        horizon_months=HORIZON_MONTHS,
    )

    summary = summarize_risk_segments(customers)

    priorities = prioritize_customers(
        customers,
        budget=CAMPAIGN_BUDGET,
        contact_cost=CONTACT_COST,
    )

    scenarios = calculate_campaign_scenarios(
        priorities
    )

    RETENTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    customers.to_csv(
        RETENTION_DIR / "customer_risk_segments.csv",
        index=False,
    )

    summary.to_csv(
        RETENTION_DIR / "risk_summary.csv",
        index=False,
    )

    priorities.to_csv(
        RETENTION_DIR / "campaign_priorities.csv",
        index=False,
    )

    scenarios.to_csv(
        RETENTION_DIR / "campaign_scenarios.csv",
        index=False,
    )

    plot_risk_distribution(summary)
    plot_risk_exposure(summary)
    plot_scenario_roi(scenarios)
    plot_budget_allocation(priorities)

    report = create_business_report(
        customers,
        summary,
        priorities,
        scenarios,
        metadata,
    )

    after = file_hashes()

    if before != after:
        raise RuntimeError(
            "Protected input files were modified!"
        )

    print(f"\nCustomers: {len(customers):,}")
    print("\nRISK SUMMARY")
    print(summary.to_string(index=False))

    print("\nCAMPAIGN SCENARIOS")
    print(scenarios.to_string(index=False))

    print(f"\nReport: {report}")
    print(f"Exports: {RETENTION_DIR}")

    print("\nMODEL AND TEST DATA PRESERVED.")
    print("RETENTION ANALYTICS COMPLETED!")


if __name__ == "__main__":
    main()
