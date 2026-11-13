
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLEAN_DATA_FILE,
    REPORTS_DIR,
)
from src.retention_analytics import (
    create_risk_segments,
    summarize_risk_segments,
    prioritize_customers,
    calculate_campaign_scenarios,
)

RETENTION_DIR = REPORTS_DIR / "retention"
FIGURES_DIR = REPORTS_DIR / "figures"

REQUIRED_FILES = {
    "customers": CLEAN_DATA_FILE,
    "comparison": REPORTS_DIR / "model_comparison.csv",
    "metrics": REPORTS_DIR / "evaluation_metrics.json",
    "importance": REPORTS_DIR / "permutation_importance.csv",
    "risk": RETENTION_DIR / "customer_risk_segments.csv",
    "risk_summary": RETENTION_DIR / "risk_summary.csv",
    "scenarios": RETENTION_DIR / "campaign_scenarios.csv",
}


def validate_files(paths=None):
    """Check that previous phase outputs exist."""

    if paths is None:
        paths = REQUIRED_FILES

    missing = [
        str(path)
        for path in paths.values()
        if not Path(path).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Required project outputs are missing:\n"
            + "\n".join(missing)
        )

    return True


def load_dashboard_data():
    """
    Load existing outputs without writing files.

    No model training or test-set evaluation occurs.
    """

    validate_files()

    data = {
        "customers": pd.read_csv(
            REQUIRED_FILES["customers"]
        ),
        "comparison": pd.read_csv(
            REQUIRED_FILES["comparison"]
        ),
        "importance": pd.read_csv(
            REQUIRED_FILES["importance"]
        ),
        "risk": pd.read_csv(
            REQUIRED_FILES["risk"]
        ),
        "risk_summary": pd.read_csv(
            REQUIRED_FILES["risk_summary"]
        ),
        "scenarios": pd.read_csv(
            REQUIRED_FILES["scenarios"]
        ),
        "metrics": json.loads(
            REQUIRED_FILES["metrics"].read_text(
                encoding="utf-8"
            )
        ),
    }

    customers = data["customers"]

    required_customer_columns = {
        "customerID",
        "ChurnBinary",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "InternetService",
        "PaymentMethod",
    }

    if not required_customer_columns.issubset(
        customers.columns
    ):
        raise ValueError(
            "Cleaned customer dataset is missing "
            "required columns."
        )

    if customers["customerID"].duplicated().any():
        raise ValueError("Duplicate customer IDs.")

    if not customers["ChurnBinary"].isin(
        [0, 1]
    ).all():
        raise ValueError("Invalid churn labels.")

    risk = data["risk"]

    required_risk_columns = {
        "test_row",
        "contract",
        "tenure",
        "monthly_charges",
        "churn_probability",
        "risk_segment",
        "planning_exposure",
    }

    if not required_risk_columns.issubset(
        risk.columns
    ):
        raise ValueError(
            "Risk dataset is missing columns."
        )

    if risk["test_row"].duplicated().any():
        raise ValueError(
            "Duplicate test row identifiers."
        )

    if not np.isfinite(
        risk["churn_probability"]
    ).all():
        raise ValueError(
            "Invalid churn probabilities."
        )

    if not risk["churn_probability"].between(
        0, 1
    ).all():
        raise ValueError(
            "Churn probabilities outside [0,1]."
        )

    if len(risk) != int(
        data["metrics"]["test_customers"]
    ):
        raise ValueError(
            "Risk dataset and evaluation count differ."
        )

    return data


def calculate_executive_kpis(customers):
    """Calculate full-dataset executive KPIs."""

    if customers.empty:
        raise ValueError("Customer data is empty.")

    total = len(customers)
    churned = int(customers["ChurnBinary"].sum())

    monthly = pd.to_numeric(
        customers["MonthlyCharges"],
        errors="raise",
    )

    churned_monthly = float(
        monthly.loc[
            customers["ChurnBinary"] == 1
        ].sum()
    )

    return {
        "total_customers": total,
        "churned_customers": churned,
        "retained_customers": total - churned,
        "churn_rate_pct": 100 * churned / total,
        "monthly_charges": float(monthly.sum()),
        "churned_monthly_charges": churned_monthly,
    }


def churn_by_category(customers, column):
    """Calculate customer counts and churn rates."""

    allowed = {
        "Contract",
        "InternetService",
        "PaymentMethod",
    }

    if column not in allowed:
        raise ValueError(
            f"Unsupported grouping column: {column}"
        )

    result = (
        customers.groupby(
            column,
            dropna=False,
        )
        .agg(
            customers=("ChurnBinary", "size"),
            churned=("ChurnBinary", "sum"),
        )
        .reset_index()
    )

    result["churn_rate_pct"] = (
        100 * result["churned"]
        / result["customers"]
    )

    return result.sort_values(
        "churn_rate_pct",
        ascending=False,
    ).reset_index(drop=True)


def churn_by_tenure(customers):
    """Group tenure into four customer segments."""

    output = customers.copy()

    output["tenure_group"] = pd.cut(
        output["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=[
            "0–12 months",
            "13–24 months",
            "25–48 months",
            "49+ months",
        ],
    )

    result = (
        output.groupby(
            "tenure_group",
            observed=False,
        )
        .agg(
            customers=("ChurnBinary", "size"),
            churned=("ChurnBinary", "sum"),
        )
        .reset_index()
    )

    result["churn_rate_pct"] = (
        100 * result["churned"]
        / result["customers"].replace(0, np.nan)
    ).fillna(0)

    result["tenure_group"] = (
        result["tenure_group"].astype(str)
    )

    return result


def filter_customers(
    customers,
    contracts=None,
    internet_services=None,
):
    """Apply executive-dashboard filters."""

    result = customers.copy()

    if contracts is not None:
        result = result.loc[
            result["Contract"].isin(contracts)
        ]

    if internet_services is not None:
        result = result.loc[
            result["InternetService"].isin(
                internet_services
            )
        ]

    return result.copy()


def filter_risk_customers(
    customers,
    segments=None,
    contracts=None,
):
    """Apply risk dashboard filters."""

    result = customers.copy()

    if segments is not None:
        result = result.loc[
            result["risk_segment"].isin(segments)
        ]

    if contracts is not None:
        result = result.loc[
            result["contract"].isin(contracts)
        ]

    return result.copy()


def simulate_campaign(
    risk_customers,
    budget=10000.0,
    contact_cost=20.0,
    horizon_months=6,
    minimum_probability=0.25,
    scenario_rates=None,
):
    """
    Recalculate exposure and campaign outcomes.

    Only existing predictions are used.
    No model or test labels are accessed.
    """

    if not isinstance(horizon_months, int):
        raise ValueError(
            "Planning horizon must be an integer."
        )

    required = {
        "contract",
        "tenure",
        "monthly_charges",
        "churn_probability",
    }

    if not required.issubset(
        risk_customers.columns
    ):
        raise ValueError(
            "Missing retention planning columns."
        )

    features = pd.DataFrame({
        "Contract": risk_customers[
            "contract"
        ].to_numpy(),
        "tenure": risk_customers[
            "tenure"
        ].to_numpy(),
        "MonthlyCharges": risk_customers[
            "monthly_charges"
        ].to_numpy(),
    })

    probabilities = risk_customers[
        "churn_probability"
    ].to_numpy()

    customers = create_risk_segments(
        features,
        probabilities,
        horizon_months=horizon_months,
    )

    # Preserve original test-row identifiers.
    customers["test_row"] = (
        risk_customers["test_row"].to_numpy()
    )

    summary = summarize_risk_segments(customers)

    priorities = prioritize_customers(
        customers,
        budget=budget,
        contact_cost=contact_cost,
        minimum_probability=minimum_probability,
    )

    scenarios = calculate_campaign_scenarios(
        priorities,
        scenario_rates=scenario_rates,
    )

    return {
        "customers": customers,
        "summary": summary,
        "priorities": priorities,
        "scenarios": scenarios,
    }


def dataframe_to_csv(dataframe):
    """Create CSV bytes without writing to disk."""

    return dataframe.to_csv(
        index=False
    ).encode("utf-8")
