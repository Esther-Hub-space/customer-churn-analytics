
import numpy as np
import pandas as pd

HORIZON_MONTHS = 6
CAMPAIGN_BUDGET = 10000.0
CONTACT_COST = 20.0

SCENARIOS = {
    "Conservative": 0.05,
    "Base": 0.15,
    "Optimistic": 0.30,
}

RISK_ORDER = [
    "Low",
    "Medium",
    "High",
    "Critical",
]


def assign_risk_segment(probability):
    """Convert churn probability to risk category."""

    probability = float(probability)

    if not np.isfinite(probability):
        raise ValueError("Probability must be finite.")

    if not 0 <= probability <= 1:
        raise ValueError("Invalid churn probability.")

    if probability < 0.25:
        return "Low"
    if probability < 0.50:
        return "Medium"
    if probability < 0.75:
        return "High"

    return "Critical"


def create_risk_segments(
    features,
    probabilities,
    horizon_months=HORIZON_MONTHS,
):
    """Create customer-level retention risk records."""

    if not isinstance(horizon_months, int):
        raise ValueError("Horizon must be an integer.")

    if horizon_months <= 0:
        raise ValueError("Horizon must be positive.")

    if len(features) != len(probabilities):
        raise ValueError("Customer counts do not match.")

    required = {
        "MonthlyCharges",
        "tenure",
        "Contract",
    }

    if not required.issubset(features.columns):
        raise ValueError("Missing customer features.")

    p = np.asarray(probabilities, dtype=float)

    if not np.isfinite(p).all():
        raise ValueError("Invalid probabilities.")

    if ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities outside [0,1].")

    monthly = pd.to_numeric(
        features["MonthlyCharges"],
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(monthly).all():
        raise ValueError("Invalid monthly charges.")

    if (monthly < 0).any():
        raise ValueError("Negative monthly charges.")

    output = pd.DataFrame({
        "test_row": np.arange(len(features)),
        "contract": features[
            "Contract"
        ].to_numpy(),
        "tenure": features[
            "tenure"
        ].to_numpy(),
        "monthly_charges": monthly,
        "churn_probability": p,
    })

    output["risk_segment"] = [
        assign_risk_segment(value)
        for value in p
    ]

    output["planning_exposure"] = (
        output["churn_probability"]
        * output["monthly_charges"]
        * horizon_months
    )

    # Priority is based on estimated exposure.
    # It does not use actual test churn labels.
    output["priority_score"] = (
        output["planning_exposure"]
    )

    return output


def summarize_risk_segments(customers):
    """Aggregate customers by churn-risk category."""

    summary = (
        customers.groupby(
            "risk_segment",
            observed=True,
        )
        .agg(
            customers=("test_row", "size"),
            mean_churn_probability=(
                "churn_probability", "mean"
            ),
            total_monthly_charges=(
                "monthly_charges", "sum"
            ),
            planning_exposure=(
                "planning_exposure", "sum"
            ),
        )
        .reindex(RISK_ORDER)
        .fillna(0)
        .reset_index()
    )

    summary["customer_share_pct"] = (
        100 * summary["customers"]
        / max(len(customers), 1)
    )

    return summary


def prioritize_customers(
    customers,
    budget=CAMPAIGN_BUDGET,
    contact_cost=CONTACT_COST,
    minimum_probability=0.25,
):
    """
    Select customers within a fixed outreach budget.

    Each contact has the same assumed cost.
    Customers are ranked by planning exposure.
    """

    if not np.isfinite(budget) or budget < 0:
        raise ValueError("Invalid campaign budget.")

    if not np.isfinite(contact_cost) or (
        contact_cost <= 0
    ):
        raise ValueError("Invalid contact cost.")

    if not 0 <= minimum_probability <= 1:
        raise ValueError("Invalid risk threshold.")

    eligible = customers.loc[
        customers["churn_probability"]
        >= minimum_probability
    ].copy()

    eligible = eligible.sort_values(
        ["priority_score", "test_row"],
        ascending=[False, True],
    ).reset_index(drop=True)

    capacity = int(budget // contact_cost)

    eligible["selected"] = False

    if capacity > 0:
        eligible.loc[
            :min(capacity, len(eligible)) - 1,
            "selected"
        ] = True

    eligible["contact_cost"] = np.where(
        eligible["selected"],
        contact_cost,
        0.0,
    )

    eligible["priority_rank"] = (
        np.arange(len(eligible)) + 1
    )

    return eligible


def calculate_campaign_scenarios(
    priorities,
    scenario_rates=None,
):
    """
    Calculate illustrative campaign outcomes.

    Assumption:
    A scenario rate represents the fraction of
    expected churn exposure that outreach prevents.

    This is not an estimated treatment effect.
    """

    if scenario_rates is None:
        scenario_rates = SCENARIOS

    selected = priorities.loc[
        priorities["selected"]
    ]

    contacts = int(len(selected))

    campaign_cost = float(
        selected["contact_cost"].sum()
    )

    exposure = float(
        selected["planning_exposure"].sum()
    )

    rows = []

    for name, effectiveness in (
        scenario_rates.items()
    ):
        if not 0 <= effectiveness <= 1:
            raise ValueError(
                "Effectiveness must be in [0,1]."
            )

        benefit = exposure * effectiveness
        net_benefit = benefit - campaign_cost

        roi_pct = (
            100 * net_benefit / campaign_cost
            if campaign_cost > 0
            else 0.0
        )

        rows.append({
            "scenario": name,
            "contacts": contacts,
            "assumed_effectiveness": effectiveness,
            "campaign_cost": campaign_cost,
            "planning_exposure": exposure,
            "potential_benefit": benefit,
            "net_benefit": net_benefit,
            "roi_pct": roi_pct,
        })

    return pd.DataFrame(rows)
