
import pandas as pd

from src.config import REPORTS_DIR
from src.exploratory_analysis import (
    load_clean_data,
    churn_by_group,
    churn_rate,
)


def calculate_revenue_exposure(df):
    """Monthly charges associated with churned customers."""
    churned = df.loc[df["ChurnBinary"] == 1]

    return float(churned["MonthlyCharges"].sum())


def get_high_risk_segments(df, column):
    """Return churn statistics by customer segment."""
    return churn_by_group(df, column)


def create_business_report(df):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    total = len(df)
    churned = int(df["ChurnBinary"].sum())
    rate = churn_rate(df)

    exposure = calculate_revenue_exposure(df)

    total_monthly = float(
        df["MonthlyCharges"].sum()
    )

    exposure_share = (
        exposure / total_monthly * 100
        if total_monthly else 0
    )

    contract = get_high_risk_segments(
        df, "Contract"
    )

    internet = get_high_risk_segments(
        df, "InternetService"
    )

    payment = get_high_risk_segments(
        df, "PaymentMethod"
    )

    highest_contract = contract.iloc[0]
    highest_internet = internet.iloc[0]
    highest_payment = payment.iloc[0]

    lines = [
        "CUSTOMER CHURN - BUSINESS INSIGHTS REPORT",
        "=" * 55,
        "",
        "1. EXECUTIVE SUMMARY",
        f"Total customers: {total:,}",
        f"Churned customers: {churned:,}",
        f"Overall churn rate: {rate:.2f}%",
        "",
        "2. REVENUE EXPOSURE",
        f"Total monthly charges: {total_monthly:,.2f}",
        f"Churned customers' monthly charges: {exposure:,.2f}",
        f"Exposure share: {exposure_share:.2f}%",
        "",
        "Note: This is a monthly-charge exposure proxy,",
        "not an observed revenue loss or forecast.",
        "The dataset does not specify a currency.",
        "",
        "3. HIGHEST CHURN SEGMENTS",
        (
            f"Contract: {highest_contract['Contract']} "
            f"({highest_contract['churn_rate']:.2f}%)"
        ),
        (
            f"Internet service: "
            f"{highest_internet['InternetService']} "
            f"({highest_internet['churn_rate']:.2f}%)"
        ),
        (
            f"Payment method: "
            f"{highest_payment['PaymentMethod']} "
            f"({highest_payment['churn_rate']:.2f}%)"
        ),
        "",
        "4. CONTRACT SEGMENT DETAILS",
        contract.to_string(index=False),
        "",
        "5. BUSINESS RECOMMENDATIONS",
        "- Investigate high-churn contract segments.",
        "- Review onboarding and early customer experience.",
        "- Analyze service complaints and satisfaction.",
        "- Test targeted retention offers experimentally.",
        "- Monitor churn rates by customer segment.",
        "",
        "6. ANALYTICAL LIMITATIONS",
        "- These findings are descriptive associations.",
        "- Churn is not measured as a monthly time series.",
        "- The dataset does not prove causal effects.",
        "- Retention campaign impact requires testing.",
    ]

    path = REPORTS_DIR / "business_insights.txt"

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    return path


def main():
    df = load_clean_data()

    print("=" * 55)
    print("CUSTOMER CHURN - BUSINESS INSIGHTS")
    print("=" * 55)

    print(f"\nCustomers: {len(df):,}")
    print(f"Churn rate: {churn_rate(df):.2f}%")

    exposure = calculate_revenue_exposure(df)

    print(
        "Churned customers' monthly charges:",
        f"{exposure:,.2f}"
    )

    print("\nChurn by contract:")
    print(
        get_high_risk_segments(df, "Contract")
        .to_string(index=False)
    )

    report = create_business_report(df)

    print(f"\nBusiness report: {report}")
    print("BUSINESS ANALYSIS COMPLETED!")


if __name__ == "__main__":
    main()
