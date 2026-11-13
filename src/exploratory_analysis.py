
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import CLEAN_DATA_FILE, REPORTS_DIR

FIGURES_DIR = REPORTS_DIR / "figures"

sns.set_theme(style="whitegrid", palette="deep")


def load_clean_data(path=CLEAN_DATA_FILE):
    """Load the cleaned customer dataset."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing cleaned dataset: {path}\n"
            "Run: python -m src.data_preprocessing"
        )

    df = pd.read_csv(path)

    required = {
        "customerID", "ChurnBinary", "Contract",
        "tenure", "MonthlyCharges",
        "InternetService", "PaymentMethod"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if not df["ChurnBinary"].isin([0, 1]).all():
        raise ValueError("Invalid churn target values.")

    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customer IDs found.")

    return df


def churn_rate(df):
    """Return the overall churn percentage."""
    return float(df["ChurnBinary"].mean() * 100)


def churn_by_group(df, column):
    """Calculate churn rate and customer counts."""
    result = (
        df.groupby(column, dropna=False)
        .agg(
            customers=("ChurnBinary", "size"),
            churned=("ChurnBinary", "sum"),
            churn_rate=("ChurnBinary", "mean")
        )
        .reset_index()
    )

    result["churn_rate"] *= 100

    return result.sort_values(
        "churn_rate", ascending=False
    ).reset_index(drop=True)


def save_chart(filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / filename,
        dpi=180,
        bbox_inches="tight"
    )
    plt.close()


def plot_churn_distribution(df):
    counts = (
        df["ChurnBinary"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x=["Retained", "Churned"],
        y=counts.values,
        hue=["Retained", "Churned"],
        palette=["#247BA0", "#E76F51"],
        legend=False
    )

    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            (
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            ha="center",
            va="bottom"
        )

    plt.title("Customer Churn Distribution")
    plt.ylabel("Number of Customers")
    plt.xlabel("Customer Status")
    save_chart("churn_distribution.png")


def plot_group_churn(df, column, filename, title):
    result = churn_by_group(df, column)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=result,
        x=column,
        y="churn_rate",
        color="#247BA0"
    )

    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            (
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            ha="center",
            va="bottom"
        )

    plt.title(title)
    plt.ylabel("Churn Rate (%)")
    plt.xlabel(column)
    plt.ylim(0, max(10, result["churn_rate"].max() * 1.2))
    plt.xticks(rotation=20, ha="right")
    save_chart(filename)


def plot_tenure_churn(df):
    plot_df = df.copy()

    plot_df["TenureGroup"] = pd.cut(
        plot_df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months"
        ]
    )

    result = churn_by_group(plot_df, "TenureGroup")

    order = [
        "0-12 months",
        "13-24 months",
        "25-48 months",
        "49-72 months"
    ]

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=result,
        x="TenureGroup",
        y="churn_rate",
        order=order,
        color="#E76F51"
    )

    for bar in ax.patches:
        height = bar.get_height()
        if pd.notna(height):
            ax.annotate(
                f"{height:.1f}%",
                (
                    bar.get_x() + bar.get_width() / 2,
                    height
                ),
                ha="center",
                va="bottom"
            )

    plt.title("Customer Churn by Tenure")
    plt.xlabel("Customer Tenure")
    plt.ylabel("Churn Rate (%)")
    plt.ylim(0, 65)
    save_chart("tenure_churn.png")


def plot_monthly_charges(df):
    plt.figure(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x="ChurnBinary",
        y="MonthlyCharges",
        hue="ChurnBinary",
        palette=["#247BA0", "#E76F51"],
        legend=False
    )

    plt.xticks([0, 1], ["Retained", "Churned"])
    plt.title("Monthly Charges by Churn Status")
    plt.xlabel("Customer Status")
    plt.ylabel("Monthly Charges (Dataset Units)")
    save_chart("monthly_charges_churn.png")


def generate_summary(df):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    total = len(df)
    churned = int(df["ChurnBinary"].sum())
    retained = total - churned

    lines = [
        "CUSTOMER CHURN - EXPLORATORY ANALYSIS",
        "=" * 50,
        f"Total customers: {total:,}",
        f"Churned customers: {churned:,}",
        f"Retained customers: {retained:,}",
        f"Churn rate: {churn_rate(df):.2f}%",
        "",
        "Average monthly charges:",
        f"{df['MonthlyCharges'].mean():.2f}",
        "",
    ]

    for column in [
        "Contract", "InternetService", "PaymentMethod"
    ]:
        lines.append(f"CHURN BY {column.upper()}")
        lines.append(
            churn_by_group(df, column)
            .to_string(index=False)
        )
        lines.append("")

    path = REPORTS_DIR / "eda_summary.txt"
    path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    return path


def main():
    df = load_clean_data()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("CUSTOMER CHURN - EXPLORATORY ANALYSIS")
    print("=" * 50)
    print(f"Customers: {len(df):,}")
    print(f"Churn rate: {churn_rate(df):.2f}%")

    plot_churn_distribution(df)

    plot_group_churn(
        df, "Contract", "contract_churn.png",
        "Churn Rate by Contract Type"
    )

    plot_tenure_churn(df)
    plot_monthly_charges(df)

    plot_group_churn(
        df, "InternetService",
        "internet_service_churn.png",
        "Churn Rate by Internet Service"
    )

    plot_group_churn(
        df, "PaymentMethod",
        "payment_method_churn.png",
        "Churn Rate by Payment Method"
    )

    summary = generate_summary(df)

    print("\nSix charts generated successfully.")
    print(f"Charts folder: {FIGURES_DIR}")
    print(f"Summary report: {summary}")
    print("EDA COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
