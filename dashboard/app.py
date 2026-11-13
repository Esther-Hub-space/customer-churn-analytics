
import sys
from pathlib import Path

# Allow direct execution using:
# streamlit run dashboard/app.py
BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.data_access import (
    load_dashboard_data,
    calculate_executive_kpis,
    churn_by_category,
    churn_by_tenure,
    filter_customers,
    filter_risk_customers,
    simulate_campaign,
    dataframe_to_csv,
    FIGURES_DIR,
)

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary": "#247BA0",
    "secondary": "#E76F51",
    "dark": "#17324D",
}

RISK_COLORS = {
    "Low": "#2A9D8F",
    "Medium": "#E9C46A",
    "High": "#F4A261",
    "Critical": "#E76F51",
}

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.06);
        border: 1px solid rgba(128,128,128,0.14);
        border-radius: 12px;
        padding: 14px;
    }
    h1, h2, h3 {
        letter-spacing: -0.02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data():
    return load_dashboard_data()


def csv_download(label, dataframe, filename):
    st.download_button(
        label=label,
        data=dataframe_to_csv(dataframe),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def show_saved_image(filename, caption=None):
    path = FIGURES_DIR / filename

    if path.exists():
        st.image(
            str(path),
            caption=caption,
            use_container_width=True,
        )
    else:
        st.info(
            f"Chart not found: {filename}"
        )


def executive_page(data):
    st.title("Executive Overview")
    st.caption(
        "Customer churn and retention indicators "
        "from the complete cleaned dataset."
    )

    customers = data["customers"]

    with st.expander("Customer filters", expanded=True):
        col1, col2 = st.columns(2)

        contracts = sorted(
            customers["Contract"]
            .dropna().unique().tolist()
        )

        internet = sorted(
            customers["InternetService"]
            .dropna().unique().tolist()
        )

        with col1:
            selected_contracts = st.multiselect(
                "Contract types",
                options=contracts,
                default=contracts,
            )

        with col2:
            selected_internet = st.multiselect(
                "Internet services",
                options=internet,
                default=internet,
            )

    filtered = filter_customers(
        customers,
        contracts=selected_contracts,
        internet_services=selected_internet,
    )

    if filtered.empty:
        st.warning(
            "No customers match these filters."
        )
        return

    kpis = calculate_executive_kpis(filtered)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Customers",
        f"{kpis['total_customers']:,}",
    )

    c2.metric(
        "Churned Customers",
        f"{kpis['churned_customers']:,}",
    )

    c3.metric(
        "Churn Rate",
        f"{kpis['churn_rate_pct']:.2f}%",
    )

    c4.metric(
        "Churned Monthly Charges",
        f"{kpis['churned_monthly_charges']:,.2f}",
    )

    st.caption(
        "Monetary values use unspecified dataset units. "
        "Churned monthly charges are an exposure proxy, "
        "not verified financial losses."
    )

    left, right = st.columns(2)

    with left:
        counts = pd.DataFrame({
            "Status": ["Retained", "Churned"],
            "Customers": [
                kpis["retained_customers"],
                kpis["churned_customers"],
            ],
        })

        fig = px.pie(
            counts,
            names="Status",
            values="Customers",
            hole=0.55,
            title="Customer Churn Distribution",
            color="Status",
            color_discrete_map={
                "Retained": COLORS["primary"],
                "Churned": COLORS["secondary"],
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        contract = churn_by_category(
            filtered,
            "Contract",
        )

        fig = px.bar(
            contract,
            x="Contract",
            y="churn_rate_pct",
            color="Contract",
            text_auto=".1f",
            title="Churn Rate by Contract",
            labels={
                "churn_rate_pct": "Churn rate (%)",
            },
        )

        fig.update_layout(
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    left, right = st.columns(2)

    with left:
        tenure = churn_by_tenure(filtered)

        fig = px.bar(
            tenure,
            x="tenure_group",
            y="churn_rate_pct",
            title="Churn Rate by Tenure",
            labels={
                "tenure_group": "Tenure",
                "churn_rate_pct": "Churn rate (%)",
            },
            color_discrete_sequence=[
                COLORS["primary"]
            ],
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        internet_summary = churn_by_category(
            filtered,
            "InternetService",
        )

        fig = px.bar(
            internet_summary,
            x="InternetService",
            y="churn_rate_pct",
            color="InternetService",
            title="Churn by Internet Service",
            labels={
                "churn_rate_pct": "Churn rate (%)",
            },
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader("Payment Method Analysis")

    payment = churn_by_category(
        filtered,
        "PaymentMethod",
    )

    fig = px.bar(
        payment,
        x="PaymentMethod",
        y="churn_rate_pct",
        color="PaymentMethod",
        title="Churn Rate by Payment Method",
        labels={
            "churn_rate_pct": "Churn rate (%)",
        },
    )

    fig.update_layout(showlegend=False)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    csv_download(
        "Download Contract Analysis",
        contract,
        "contract_churn_dashboard.csv",
    )


def model_page(data):
    st.title("Model Performance")
    st.caption(
        "Held-out test evaluation from Phase 8. "
        "No retraining or model selection."
    )

    metrics = data["metrics"]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "ROC-AUC",
        f"{metrics['roc_auc']:.3f}",
    )

    c2.metric(
        "Precision",
        f"{metrics['precision']:.3f}",
    )

    c3.metric(
        "Recall",
        f"{metrics['recall']:.3f}",
    )

    c4.metric(
        "F1-score",
        f"{metrics['f1']:.3f}",
    )

    st.markdown(
        f"**Test customers:** "
        f"{metrics['test_customers']:,}  |  "
        f"**Classification threshold:** "
        f"{metrics['threshold']:.2f}"
    )

    st.subheader("Cross-Validation Comparison")

    comparison = data["comparison"]

    display_columns = [
        "model",
        "mean_roc_auc",
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "mean_accuracy",
    ]

    st.dataframe(
        comparison[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        comparison,
        x="model",
        y="mean_roc_auc",
        color="model",
        title="Training Cross-Validation ROC-AUC",
        labels={
            "mean_roc_auc": "Mean ROC-AUC",
            "model": "Model",
        },
    )

    fig.update_layout(showlegend=False)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Confusion Matrix")

    cm = [
        [
            metrics["true_negative"],
            metrics["false_positive"],
        ],
        [
            metrics["false_negative"],
            metrics["true_positive"],
        ],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[
                "Predicted Retained",
                "Predicted Churned",
            ],
            y=[
                "Actual Retained",
                "Actual Churned",
            ],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            hovertemplate=(
                "%{y}<br>%{x}<br>"
                "Customers: %{z}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Test Set Confusion Matrix",
        height=420,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    left, right = st.columns(2)

    with left:
        show_saved_image("roc_curve.png")

    with right:
        show_saved_image(
            "precision_recall_curve.png"
        )

    st.subheader("Feature Explainability")

    importance = data["importance"].head(15)

    fig = px.bar(
        importance.iloc[::-1],
        x="importance_mean",
        y="feature",
        orientation="h",
        error_x="importance_std",
        title="Permutation Feature Importance",
        labels={
            "importance_mean":
                "Mean ROC-AUC decrease",
            "feature": "Feature",
        },
        color_discrete_sequence=[
            COLORS["primary"]
        ],
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.info(
        "Permutation importance describes model "
        "reliance, not causal effects. "
        "The test results must not be used to "
        "retune or reselect the model."
    )

    csv_download(
        "Download Model Comparison",
        comparison,
        "model_comparison_dashboard.csv",
    )


def risk_page(data):
    st.title("Customer Risk Segmentation")

    st.caption(
        "Retrospective demonstration using Phase 8 "
        "test predictions. No actual churn labels "
        "are used to rank customers."
    )

    risk = data["risk"]

    left, right = st.columns(2)

    with left:
        segments = st.multiselect(
            "Risk segments",
            options=[
                "Low",
                "Medium",
                "High",
                "Critical",
            ],
            default=[
                "Low",
                "Medium",
                "High",
                "Critical",
            ],
        )

    with right:
        contracts = sorted(
            risk["contract"]
            .dropna().unique().tolist()
        )

        selected_contracts = st.multiselect(
            "Contract types",
            options=contracts,
            default=contracts,
            key="risk_contracts",
        )

    filtered = filter_risk_customers(
        risk,
        segments=segments,
        contracts=selected_contracts,
    )

    if filtered.empty:
        st.warning(
            "No customers match the selected filters."
        )
        return

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Customers in Selection",
        f"{len(filtered):,}",
    )

    c2.metric(
        "Average Churn Risk",
        f"{filtered['churn_probability'].mean()*100:.1f}%",
    )

    c3.metric(
        "Planning Exposure",
        f"{filtered['planning_exposure'].sum():,.2f}",
    )

    summary = (
        filtered.groupby("risk_segment")
        .agg(
            customers=("test_row", "size"),
            exposure=("planning_exposure", "sum"),
        )
        .reset_index()
    )

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            summary,
            x="risk_segment",
            y="customers",
            color="risk_segment",
            color_discrete_map=RISK_COLORS,
            title="Customers by Risk Segment",
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        fig = px.bar(
            summary,
            x="risk_segment",
            y="exposure",
            color="risk_segment",
            color_discrete_map=RISK_COLORS,
            title="Planning Exposure by Risk",
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader("Risk Distribution")

    fig = px.histogram(
        filtered,
        x="churn_probability",
        nbins=20,
        title="Distribution of Churn Probabilities",
        labels={
            "churn_probability":
                "Predicted churn probability",
        },
        color_discrete_sequence=[
            COLORS["primary"]
        ],
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Customer Risk Records")

    st.dataframe(
        filtered.sort_values(
            "planning_exposure",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    csv_download(
        "Download Filtered Risk Data",
        filtered,
        "filtered_customer_risk.csv",
    )


def campaign_page(data):
    st.title("Retention Campaign Simulator")

    st.warning(
        "Financial outcomes are hypothetical "
        "sensitivity scenarios, not measured "
        "campaign uplift or profit forecasts."
    )

    risk = data["risk"]

    with st.sidebar:
        st.markdown("---")
        st.subheader("Campaign Controls")

        budget = st.number_input(
            "Campaign budget",
            min_value=0.0,
            max_value=1000000.0,
            value=10000.0,
            step=1000.0,
        )

        contact_cost = st.number_input(
            "Cost per customer contact",
            min_value=0.01,
            max_value=10000.0,
            value=20.0,
            step=5.0,
        )

        horizon = st.slider(
            "Planning horizon (months)",
            min_value=1,
            max_value=24,
            value=6,
        )

        minimum_probability = st.slider(
            "Minimum churn probability",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
        )

    st.subheader("Scenario Assumptions")

    c1, c2, c3 = st.columns(3)

    with c1:
        conservative = st.slider(
            "Conservative effectiveness (%)",
            0, 100, 5,
        )

    with c2:
        base = st.slider(
            "Base effectiveness (%)",
            0, 100, 15,
        )

    with c3:
        optimistic = st.slider(
            "Optimistic effectiveness (%)",
            0, 100, 30,
        )

    scenario_rates = {
        "Conservative": conservative / 100,
        "Base": base / 100,
        "Optimistic": optimistic / 100,
    }

    result = simulate_campaign(
        risk,
        budget=float(budget),
        contact_cost=float(contact_cost),
        horizon_months=int(horizon),
        minimum_probability=float(
            minimum_probability
        ),
        scenario_rates=scenario_rates,
    )

    priorities = result["priorities"]
    scenarios = result["scenarios"]

    selected = priorities.loc[
        priorities["selected"]
    ]

    spent = float(
        selected["contact_cost"].sum()
    )

    exposure = float(
        selected["planning_exposure"].sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Eligible Customers",
        f"{len(priorities):,}",
    )

    c2.metric(
        "Selected Contacts",
        f"{len(selected):,}",
    )

    c3.metric(
        "Allocated Budget",
        f"{spent:,.2f}",
    )

    c4.metric(
        "Selected Exposure",
        f"{exposure:,.2f}",
    )

    left, right = st.columns(2)

    with left:
        allocation = pd.DataFrame({
            "Category": [
                "Allocated",
                "Unallocated",
            ],
            "Amount": [
                spent,
                max(float(budget) - spent, 0),
            ],
        })

        if budget > 0:
            fig = px.pie(
                allocation,
                names="Category",
                values="Amount",
                hole=0.5,
                title="Campaign Budget Allocation",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )
        else:
            st.info("No budget allocated.")

    with right:
        fig = px.bar(
            scenarios,
            x="scenario",
            y="roi_pct",
            color="scenario",
            title="Illustrative ROI by Scenario",
            labels={
                "roi_pct": "ROI (%)",
                "scenario": "Scenario",
            },
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader("Cost-Benefit Scenarios")

    st.dataframe(
        scenarios.round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Selected Campaign Contacts")

    st.dataframe(
        selected,
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)

    with left:
        csv_download(
            "Download Campaign Priorities",
            priorities,
            "campaign_priorities_dashboard.csv",
        )

    with right:
        csv_download(
            "Download Campaign Scenarios",
            scenarios,
            "campaign_scenarios_dashboard.csv",
        )

    st.caption(
        "Amounts are in unspecified dataset monetary "
        "units. Planning exposure is not actual loss. "
        "Effectiveness rates are assumed reductions "
        "in exposure, not estimated treatment effects."
    )


def reports_page(data):
    st.title("Reports & Downloads")

    st.write(
        "Download existing analysis outputs or "
        "review the reports generated in previous phases."
    )

    exports = [
        (
            "Cleaned Customer Dataset",
            data["customers"],
            "cleaned_customer_data.csv",
        ),
        (
            "Model Comparison",
            data["comparison"],
            "model_comparison.csv",
        ),
        (
            "Risk Summary",
            data["risk_summary"],
            "risk_summary.csv",
        ),
        (
            "Campaign Scenarios",
            data["scenarios"],
            "campaign_scenarios.csv",
        ),
        (
            "Permutation Importance",
            data["importance"],
            "permutation_importance.csv",
        ),
    ]

    for label, frame, filename in exports:
        with st.container(border=True):
            st.subheader(label)

            st.caption(
                f"{len(frame):,} rows available"
            )

            csv_download(
                f"Download {label}",
                frame,
                filename,
            )

    st.info(
        "Customer-level exports are intended for "
        "authorized analysis. Avoid publicly sharing "
        "confidential customer data or predictions."
    )


def main():
    with st.sidebar:
        st.title("Churn Analytics")
        st.caption("Customer Retention Intelligence")

        page = st.radio(
            "Navigation",
            [
                "Executive Overview",
                "Model Performance",
                "Risk Segmentation",
                "Campaign Simulator",
                "Reports & Downloads",
            ],
        )

        st.markdown("---")
        st.caption(
            "Read-only dashboard | Phase 10"
        )

    try:
        data = get_data()
    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        st.error(
            "Unable to load project analytics."
        )
        st.code(str(error))
        st.stop()

    if page == "Executive Overview":
        executive_page(data)

    elif page == "Model Performance":
        model_page(data)

    elif page == "Risk Segmentation":
        risk_page(data)

    elif page == "Campaign Simulator":
        campaign_page(data)

    else:
        reports_page(data)


if __name__ == "__main__":
    main()
