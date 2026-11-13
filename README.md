
# Customer Churn & Retention Analytics

An end-to-end Python data science project for analyzing
customer churn, predicting churn risk, and supporting
data-driven customer retention decisions.

## Project Overview

This project uses the IBM Telco Customer Churn dataset
to demonstrate a complete analytics workflow:

- Data collection and validation
- Data cleaning and preprocessing
- Exploratory data analysis
- SQL-based business analytics
- Feature engineering
- Machine learning model training
- Cross-validation and model selection
- Model evaluation and explainability
- Retention campaign scenario analysis
- Interactive Streamlit dashboard

## Technology Stack

- Python 3.13.5
- pandas and NumPy
- scikit-learn
- SQLite and SQLAlchemy
- matplotlib and seaborn
- Plotly
- Streamlit
- pytest
- joblib

## Dataset

IBM Telco Customer Churn sample dataset.

Dataset characteristics:

- 7,043 customers
- 21 original columns
- 1,869 churned customers
- 5,174 retained customers
- Overall churn rate: approximately 26.54%

The dataset does not specify a monetary currency.

Monthly charges are not equivalent to profit or
verified financial losses.

## Machine Learning

Three classification models are evaluated:

1. Logistic Regression
2. Random Forest
3. HistGradientBoosting

The project uses:

- Stratified train/test splitting
- Five-fold stratified cross-validation
- Leakage-safe preprocessing pipelines
- ROC-AUC-based model selection
- Held-out test evaluation
- Permutation feature importance

The training split contains 5,634 customers.

The held-out test split contains 1,409 customers.

The best model is selected using training
cross-validation only.

## Dashboard Features

The Streamlit dashboard contains:

### Executive Overview

- Total customers
- Churn rate
- Contract analysis
- Tenure analysis
- Internet service analysis
- Payment method analysis

### Model Performance

- ROC-AUC
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curve
- Precision-recall curve
- Feature importance

### Customer Risk Segmentation

- Low risk
- Medium risk
- High risk
- Critical risk
- Planning exposure
- Interactive filters

### Retention Campaign Simulator

- Adjustable campaign budget
- Contact cost
- Planning horizon
- Minimum risk threshold
- Conservative scenario
- Base scenario
- Optimistic scenario
- Illustrative ROI calculations

### Reports

- Downloadable CSV reports
- Customer segmentation
- Model comparison
- Campaign priorities
- Scenario analysis

## Installation

Clone the repository:

    git clone https://github.com/YOUR_USERNAME/customer-churn-analytics.git

Navigate into the project:

    cd customer-churn-analytics

Create a Python virtual environment:

    py -3.13 -m venv .venv

Activate it on Windows:

    .\.venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install -r requirements-lock.txt

## Data Preparation

Download the public IBM Telco Customer Churn
dataset using the project's data download script.

Run:

    python -m src.download_data
    python -m src.data_preprocessing

Inspect the source files for any additional
data preparation requirements.

## Reproduce the Analysis

Run the phases in order:

    python -m src.exploratory_analysis
    python -m src.business_insights
    python -m src.database
    python -m src.sql_analytics
    python -m src.prepare_features
    python -m src.train_models
    python -m src.evaluate_model
    python -m src.retention_campaign

Some earlier modules may have their own command-line
entry points. Consult their source files if needed.

## Launch the Dashboard

    python -m streamlit run dashboard/app.py

Open:

    http://localhost:8501

The dashboard reads existing generated reports.
It does not train machine learning models.

## Automated Testing

Run all tests:

    python -m pytest tests/ -v

Run only dashboard tests:

    python -m pytest tests/test_dashboard.py -v

## Reproducibility

Use requirements-lock.txt for the exact package
versions recorded in the development environment.

Trained scikit-learn models should be loaded using
a compatible scikit-learn version.

Model artifacts and customer-level data are
excluded from Git by default.

## Limitations

This is a portfolio and educational project.

- Churn probabilities are statistical estimates.
- Model associations are not causal relationships.
- Campaign effectiveness rates are hypothetical.
- Scenario ROI is not an observed business return.
- No randomized retention intervention was conducted.
- The model has not been validated on a live population.
- The dataset does not specify currency.

## Security and Privacy

Do not commit:

- Customer-level confidential data
- API keys or credentials
- Environment secrets
- Private model artifacts
- Sensitive operational reports

Use appropriate authentication and access controls
before deploying with real customer information.

## Author

Data Science Portfolio Project

## License

Add a license after confirming the intended
reuse and distribution permissions.
