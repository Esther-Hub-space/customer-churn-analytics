
# System Architecture

## Overview

Customer Churn & Retention Analytics is a modular
Python data science application.

It separates data preparation, analytics, machine
learning, business calculations, and visualization.

## Data Pipeline

Raw Customer Data
    |
    v
Data Validation
    |
    v
Cleaning and Preprocessing
    |
    +--> Exploratory Data Analysis
    |
    +--> SQLite Analytics
    |
    v
Stratified Train/Test Split
    |
    +--> Training Data
    |       |
    |       v
    |   Cross-Validation
    |       |
    |       v
    |   Model Selection
    |       |
    |       v
    |   Saved Model Pipeline
    |
    +--> Held-Out Test Data
            |
            v
        Final Evaluation
            |
            v
        Saved Predictions
            |
            v
        Retention Analytics
            |
            v
        Streamlit Dashboard

## Application Components

### src/

Contains data preparation, feature engineering,
model training, evaluation, and retention analytics.

### dashboard/

Contains the Streamlit application and read-only
data access functions.

### data/

Contains raw, cleaned, and processed datasets.

### models/

Contains trained model artifacts and metadata.

### reports/

Contains business reports, charts, and CSV outputs.

### tests/

Contains automated tests for each project phase.

## Leakage Prevention

- Train/test split is performed before fitting
  preprocessing transformations.
- Cross-validation fits preprocessing separately
  within each training fold.
- Model selection uses training cross-validation.
- The test set is reserved for final evaluation.
- Phase 9 and Phase 10 do not retrain the model.
- The dashboard reads existing outputs.

## Model Persistence

Models are saved as complete scikit-learn pipelines.

Serialized artifacts must be loaded only from
trusted sources and compatible environments.

## Retention Analytics

Retention decisions are simulated using:

- Churn probability
- Monthly charges
- Planning horizon
- Contact cost
- Campaign budget
- Hypothetical intervention effectiveness

These calculations do not establish causal uplift.

## Deployment Considerations

The current dashboard requires pre-generated
analytics files.

A deployment must either:

1. Include approved, non-sensitive demo artifacts,
   or
2. Generate them in a controlled build process.

Do not expose customer-level data publicly without
appropriate authorization and safeguards.
