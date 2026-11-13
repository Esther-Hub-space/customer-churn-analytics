
# Deployment Guide

## Local Windows Deployment

Requirements:

- Python 3.13
- VS Code
- Git
- Project dependencies

Install:

    python -m pip install -r requirements-lock.txt

Run the data preparation and analysis scripts
before starting the dashboard.

Start:

    python -m streamlit run dashboard/app.py

Open:

    http://localhost:8501

## GitHub Repository

Create a repository named:

    customer-churn-analytics

Commit source code, documentation, configuration,
and approved aggregate reports.

Do not commit secrets or confidential customer data.

## Streamlit Community Cloud

1. Push the project to GitHub.
2. Open https://share.streamlit.io/
3. Select the repository.
4. Choose dashboard/app.py as the entry point.
5. Configure dependencies.
6. Deploy the application.

Important:

The dashboard requires existing analytics CSV and
JSON files.

The default .gitignore excludes customer-level data,
model artifacts, and certain generated reports.

A clean GitHub checkout may therefore not have
everything needed to start the dashboard.

Before public deployment, provide a separate,
approved demonstration dataset and corresponding
aggregate outputs, or implement a secure build
process that generates them.

Do not simply remove privacy exclusions to make
the dashboard work.

## Production Security

For real customer data:

- Add authentication.
- Enforce authorization.
- Disable unrestricted customer-level exports.
- Use HTTPS.
- Protect secrets.
- Restrict network access.
- Maintain audit logs.
- Validate all uploaded inputs.
- Keep dependencies patched.
- Implement data retention rules.
- Review applicable data protection requirements.

## Model Version Compatibility

Record the scikit-learn version used to train
and save the model.

Avoid loading model artifacts created with
incompatible library versions.

## Production Readiness

The project is suitable as a local portfolio
demonstration after tests pass.

It is not yet a production customer decisioning
system.

Production use requires monitoring, governance,
data protection, external validation, and
measured intervention effectiveness.
