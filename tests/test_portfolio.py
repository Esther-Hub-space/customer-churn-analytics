
from pathlib import Path

import pytest

from src.config import BASE_DIR


def test_readme_exists():
    path = BASE_DIR / "README.md"
    assert path.is_file()
    assert path.stat().st_size > 100


def test_gitignore_exists():
    assert (BASE_DIR / ".gitignore").is_file()


def test_requirements_lock_exists():
    path = BASE_DIR / "requirements-lock.txt"
    assert path.is_file()
    assert path.stat().st_size > 0


def test_pyproject_exists():
    assert (BASE_DIR / "pyproject.toml").is_file()


def test_streamlit_config_exists():
    assert (
        BASE_DIR / ".streamlit" / "config.toml"
    ).is_file()


def test_architecture_document_exists():
    assert (
        BASE_DIR / "docs" / "ARCHITECTURE.md"
    ).is_file()


def test_deployment_document_exists():
    assert (
        BASE_DIR / "docs" / "DEPLOYMENT.md"
    ).is_file()


def test_environment_example_exists():
    assert (BASE_DIR / ".env.example").is_file()


def test_dashboard_source_exists():
    assert (
        BASE_DIR / "dashboard" / "app.py"
    ).is_file()


def test_dashboard_data_access_exists():
    assert (
        BASE_DIR / "dashboard" / "data_access.py"
    ).is_file()


def test_model_training_source_exists():
    assert (
        BASE_DIR / "src" / "train_models.py"
    ).is_file()


def test_model_evaluation_source_exists():
    assert (
        BASE_DIR / "src" / "evaluate_model.py"
    ).is_file()


def test_retention_source_exists():
    assert (
        BASE_DIR / "src" / "retention_campaign.py"
    ).is_file()


def test_gitignore_excludes_environment():
    content = (
        BASE_DIR / ".gitignore"
    ).read_text(encoding="utf-8")

    assert ".venv/" in content
    assert ".env" in content


def test_gitignore_excludes_model_artifacts():
    content = (
        BASE_DIR / ".gitignore"
    ).read_text(encoding="utf-8")

    assert "models/*.joblib" in content


def test_gitignore_excludes_customer_data():
    content = (
        BASE_DIR / ".gitignore"
    ).read_text(encoding="utf-8")

    assert "data/raw/*.csv" in content
    assert "data/processed/*.csv" in content


def test_gitignore_excludes_secrets():
    content = (
        BASE_DIR / ".gitignore"
    ).read_text(encoding="utf-8")

    assert ".streamlit/secrets.toml" in content


def test_readme_documents_dashboard():
    content = (
        BASE_DIR / "README.md"
    ).read_text(encoding="utf-8")

    assert "Streamlit" in content
    assert "Customer Churn" in content


def test_architecture_documents_leakage():
    content = (
        BASE_DIR / "docs" / "ARCHITECTURE.md"
    ).read_text(encoding="utf-8")

    assert "Leakage Prevention" in content


def test_deployment_documents_security():
    content = (
        BASE_DIR / "docs" / "DEPLOYMENT.md"
    ).read_text(encoding="utf-8")

    assert "Production Security" in content
