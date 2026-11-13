
import pandas as pd
import pytest

from src.config import (
    RAW_DATA_FILE,
    CLEAN_DATA_FILE,
)
from src.data_loader import load_data
from src.data_preprocessing import (
    clean_data,
    file_sha256,
)


@pytest.fixture(scope="module")
def original_data():
    return load_data()


@pytest.fixture(scope="module")
def cleaned_data(original_data):
    return clean_data(original_data)


def test_cleaning_preserves_customer_count(
    cleaned_data
):
    assert len(cleaned_data) == 7043


def test_cleaned_column_count(cleaned_data):
    assert len(cleaned_data.columns) == 22


def test_no_duplicate_customer_ids(cleaned_data):
    assert cleaned_data["customerID"].is_unique


def test_total_charges_numeric(cleaned_data):
    assert pd.api.types.is_numeric_dtype(
        cleaned_data["TotalCharges"]
    )


def test_expected_missing_total_charges(
    cleaned_data
):
    assert (
        cleaned_data["TotalCharges"].isna().sum()
        == 11
    )


def test_churn_binary_values(cleaned_data):
    assert set(
        cleaned_data["ChurnBinary"].unique()
    ) == {0, 1}


def test_churn_binary_matches_labels(
    cleaned_data
):
    expected = (
        cleaned_data["Churn"] == "Yes"
    ).astype(int)

    pd.testing.assert_series_equal(
        cleaned_data["ChurnBinary"],
        expected.rename("ChurnBinary"),
        check_dtype=False,
    )


def test_no_negative_charges(cleaned_data):
    assert (
        cleaned_data["MonthlyCharges"] >= 0
    ).all()

    assert (
        cleaned_data["TotalCharges"]
        .dropna() >= 0
    ).all()


def test_no_negative_tenure(cleaned_data):
    assert (cleaned_data["tenure"] >= 0).all()


def test_original_dataframe_unchanged(
    original_data
):
    before = original_data.copy(deep=True)

    clean_data(original_data)

    pd.testing.assert_frame_equal(
        original_data,
        before
    )


def test_cleaning_is_repeatable(original_data):
    first = clean_data(original_data)
    second = clean_data(original_data)

    pd.testing.assert_frame_equal(
        first,
        second
    )


def test_raw_file_not_modified():
    before = file_sha256(RAW_DATA_FILE)

    original = load_data()
    clean_data(original)

    after = file_sha256(RAW_DATA_FILE)

    assert before == after


def test_saved_cleaned_file_exists():
    assert CLEAN_DATA_FILE.exists()


def test_saved_cleaned_file_valid():
    df = pd.read_csv(CLEAN_DATA_FILE)

    assert len(df) == 7043
    assert "ChurnBinary" in df.columns
    assert df["customerID"].is_unique
