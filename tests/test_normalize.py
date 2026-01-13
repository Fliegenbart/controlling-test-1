"""Tests for normalize module."""

from pathlib import Path

import pandas as pd
import pytest

from variance_copilot.io import load_csv
from variance_copilot.normalize import ColumnMapping, SignMode, normalize


SAMPLE_DIR = Path(__file__).parent.parent / "sample_data"


@pytest.fixture
def mapping():
    return ColumnMapping(
        posting_date="posting_date",
        amount="amount",
        account="account",
        account_name="account_name",
        cost_center="cost_center",
        vendor="vendor",
        text="text",
    )


def test_normalize_types(mapping):
    raw = load_csv(SAMPLE_DIR / "current_quarter.csv")
    df = normalize(raw, mapping)

    assert pd.api.types.is_datetime64_any_dtype(df["posting_date"])
    assert pd.api.types.is_numeric_dtype(df["amount"])
    assert df["account"].dtype == object  # string


def test_normalize_year_quarter(mapping):
    raw = load_csv(SAMPLE_DIR / "current_quarter.csv")
    df = normalize(raw, mapping)

    assert "year" in df.columns
    assert "quarter" in df.columns
    assert df["year"].iloc[0] == 2025
    assert df["quarter"].iloc[0] == 1


def test_normalize_sign_invert(mapping):
    raw = load_csv(SAMPLE_DIR / "current_quarter.csv")
    df = normalize(raw, mapping, sign_mode=SignMode.INVERT)

    assert df["amount"].iloc[0] < 0  # Original 12500 becomes -12500


def test_normalize_sign_abs(mapping):
    raw = pd.DataFrame({
        "posting_date": ["2025-01-01"],
        "amount": [-100],
        "account": ["1000"],
    })
    mapping_simple = ColumnMapping(
        posting_date="posting_date",
        amount="amount",
        account="account",
    )
    df = normalize(raw, mapping_simple, sign_mode=SignMode.ABS)

    assert df["amount"].iloc[0] == 100
