"""Tests for io module."""

from pathlib import Path

import pytest

from variance_copilot.io import load_csv


SAMPLE_DIR = Path(__file__).parent.parent / "sample_data"


def test_load_csv_current():
    df = load_csv(SAMPLE_DIR / "current_quarter.csv")
    assert len(df) == 6
    assert "posting_date" in df.columns
    assert "amount" in df.columns


def test_load_csv_prior():
    df = load_csv(SAMPLE_DIR / "prior_year_same_quarter.csv")
    assert len(df) == 5
    assert "account" in df.columns
