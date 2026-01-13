"""Tests for variance module."""

from pathlib import Path

import pandas as pd
import pytest

from variance_copilot.io import load_csv
from variance_copilot.normalize import ColumnMapping, normalize
from variance_copilot.variance import (
    drivers_for_account,
    materiality_filter,
    samples_for_account,
    variance_by_account,
)


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


@pytest.fixture
def prior_df(mapping):
    raw = load_csv(SAMPLE_DIR / "prior_year_same_quarter.csv")
    return normalize(raw, mapping)


@pytest.fixture
def curr_df(mapping):
    raw = load_csv(SAMPLE_DIR / "current_quarter.csv")
    return normalize(raw, mapping)


class TestVarianceByAccount:
    def test_returns_all_accounts(self, prior_df, curr_df):
        result = variance_by_account(prior_df, curr_df)
        # 6000, 6200, 6300 in both files
        assert len(result) == 3

    def test_calculates_delta(self, prior_df, curr_df):
        result = variance_by_account(prior_df, curr_df)
        # Account 6000: prior=34900, current=38300, delta=3400
        acc_6000 = result[result["account"] == "6000"].iloc[0]
        assert acc_6000["prior"] == 34900
        assert acc_6000["current"] == 38300
        assert acc_6000["delta"] == 3400

    def test_calculates_delta_pct(self, prior_df, curr_df):
        result = variance_by_account(prior_df, curr_df)
        acc_6000 = result[result["account"] == "6000"].iloc[0]
        expected_pct = 3400 / 34900
        assert acc_6000["delta_pct"] == pytest.approx(expected_pct, rel=0.01)

    def test_sorted_by_abs_delta(self, prior_df, curr_df):
        result = variance_by_account(prior_df, curr_df)
        deltas = result["delta"].abs().tolist()
        assert deltas == sorted(deltas, reverse=True)


class TestMaterialityFilter:
    def test_no_filter_returns_all(self, prior_df, curr_df):
        var_df = variance_by_account(prior_df, curr_df)
        result = materiality_filter(var_df)
        assert len(result) == len(var_df)

    def test_abs_delta_filter(self, prior_df, curr_df):
        var_df = variance_by_account(prior_df, curr_df)
        result = materiality_filter(var_df, min_abs_delta=40000)
        # Only 6200 has delta >= 40000 (45000 ERP Migration)
        assert len(result) == 1
        assert result.iloc[0]["account"] == "6200"

    def test_pct_filter_needs_base(self, prior_df, curr_df):
        var_df = variance_by_account(prior_df, curr_df)
        # pct filter without base = no match via pct path
        result = materiality_filter(var_df, min_pct_delta=0.5, min_base=None)
        assert len(result) == 0


class TestDriversForAccount:
    def test_returns_drivers(self, prior_df, curr_df):
        result = drivers_for_account(prior_df, curr_df, "6000", "cost_center")
        assert len(result) > 0
        assert "cost_center" in result.columns
        assert "delta" in result.columns

    def test_missing_dimension_empty(self, prior_df, curr_df):
        result = drivers_for_account(prior_df, curr_df, "6000", "nonexistent")
        assert result.empty


class TestSamplesForAccount:
    def test_returns_top_samples(self, curr_df):
        result = samples_for_account(curr_df, "6200")
        assert len(result) == 2  # Only 2 rows for account 6200
        # First should be 45000 (ERP Migration)
        assert result.iloc[0]["amount"] == 45000

    def test_respects_top_n(self, curr_df):
        result = samples_for_account(curr_df, "6000", top_n=2)
        assert len(result) == 2
