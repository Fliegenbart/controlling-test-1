"""Tests for variance calculation engine."""

from __future__ import annotations

import pandas as pd
import pytest

from variance_copilot.variance import (
    MaterialityConfig,
    variance_by_account,
    materiality_filter,
)


def make_df(records: list[dict]) -> pd.DataFrame:
    """Create DataFrame from records with required columns."""
    return pd.DataFrame(records)


class TestVarianceByAccount:
    """Tests for variance_by_account function."""

    def test_prior_zero_edge_case(self):
        """Test that prior=0 results in delta_pct=None (no division by zero)."""
        prior = make_df([
            {"account": "1000", "account_name": "Test", "amount": 0.0},
        ])
        curr = make_df([
            {"account": "1000", "account_name": "Test", "amount": 5000.0},
        ])

        result = variance_by_account(prior, curr)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["account"] == "1000"
        assert row["prior"] == 0.0
        assert row["current"] == 5000.0
        assert row["delta"] == 5000.0
        assert pd.isna(row["delta_pct"])  # Should be None, not inf
        assert row["abs_delta"] == 5000.0

    def test_share_of_total_abs_delta_correct(self):
        """Test that share_of_total_abs_delta sums to 1.0."""
        prior = make_df([
            {"account": "1000", "account_name": "A", "amount": 100.0},
            {"account": "2000", "account_name": "B", "amount": 200.0},
            {"account": "3000", "account_name": "C", "amount": 300.0},
        ])
        curr = make_df([
            {"account": "1000", "account_name": "A", "amount": 150.0},  # delta=50
            {"account": "2000", "account_name": "B", "amount": 100.0},  # delta=-100
            {"account": "3000", "account_name": "C", "amount": 350.0},  # delta=50
        ])

        result = variance_by_account(prior, curr)

        # Total abs delta = 50 + 100 + 50 = 200
        total_abs = result["abs_delta"].sum()
        assert total_abs == 200.0

        # Share should sum to 1.0
        share_sum = result["share_of_total_abs_delta"].sum()
        assert abs(share_sum - 1.0) < 0.0001

        # Individual shares
        acc_1000 = result[result["account"] == "1000"].iloc[0]
        acc_2000 = result[result["account"] == "2000"].iloc[0]
        acc_3000 = result[result["account"] == "3000"].iloc[0]

        assert abs(acc_1000["share_of_total_abs_delta"] - 0.25) < 0.0001  # 50/200
        assert abs(acc_2000["share_of_total_abs_delta"] - 0.50) < 0.0001  # 100/200
        assert abs(acc_3000["share_of_total_abs_delta"] - 0.25) < 0.0001  # 50/200

    def test_sorted_by_abs_delta_descending(self):
        """Test results are sorted by abs_delta descending."""
        prior = make_df([
            {"account": "1000", "account_name": "Small", "amount": 100.0},
            {"account": "2000", "account_name": "Large", "amount": 100.0},
            {"account": "3000", "account_name": "Medium", "amount": 100.0},
        ])
        curr = make_df([
            {"account": "1000", "account_name": "Small", "amount": 110.0},   # delta=10
            {"account": "2000", "account_name": "Large", "amount": 200.0},   # delta=100
            {"account": "3000", "account_name": "Medium", "amount": 150.0},  # delta=50
        ])

        result = variance_by_account(prior, curr)

        assert result.iloc[0]["account"] == "2000"  # delta=100
        assert result.iloc[1]["account"] == "3000"  # delta=50
        assert result.iloc[2]["account"] == "1000"  # delta=10


class TestMaterialityFilter:
    """Tests for materiality_filter function."""

    def test_filter_by_abs_delta(self):
        """Test Rule 1: abs_delta >= MIN_ABS_DELTA."""
        prior = make_df([
            {"account": "1000", "account_name": "Below", "amount": 100.0},
            {"account": "2000", "account_name": "Above", "amount": 100.0},
        ])
        curr = make_df([
            {"account": "1000", "account_name": "Below", "amount": 105.0},   # delta=5
            {"account": "2000", "account_name": "Above", "amount": 200.0},   # delta=100
        ])

        variance_df = variance_by_account(prior, curr)
        filtered = materiality_filter(
            variance_df,
            min_abs_delta=50,
            min_pct_delta=1.0,  # Disable pct rule
            min_base=1000000,   # Disable base rule
            min_share_total=1.0,  # Disable share rule
        )

        assert len(filtered) == 1
        assert filtered.iloc[0]["account"] == "2000"

    def test_filter_by_pct_delta_with_base(self):
        """Test Rule 2: abs(prior) >= MIN_BASE AND abs(delta_pct) >= MIN_PCT_DELTA."""
        prior = make_df([
            {"account": "1000", "account_name": "SmallBase", "amount": 100.0},
            {"account": "2000", "account_name": "LargeBase", "amount": 10000.0},
        ])
        curr = make_df([
            {"account": "1000", "account_name": "SmallBase", "amount": 120.0},  # 20% but base too small
            {"account": "2000", "account_name": "LargeBase", "amount": 12000.0},  # 20% with large base
        ])

        variance_df = variance_by_account(prior, curr)
        filtered = materiality_filter(
            variance_df,
            min_abs_delta=1000000,  # Disable abs rule
            min_pct_delta=0.15,     # 15%
            min_base=5000,
            min_share_total=1.0,    # Disable share rule
        )

        assert len(filtered) == 1
        assert filtered.iloc[0]["account"] == "2000"

    def test_filter_by_share_of_total(self):
        """Test Rule 3: share_of_total_abs_delta >= MIN_SHARE_TOTAL."""
        prior = make_df([
            {"account": "1000", "account_name": "Tiny", "amount": 1000.0},
            {"account": "2000", "account_name": "Big", "amount": 1000.0},
        ])
        curr = make_df([
            {"account": "1000", "account_name": "Tiny", "amount": 1001.0},  # delta=1 (tiny share)
            {"account": "2000", "account_name": "Big", "amount": 2000.0},   # delta=1000 (huge share)
        ])

        variance_df = variance_by_account(prior, curr)
        # Total abs delta = 1001, so account 2000 has 1000/1001 = ~99.9% share

        filtered = materiality_filter(
            variance_df,
            min_abs_delta=1000000,  # Disable abs rule
            min_pct_delta=1.0,      # Disable pct rule
            min_base=1000000,       # Disable base rule
            min_share_total=0.50,   # 50% share required
        )

        assert len(filtered) == 1
        assert filtered.iloc[0]["account"] == "2000"

    def test_filter_or_logic(self):
        """Test that filter uses OR logic - any rule passes = material."""
        prior = make_df([
            {"account": "1000", "account_name": "AbsRule", "amount": 100.0},
            {"account": "2000", "account_name": "PctRule", "amount": 10000.0},
            {"account": "3000", "account_name": "ShareRule", "amount": 50.0},
            {"account": "4000", "account_name": "NoRule", "amount": 1000.0},
        ])
        curr = make_df([
            {"account": "1000", "account_name": "AbsRule", "amount": 20100.0},    # abs_delta=20000 (passes Rule 1)
            {"account": "2000", "account_name": "PctRule", "amount": 12000.0},    # 20% with large base (passes Rule 2)
            {"account": "3000", "account_name": "ShareRule", "amount": 5050.0},   # delta=5000 (passes Rule 3)
            {"account": "4000", "account_name": "NoRule", "amount": 1050.0},      # delta=50 (passes nothing)
        ])

        variance_df = variance_by_account(prior, curr)
        config = MaterialityConfig(
            min_abs_delta=10000,
            min_pct_delta=0.15,
            min_base=5000,
            min_share_total=0.10,  # 10% share
        )

        filtered = materiality_filter(variance_df, config=config)

        # Account 4000 should be filtered out
        accounts = filtered["account"].tolist()
        assert "1000" in accounts  # Rule 1: abs_delta >= 10000
        assert "2000" in accounts  # Rule 2: base >= 5000 AND pct >= 15%
        assert "3000" in accounts  # Rule 3: share >= 10% (5000/27050 = 18.5%)
        assert "4000" not in accounts  # Fails all rules

    def test_config_object_works(self):
        """Test that MaterialityConfig object is properly used."""
        prior = make_df([
            {"account": "1000", "account_name": "Test", "amount": 100.0},
        ])
        curr = make_df([
            {"account": "1000", "account_name": "Test", "amount": 200.0},
        ])

        variance_df = variance_by_account(prior, curr)
        config = MaterialityConfig(
            min_abs_delta=50,
            min_pct_delta=0.5,
            min_base=50,
            min_share_total=0.5,
        )

        filtered = materiality_filter(variance_df, config=config)

        assert len(filtered) == 1

    def test_zero_total_abs_delta(self):
        """Test edge case where no variance exists (total abs delta = 0)."""
        prior = make_df([
            {"account": "1000", "account_name": "Same", "amount": 100.0},
        ])
        curr = make_df([
            {"account": "1000", "account_name": "Same", "amount": 100.0},  # No change
        ])

        variance_df = variance_by_account(prior, curr)

        assert variance_df.iloc[0]["abs_delta"] == 0.0
        assert variance_df.iloc[0]["share_of_total_abs_delta"] == 0.0

        # Should not crash
        filtered = materiality_filter(variance_df)
        assert len(filtered) == 0  # Nothing is material when there's no variance
