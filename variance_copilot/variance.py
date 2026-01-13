"""Variance calculation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


# Default materiality thresholds
DEFAULT_MIN_ABS_DELTA = 10000
DEFAULT_MIN_PCT_DELTA = 0.10
DEFAULT_MIN_BASE = 5000
DEFAULT_MIN_SHARE_TOTAL = 0.03


@dataclass
class MaterialityConfig:
    """Configuration for materiality filtering."""
    min_abs_delta: float = DEFAULT_MIN_ABS_DELTA
    min_pct_delta: float = DEFAULT_MIN_PCT_DELTA
    min_base: float = DEFAULT_MIN_BASE
    min_share_total: float = DEFAULT_MIN_SHARE_TOTAL


def variance_by_account(prior_df: pd.DataFrame, curr_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate variance by account with materiality metrics.

    Args:
        prior_df: Prior period (normalized)
        curr_df: Current period (normalized)

    Returns:
        DataFrame with: account, account_name, prior, current, delta, delta_pct,
                       abs_delta, share_of_total_abs_delta
    """
    prior_agg = (
        prior_df.groupby("account")
        .agg(prior=("amount", "sum"), account_name=("account_name", "first"))
        .reset_index()
    )

    curr_agg = (
        curr_df.groupby("account")
        .agg(current=("amount", "sum"), account_name_curr=("account_name", "first"))
        .reset_index()
    )

    merged = pd.merge(prior_agg, curr_agg, on="account", how="outer").fillna(0)

    # Use account_name from either side
    if "account_name_curr" in merged.columns:
        merged["account_name"] = merged["account_name"].combine_first(merged["account_name_curr"])
        merged = merged.drop(columns=["account_name_curr"])

    # Calculate delta and delta_pct
    merged["delta"] = merged["current"] - merged["prior"]
    merged["delta_pct"] = merged.apply(
        lambda r: (r["delta"] / abs(r["prior"])) if r["prior"] != 0 else None,
        axis=1,
    )

    # Calculate abs_delta
    merged["abs_delta"] = merged["delta"].abs()

    # Calculate share_of_total_abs_delta
    total_abs_delta = merged["abs_delta"].sum()
    if total_abs_delta > 0:
        merged["share_of_total_abs_delta"] = merged["abs_delta"] / total_abs_delta
    else:
        merged["share_of_total_abs_delta"] = 0.0

    # Sort by abs_delta descending
    return merged.sort_values("abs_delta", ascending=False).reset_index(drop=True)


def materiality_filter(
    df: pd.DataFrame,
    config: Optional[MaterialityConfig] = None,
    min_abs_delta: Optional[float] = None,
    min_pct_delta: Optional[float] = None,
    min_base: Optional[float] = None,
    min_share_total: Optional[float] = None,
) -> pd.DataFrame:
    """Filter variances by materiality.

    Logic: material = (abs_delta >= MIN_ABS_DELTA)
                   OR (abs(prior) >= MIN_BASE AND abs(delta_pct) >= MIN_PCT_DELTA)
                   OR (share_of_total_abs_delta >= MIN_SHARE_TOTAL)

    Args:
        df: Variance DataFrame from variance_by_account
        config: MaterialityConfig object (alternative to individual params)
        min_abs_delta: Minimum absolute delta
        min_pct_delta: Minimum percentage delta (0.1 = 10%)
        min_base: Minimum base value for pct filter
        min_share_total: Minimum share of total abs delta

    Returns:
        Filtered DataFrame sorted by abs_delta descending
    """
    # Use config or individual parameters
    if config:
        _min_abs_delta = config.min_abs_delta
        _min_pct_delta = config.min_pct_delta
        _min_base = config.min_base
        _min_share_total = config.min_share_total
    else:
        _min_abs_delta = min_abs_delta if min_abs_delta is not None else DEFAULT_MIN_ABS_DELTA
        _min_pct_delta = min_pct_delta if min_pct_delta is not None else DEFAULT_MIN_PCT_DELTA
        _min_base = min_base if min_base is not None else DEFAULT_MIN_BASE
        _min_share_total = min_share_total if min_share_total is not None else DEFAULT_MIN_SHARE_TOTAL

    def is_material(row) -> bool:
        abs_delta = row["abs_delta"]
        abs_prior = abs(row["prior"])
        delta_pct = row["delta_pct"]
        share = row["share_of_total_abs_delta"]

        # Rule 1: abs_delta >= MIN_ABS_DELTA
        if abs_delta >= _min_abs_delta:
            return True

        # Rule 2: abs(prior) >= MIN_BASE AND abs(delta_pct) >= MIN_PCT_DELTA
        if abs_prior >= _min_base and delta_pct is not None:
            if abs(delta_pct) >= _min_pct_delta:
                return True

        # Rule 3: share_of_total_abs_delta >= MIN_SHARE_TOTAL
        if share >= _min_share_total:
            return True

        return False

    mask = df.apply(is_material, axis=1)
    result = df[mask].copy()

    # Sort by abs_delta descending
    return result.sort_values("abs_delta", ascending=False).reset_index(drop=True)


def drivers_for_account(
    prior_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    account: str,
    dimension: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """Get top drivers for an account by dimension.

    Args:
        prior_df: Prior period (normalized)
        curr_df: Current period (normalized)
        account: Account to analyze
        dimension: Grouping dimension (e.g., 'cost_center', 'vendor')
        top_n: Number of top drivers

    Returns:
        DataFrame with dimension, prior, current, delta, share
    """
    if dimension not in prior_df.columns or dimension not in curr_df.columns:
        return pd.DataFrame(columns=[dimension, "prior", "current", "delta", "share"])

    prior_acc = prior_df[prior_df["account"] == str(account)]
    curr_acc = curr_df[curr_df["account"] == str(account)]

    prior_grp = prior_acc.groupby(dimension).agg(prior=("amount", "sum")).reset_index()
    curr_grp = curr_acc.groupby(dimension).agg(current=("amount", "sum")).reset_index()

    merged = pd.merge(prior_grp, curr_grp, on=dimension, how="outer").fillna(0)
    merged["delta"] = merged["current"] - merged["prior"]

    total_delta = abs(merged["delta"].sum())
    merged["share"] = merged["delta"].abs() / total_delta if total_delta > 0 else 0

    return merged.sort_values("delta", key=abs, ascending=False).head(top_n).reset_index(drop=True)


def samples_for_account(df: pd.DataFrame, account: str, top_n: int = 8) -> pd.DataFrame:
    """Get top postings by amount for an account.

    Args:
        df: Normalized DataFrame
        account: Account to filter
        top_n: Number of samples

    Returns:
        Top postings sorted by abs(amount)
    """
    acc_df = df[df["account"] == str(account)]

    cols = ["posting_date", "amount", "cost_center", "vendor", "text"]
    available = [c for c in cols if c in acc_df.columns]

    return (
        acc_df[available]
        .sort_values("amount", key=abs, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
