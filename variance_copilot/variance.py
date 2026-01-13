"""Variance calculation engine."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def variance_by_account(prior_df: pd.DataFrame, curr_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate variance by account.

    Args:
        prior_df: Prior period (normalized)
        curr_df: Current period (normalized)

    Returns:
        DataFrame with account, account_name, prior, current, delta, delta_pct
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

    merged["delta"] = merged["current"] - merged["prior"]
    merged["delta_pct"] = merged.apply(
        lambda r: (r["delta"] / abs(r["prior"])) if r["prior"] != 0 else None,
        axis=1,
    )

    return merged.sort_values("delta", key=abs, ascending=False).reset_index(drop=True)


def materiality_filter(
    df: pd.DataFrame,
    min_abs_delta: Optional[float] = None,
    min_pct_delta: Optional[float] = None,
    min_base: Optional[float] = None,
) -> pd.DataFrame:
    """Filter variances by materiality.

    Logic: min_abs_delta OR (min_pct_delta AND min_base)

    Args:
        df: Variance DataFrame from variance_by_account
        min_abs_delta: Minimum absolute delta
        min_pct_delta: Minimum percentage delta (0.1 = 10%)
        min_base: Minimum base value for pct filter

    Returns:
        Filtered DataFrame
    """
    def passes(row):
        abs_delta = abs(row["delta"])
        base = max(abs(row["prior"]), abs(row["current"]))

        # No filter = pass all
        if min_abs_delta is None and min_pct_delta is None:
            return True

        # Check abs threshold
        if min_abs_delta is not None and abs_delta >= min_abs_delta:
            return True

        # Check pct threshold (needs base)
        if min_pct_delta is not None and min_base is not None:
            if base >= min_base:
                pct = abs_delta / base if base > 0 else 0
                if pct >= min_pct_delta:
                    return True

        return False

    mask = df.apply(passes, axis=1)
    return df[mask].reset_index(drop=True)


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
