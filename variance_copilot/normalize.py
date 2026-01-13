"""Data normalization and column mapping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class SignMode(Enum):
    AS_IS = "as_is"
    INVERT = "invert"
    ABS = "abs"


@dataclass
class ColumnMapping:
    """Maps source columns to standard names."""
    posting_date: str
    amount: str
    account: str
    account_name: Optional[str] = None
    cost_center: Optional[str] = None
    vendor: Optional[str] = None
    text: Optional[str] = None


def normalize(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    sign_mode: SignMode = SignMode.AS_IS,
) -> pd.DataFrame:
    """Normalize DataFrame with column mapping and type conversion.

    Args:
        df: Raw DataFrame
        mapping: Column name mapping
        sign_mode: How to handle amount signs

    Returns:
        Normalized DataFrame with standard columns + year/quarter
    """
    # Build rename map
    rename = {mapping.posting_date: "posting_date", mapping.amount: "amount", mapping.account: "account"}

    if mapping.account_name:
        rename[mapping.account_name] = "account_name"
    if mapping.cost_center:
        rename[mapping.cost_center] = "cost_center"
    if mapping.vendor:
        rename[mapping.vendor] = "vendor"
    if mapping.text:
        rename[mapping.text] = "text"

    result = df.rename(columns=rename).copy()

    # Parse date
    result["posting_date"] = pd.to_datetime(result["posting_date"])

    # Numeric amount
    result["amount"] = pd.to_numeric(result["amount"], errors="coerce")

    # Sign mode
    if sign_mode == SignMode.INVERT:
        result["amount"] = -result["amount"]
    elif sign_mode == SignMode.ABS:
        result["amount"] = result["amount"].abs()

    # Account as string
    result["account"] = result["account"].astype(str)

    # Add year/quarter
    result["year"] = result["posting_date"].dt.year
    result["quarter"] = result["posting_date"].dt.quarter

    return result
