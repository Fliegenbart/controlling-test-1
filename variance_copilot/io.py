"""File I/O operations."""

from __future__ import annotations

from pathlib import Path
from io import BytesIO, StringIO
from typing import Union

import pandas as pd


def load_csv(source: Union[str, Path, BytesIO, StringIO]) -> pd.DataFrame:
    """Load CSV file with encoding detection.

    Args:
        source: File path or file-like object

    Returns:
        Raw DataFrame
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    for enc in encodings:
        try:
            if isinstance(source, (BytesIO, StringIO)):
                source.seek(0)
            return pd.read_csv(source, encoding=enc)
        except UnicodeDecodeError:
            continue

    raise ValueError("Could not decode CSV with supported encodings")
