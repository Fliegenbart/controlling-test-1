"""Keyword extraction from posting texts."""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple

import pandas as pd

# Minimal German stopwords
STOPWORDS_DE = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "und",
    "oder", "aber", "wenn", "weil", "dass", "als", "auch", "noch", "schon",
    "von", "vom", "zum", "zur", "mit", "bei", "nach", "vor", "fuer", "durch",
    "ist", "sind", "war", "hat", "haben", "wird", "werden", "kann", "muss",
}

# Minimal English stopwords
STOPWORDS_EN = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
}

STOPWORDS = STOPWORDS_DE | STOPWORDS_EN


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words.

    Args:
        text: Input text

    Returns:
        List of tokens (lowercase, len > 2)
    """
    if not text or not isinstance(text, str):
        return []

    text = text.lower()
    text = re.sub(r"[0-9]+", " ", text)  # Remove numbers
    tokens = re.findall(r"[a-zäöüß]+", text)
    return [t for t in tokens if len(t) > 2]


def top_keywords(
    texts: pd.Series,
    top_n: int = 10,
    exclude_stopwords: bool = True,
) -> List[Tuple[str, int]]:
    """Extract top keywords from text series.

    Args:
        texts: Series of text values
        top_n: Number of top keywords
        exclude_stopwords: Whether to filter stopwords

    Returns:
        List of (keyword, count) tuples
    """
    all_tokens: List[str] = []

    for text in texts.dropna():
        tokens = tokenize(str(text))
        all_tokens.extend(tokens)

    if exclude_stopwords:
        all_tokens = [t for t in all_tokens if t not in STOPWORDS]

    return Counter(all_tokens).most_common(top_n)


def keywords_for_account(
    df: pd.DataFrame,
    account: str,
    top_n: int = 10,
) -> List[Tuple[str, int]]:
    """Get top keywords for a specific account.

    Args:
        df: Normalized DataFrame with 'text' column
        account: Account to filter
        top_n: Number of keywords

    Returns:
        List of (keyword, count) tuples
    """
    if "text" not in df.columns:
        return []

    acc_df = df[df["account"] == str(account)]
    return top_keywords(acc_df["text"], top_n=top_n)
