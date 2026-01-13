"""Tests for keywords module."""

import pandas as pd
import pytest

from variance_copilot.keywords import keywords_for_account, tokenize, top_keywords


def test_tokenize_basic():
    tokens = tokenize("Gehalt Januar Produktion")
    assert "gehalt" in tokens
    assert "januar" in tokens


def test_tokenize_removes_numbers():
    tokens = tokenize("Rechnung 12345 vom 01.01.2025")
    assert "12345" not in tokens
    assert "01" not in tokens


def test_tokenize_short_words():
    tokens = tokenize("a ab abc abcd")
    assert "a" not in tokens
    assert "ab" not in tokens
    assert "abc" in tokens


def test_top_keywords_basic():
    texts = pd.Series(["Gehalt Januar", "Gehalt Februar", "Miete Januar"])
    result = top_keywords(texts, top_n=3)

    keywords = [k for k, _ in result]
    assert "gehalt" in keywords
    assert "januar" in keywords


def test_top_keywords_excludes_stopwords():
    texts = pd.Series(["der die das Gehalt", "und oder aber Miete"])
    result = top_keywords(texts, top_n=5, exclude_stopwords=True)

    keywords = [k for k, _ in result]
    assert "der" not in keywords
    assert "gehalt" in keywords


def test_keywords_for_account():
    df = pd.DataFrame({
        "account": ["1000", "1000", "2000"],
        "text": ["Gehalt Januar", "Gehalt Februar", "Miete März"],
    })
    result = keywords_for_account(df, "1000", top_n=5)

    keywords = [k for k, _ in result]
    assert "gehalt" in keywords
    assert "miete" not in keywords  # Account 2000
