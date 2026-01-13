"""Tests for Ollama client, especially JSON extraction."""

from __future__ import annotations

import pytest

from variance_copilot.ollama_client import extract_json, validate_comment_json


class TestExtractJson:
    """Tests for extract_json function."""

    def test_extract_from_text_with_surrounding_text(self):
        """Test extraction from 'bla bla {...json...} bla' pattern."""
        text = '''Hier ist meine Analyse:

bla bla {"headline": "Test", "summary": ["Punkt 1"]} bla bla

Das war es.'''

        result = extract_json(text)

        assert result is not None
        assert result["headline"] == "Test"
        assert result["summary"] == ["Punkt 1"]

    def test_extract_from_markdown_code_block(self):
        """Test extraction from markdown code block."""
        text = '''Here is the JSON:

```json
{
  "headline": "Kostenanstieg",
  "summary": ["Bullet 1", "Bullet 2"],
  "drivers": []
}
```

Done.'''

        result = extract_json(text)

        assert result is not None
        assert result["headline"] == "Kostenanstieg"
        assert len(result["summary"]) == 2

    def test_extract_pure_json(self):
        """Test extraction when response is pure JSON."""
        text = '{"headline": "Pure", "summary": []}'

        result = extract_json(text)

        assert result is not None
        assert result["headline"] == "Pure"

    def test_extract_nested_braces(self):
        """Test extraction with nested JSON objects."""
        text = '''Some text {"headline": "Nested", "drivers": [{"name": "A", "delta": 100}]} more text'''

        result = extract_json(text)

        assert result is not None
        assert result["headline"] == "Nested"
        assert len(result["drivers"]) == 1
        assert result["drivers"][0]["name"] == "A"

    def test_extract_with_escaped_quotes(self):
        """Test extraction with escaped quotes in strings."""
        text = r'''{"headline": "Test \"quoted\"", "summary": []}'''

        result = extract_json(text)

        assert result is not None
        assert "quoted" in result["headline"]

    def test_extract_returns_none_for_invalid_json(self):
        """Test that invalid JSON returns None."""
        text = "This is not JSON at all"

        result = extract_json(text)

        assert result is None

    def test_extract_returns_none_for_empty_string(self):
        """Test that empty string returns None."""
        result = extract_json("")
        assert result is None

        result = extract_json("   ")
        assert result is None

    def test_extract_returns_none_for_none(self):
        """Test that None input returns None."""
        result = extract_json(None)
        assert result is None

    def test_extract_with_german_umlauts(self):
        """Test extraction with German special characters."""
        text = '{"headline": "Kostenüberschreitung bei Müller GmbH", "summary": ["Größere Ausgaben"]}'

        result = extract_json(text)

        assert result is not None
        assert "überschreitung" in result["headline"]
        assert "Größere" in result["summary"][0]

    def test_extract_complex_real_world_example(self):
        """Test extraction from realistic LLM response."""
        text = '''Okay, ich analysiere die Daten für Konto 6200.

{
  "headline": "Wartungskosten +45% durch Sonderinspektion",
  "summary": [
    "[Datenbasiert] Delta von -120.000 EUR durch Einzelbuchung BK999999",
    "[Indiz] Keyword 'Sonder' deutet auf außerplanmäßige Wartung",
    "[Offen] Grund für Sonderinspektion klären"
  ],
  "drivers": [
    {"name": "Roche Diagnostics", "delta": -120000, "share": 0.85}
  ],
  "evidence": [
    {"label": "Datenbasiert", "text": "Top-Buchung zeigt Einzelposten"}
  ],
  "questions": ["War die Sonderinspektion geplant?"]
}

Das war meine Analyse.'''

        result = extract_json(text)

        assert result is not None
        assert "Wartungskosten" in result["headline"]
        assert len(result["summary"]) == 3
        assert result["drivers"][0]["delta"] == -120000
        assert len(result["questions"]) == 1


class TestValidateCommentJson:
    """Tests for validate_comment_json function."""

    def test_normalizes_missing_fields(self):
        """Test that missing fields get default values."""
        data = {"headline": "Test"}

        result = validate_comment_json(data)

        assert result["headline"] == "Test"
        assert result["summary"] == []
        assert result["drivers"] == []
        assert result["evidence"] == []
        assert result["questions"] == []

    def test_preserves_existing_fields(self):
        """Test that existing fields are preserved."""
        data = {
            "headline": "Original",
            "summary": ["A", "B"],
            "drivers": [{"name": "X"}],
            "evidence": [{"label": "Datenbasiert", "text": "Y"}],
            "questions": ["Q?"],
        }

        result = validate_comment_json(data)

        assert result == data

    def test_empty_dict_gets_defaults(self):
        """Test that empty dict gets all defaults."""
        result = validate_comment_json({})

        assert result["headline"] == "Keine Überschrift"
        assert result["summary"] == []
