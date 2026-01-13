"""Tests for ollama_client module."""

import pytest

from variance_copilot.ollama_client import extract_json


class TestExtractJson:
    def test_plain_json(self):
        text = '{"headline": "Test", "value": 123}'
        result = extract_json(text)
        assert result is not None
        assert result["headline"] == "Test"
        assert result["value"] == 123

    def test_json_in_code_block(self):
        text = '''Here is the result:
```json
{"headline": "Block Test", "items": [1, 2, 3]}
```
Done.'''
        result = extract_json(text)
        assert result is not None
        assert result["headline"] == "Block Test"
        assert result["items"] == [1, 2, 3]

    def test_json_in_code_block_no_lang(self):
        text = '''```
{"key": "value"}
```'''
        result = extract_json(text)
        assert result is not None
        assert result["key"] == "value"

    def test_embedded_json(self):
        text = 'The analysis shows {"data": "found"} in the results.'
        result = extract_json(text)
        assert result is not None
        assert result["data"] == "found"

    def test_nested_json(self):
        text = '''```json
{
    "headline": "Nested",
    "drivers": [
        {"name": "A", "delta": 100},
        {"name": "B", "delta": 200}
    ]
}
```'''
        result = extract_json(text)
        assert result is not None
        assert len(result["drivers"]) == 2

    def test_no_json(self):
        text = "This is just plain text without any JSON."
        result = extract_json(text)
        assert result is None

    def test_invalid_json(self):
        text = '{"incomplete": '
        result = extract_json(text)
        assert result is None

    def test_tricky_multiline(self):
        text = '''I'll provide the analysis:

```json
{
  "headline": "IT-Kosten Anstieg",
  "summary": [
    "ERP Migration verursacht +45.000 EUR",
    "Wartungskosten stabil"
  ],
  "drivers": [{"name": "CC200", "delta": 45000, "share": 0.95}],
  "evidence": [{"label": "Datenbasiert", "text": "ERP Projekt dokumentiert"}],
  "questions": ["War ERP geplant?"]
}
```

Let me know if you need more details.'''
        result = extract_json(text)
        assert result is not None
        assert result["headline"] == "IT-Kosten Anstieg"
        assert len(result["summary"]) == 2
