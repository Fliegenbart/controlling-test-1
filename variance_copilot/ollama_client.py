"""Ollama API client for local LLM inference."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

import requests

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TIMEOUT = 120


def get_config() -> Dict[str, Any]:
    """Get Ollama config from environment."""
    return {
        "base_url": os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
        "model": os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
        "timeout": int(os.getenv("OLLAMA_TIMEOUT", DEFAULT_TIMEOUT)),
    }


def is_available(base_url: Optional[str] = None) -> bool:
    """Check if Ollama is reachable.

    Args:
        base_url: Ollama API base URL

    Returns:
        True if reachable
    """
    url = base_url or get_config()["base_url"]
    try:
        resp = requests.get(f"{url}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def list_models(base_url: Optional[str] = None) -> list:
    """List available models.

    Args:
        base_url: Ollama API base URL

    Returns:
        List of model names
    """
    url = base_url or get_config()["base_url"]
    try:
        resp = requests.get(f"{url}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except requests.exceptions.RequestException:
        return []


def ollama_generate(
    user_prompt: str,
    system_prompt: str,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.3,
    timeout: Optional[int] = None,
) -> str:
    """Generate text with Ollama.

    Args:
        user_prompt: User prompt
        system_prompt: System prompt
        base_url: API base URL
        model: Model name
        temperature: Sampling temperature
        timeout: Request timeout

    Returns:
        Generated text

    Raises:
        ConnectionError: If Ollama not reachable
        RuntimeError: If generation fails
    """
    config = get_config()
    url = base_url or config["base_url"]
    mdl = model or config["model"]
    tout = timeout or config["timeout"]

    payload = {
        "model": mdl,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }

    try:
        resp = requests.post(f"{url}/api/generate", json=payload, timeout=tout)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Ollama nicht erreichbar unter {url}. "
            "Bitte starten: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Ollama Timeout nach {tout}s")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama API Fehler: {e}")


def _fix_json_issues(text: str) -> str:
    """Fix common JSON issues from LLMs.

    LLMs sometimes output invalid JSON like:
    - "delta": +380376  (leading + is invalid)
    - "delta": -115,209  (German thousand separator - invalid in JSON)
    - trailing commas in arrays/objects
    - single quotes instead of double quotes (careful with this)

    Args:
        text: JSON string with potential issues

    Returns:
        Fixed JSON string
    """
    # Fix German/US thousand separators in numbers (e.g., -115,209 -> -115209)
    # Pattern: numbers with comma followed by exactly 3 digits (thousand separator pattern)
    # We need to be careful not to break JSON structure
    # Match pattern: digit(s),3digits - this is a thousand separator
    # Apply multiple times to handle numbers like 1,234,567
    prev_text = None
    while prev_text != text:
        prev_text = text
        # Match: optional sign, 1-3 digits, comma, exactly 3 digits
        # The comma must be followed by exactly 3 digits (then non-digit or end)
        text = re.sub(r'([+-]?\d{1,3}),(\d{3})(?!\d)', r'\1\2', text)

    # Fix leading + on numbers (e.g., ": +123" -> ": 123")
    text = re.sub(r':\s*\+(\d)', r': \1', text)

    # Fix trailing commas before ] or }
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # Fix missing commas between array elements (common LLM mistake)
    # e.g., "item1" "item2" -> "item1", "item2"
    text = re.sub(r'"\s+(?=")', '", ', text)

    return text


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response.

    Robust extraction that handles:
    - Markdown code blocks (```json ... ```)
    - JSON embedded in text ("bla bla {...} bla")
    - Multiple JSON objects (returns first valid one)
    - Nested braces
    - Invalid JSON like +numbers (fixed automatically)

    Args:
        text: Raw response text

    Returns:
        Parsed dict or None
    """
    if not text or not text.strip():
        return None

    # Fix common LLM JSON issues
    text = _fix_json_issues(text)

    # Strategy 1: Try markdown code block first
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 2: Try direct parse (if response is pure JSON)
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find JSON object with brace matching
    # This handles nested braces correctly
    start_idx = text.find("{")
    if start_idx != -1:
        depth = 0
        end_idx = start_idx
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start_idx):
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break

        if depth == 0 and end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

    # Strategy 4: Fallback regex (less precise but catches edge cases)
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def validate_comment_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize comment JSON structure.

    Args:
        data: Parsed JSON dict

    Returns:
        Normalized dict with all expected fields
    """
    return {
        "headline": data.get("headline", "Keine Überschrift"),
        "summary": data.get("summary", []),
        "drivers": data.get("drivers", []),
        "evidence": data.get("evidence", []),
        "questions": data.get("questions", []),
    }
