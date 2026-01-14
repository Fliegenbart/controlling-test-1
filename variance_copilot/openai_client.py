"""OpenAI API client for cloud LLM inference."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

import requests

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 60


def get_api_key() -> Optional[str]:
    """Get OpenAI API key from environment or Streamlit secrets."""
    # Try environment variable first
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    # Try Streamlit secrets (for Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    return None


def is_available() -> bool:
    """Check if OpenAI API is configured.

    Returns:
        True if API key is set
    """
    return get_api_key() is not None


def openai_generate(
    user_prompt: str,
    system_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.3,
    timeout: Optional[int] = None,
) -> str:
    """Generate text with OpenAI API.

    Args:
        user_prompt: User prompt
        system_prompt: System prompt
        model: Model name (default: gpt-4o-mini)
        temperature: Sampling temperature
        timeout: Request timeout

    Returns:
        Generated text

    Raises:
        ConnectionError: If API not configured
        RuntimeError: If generation fails
    """
    api_key = get_api_key()
    if not api_key:
        raise ConnectionError("OpenAI API key not configured")

    mdl = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    tout = timeout or int(os.getenv("OPENAI_TIMEOUT", DEFAULT_TIMEOUT))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": mdl,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=tout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Could not connect to OpenAI API")
    except requests.exceptions.Timeout:
        raise RuntimeError(f"OpenAI API timeout after {tout}s")
    except requests.exceptions.HTTPError as e:
        error_msg = "OpenAI API error"
        try:
            error_data = e.response.json()
            if "error" in error_data:
                error_msg = error_data["error"].get("message", str(e))
        except Exception:
            error_msg = str(e)
        raise RuntimeError(f"OpenAI API error: {error_msg}")
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected OpenAI response format: {e}")


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response.

    Reuses the same robust extraction logic as ollama_client.

    Args:
        text: Raw response text

    Returns:
        Parsed dict or None
    """
    if not text or not text.strip():
        return None

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

    # Strategy 4: Fallback regex
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None
