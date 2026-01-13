"""Ollama API client for local LLM inference."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

import requests

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
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


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response.

    Handles markdown code blocks and embedded JSON.

    Args:
        text: Raw response text

    Returns:
        Parsed dict or None
    """
    # Try markdown code block first
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None
