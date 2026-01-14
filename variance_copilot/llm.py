"""Unified LLM interface with automatic backend selection.

Supports:
- Ollama (local, private) - preferred when available
- OpenAI API (cloud) - fallback for deployments

Backend selection priority:
1. If Ollama is running locally → use Ollama
2. If OPENAI_API_KEY is set → use OpenAI
3. Otherwise → AI features disabled
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from . import ollama_client
from . import openai_client


class Backend(Enum):
    """Available LLM backends."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    NONE = "none"


@dataclass
class BackendStatus:
    """Status of LLM backend."""
    backend: Backend
    available: bool
    model: str
    display_name: str
    icon: str


def detect_backend(
    prefer_ollama: bool = True,
    ollama_url: Optional[str] = None,
) -> BackendStatus:
    """Detect which LLM backend is available.

    Args:
        prefer_ollama: Prefer Ollama over OpenAI when both available
        ollama_url: Custom Ollama URL to check

    Returns:
        BackendStatus with detected backend info
    """
    ollama_available = ollama_client.is_available(ollama_url)
    openai_available = openai_client.is_available()

    if prefer_ollama and ollama_available:
        config = ollama_client.get_config()
        return BackendStatus(
            backend=Backend.OLLAMA,
            available=True,
            model=config["model"],
            display_name="Ollama (Local)",
            icon="🏠",
        )
    elif openai_available:
        import os
        model = os.getenv("OPENAI_MODEL", openai_client.DEFAULT_MODEL)
        return BackendStatus(
            backend=Backend.OPENAI,
            available=True,
            model=model,
            display_name="OpenAI (Cloud)",
            icon="☁️",
        )
    elif ollama_available:
        # Ollama available but not preferred (edge case)
        config = ollama_client.get_config()
        return BackendStatus(
            backend=Backend.OLLAMA,
            available=True,
            model=config["model"],
            display_name="Ollama (Local)",
            icon="🏠",
        )
    else:
        return BackendStatus(
            backend=Backend.NONE,
            available=False,
            model="",
            display_name="No AI Available",
            icon="⚠️",
        )


def generate(
    user_prompt: str,
    system_prompt: str,
    backend: Optional[Backend] = None,
    ollama_url: Optional[str] = None,
    ollama_model: Optional[str] = None,
    openai_model: Optional[str] = None,
    temperature: float = 0.3,
) -> str:
    """Generate text using the appropriate backend.

    Args:
        user_prompt: User prompt
        system_prompt: System prompt
        backend: Force specific backend (auto-detect if None)
        ollama_url: Custom Ollama URL
        ollama_model: Custom Ollama model
        openai_model: Custom OpenAI model
        temperature: Sampling temperature

    Returns:
        Generated text

    Raises:
        ConnectionError: If no backend available
        RuntimeError: If generation fails
    """
    if backend is None:
        status = detect_backend(ollama_url=ollama_url)
        backend = status.backend

    if backend == Backend.OLLAMA:
        return ollama_client.ollama_generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            base_url=ollama_url,
            model=ollama_model,
            temperature=temperature,
        )
    elif backend == Backend.OPENAI:
        return openai_client.openai_generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=openai_model,
            temperature=temperature,
        )
    else:
        raise ConnectionError(
            "No LLM backend available. "
            "Either run Ollama locally (ollama serve) or set OPENAI_API_KEY."
        )


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response.

    Uses the same extraction logic regardless of backend.

    Args:
        text: Raw response text

    Returns:
        Parsed dict or None
    """
    # Both clients have identical extraction logic, use either
    return ollama_client.extract_json(text)


def get_backend_info(ollama_url: Optional[str] = None) -> Dict[str, Any]:
    """Get detailed information about available backends.

    Args:
        ollama_url: Custom Ollama URL to check

    Returns:
        Dict with backend availability info
    """
    ollama_available = ollama_client.is_available(ollama_url)
    openai_available = openai_client.is_available()

    ollama_config = ollama_client.get_config()

    return {
        "ollama": {
            "available": ollama_available,
            "url": ollama_url or ollama_config["base_url"],
            "model": ollama_config["model"],
        },
        "openai": {
            "available": openai_available,
            "model": openai_client.DEFAULT_MODEL,
        },
        "active": detect_backend(ollama_url=ollama_url),
    }
