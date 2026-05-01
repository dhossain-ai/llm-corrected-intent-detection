"""Small Ollama HTTP client for local text correction."""

from __future__ import annotations

import re

import requests


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma3n:e2b"


def is_ollama_available(base_url: str = DEFAULT_BASE_URL) -> bool:
    """Return True when the local Ollama server responds."""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
        return response.ok
    except requests.RequestException:
        return False


def correct_with_ollama(
    text: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = 60,
) -> str:
    """Correct spelling and typing mistakes with a local Ollama model."""
    if not is_ollama_available(base_url=base_url):
        raise RuntimeError(
            "Ollama is unavailable. Start Ollama locally and ensure it is reachable "
            f"at {base_url}."
        )

    prompt = (
        "Correct spelling and typing mistakes in this chatbot message.\n"
        "Do not change the meaning.\n"
        "Do not add new information.\n"
        "Return only the corrected message.\n\n"
        f'Message: "{text}"'
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
    }

    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise RuntimeError(
            f"Ollama correction timed out after {timeout} seconds."
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama correction request failed: {exc}") from exc

    try:
        response_text = response.json().get("response", "")
    except ValueError as exc:
        raise RuntimeError("Ollama returned a non-JSON response.") from exc

    cleaned_text = _clean_ollama_response(response_text)
    if not cleaned_text:
        raise RuntimeError("Ollama returned an empty correction.")
    return cleaned_text


def _clean_ollama_response(response_text: str) -> str:
    cleaned = response_text.strip()
    cleaned = re.sub(r"^```(?:text)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    cleaned = re.sub(
        r"^(corrected|correction|corrected message)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = cleaned.strip("\"'`“”‘’")
    return cleaned.strip()
