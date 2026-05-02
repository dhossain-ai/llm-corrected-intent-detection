"""Text correction helpers for noisy chatbot messages."""

from __future__ import annotations

import re
from functools import lru_cache

from spellchecker import SpellChecker

from src.ollama_client import correct_with_ollama


SUPPORTED_METHODS = {"none", "spellchecker", "ollama"}


def correct_text(text: str, method: str = "none") -> dict:
    """Correct noisy text using the selected correction method."""
    normalized_method = method.lower().strip()
    if normalized_method not in SUPPORTED_METHODS:
        return _result(
            original_text=text,
            corrected_text=text,
            method=method,
            error=f"Unsupported correction method: {method}",
        )

    if normalized_method == "none":
        return _result(text, text, normalized_method)

    if not text:
        return _result(text, text, normalized_method)

    if normalized_method == "spellchecker":
        corrected_text = _correct_with_spellchecker(text)
        return _result(text, corrected_text, normalized_method)

    try:
        corrected_text = correct_with_ollama(text)
    except RuntimeError as exc:
        return _result(text, text, normalized_method, error=str(exc))

    return _result(text, corrected_text, normalized_method)


def _correct_with_spellchecker(text: str) -> str:
    spellchecker = _spellchecker()
    tokens = re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)
    corrected_tokens = []

    for token in tokens:
        if not token.isalpha() or len(token) <= 1:
            corrected_tokens.append(token)
            continue

        lower_token = token.lower()
        if lower_token not in spellchecker.unknown([lower_token]):
            corrected_tokens.append(token)
            continue

        correction = spellchecker.correction(lower_token)
        if not correction:
            corrected_tokens.append(token)
            continue

        corrected_tokens.append(_match_case(correction, token))

    corrected_text = "".join(corrected_tokens)
    return corrected_text if corrected_text.strip() else text


def _match_case(corrected_token: str, original_token: str) -> str:
    if original_token.isupper():
        return corrected_token.upper()
    if original_token.istitle():
        return corrected_token.title()
    return corrected_token


@lru_cache(maxsize=1)
def _spellchecker() -> SpellChecker:
    return SpellChecker()


def _result(
    original_text: str,
    corrected_text: str,
    method: str,
    error: str | None = None,
) -> dict:
    return {
        "original_text": original_text,
        "corrected_text": corrected_text,
        "method": method,
        "changed": corrected_text != original_text,
        "error": error,
    }
