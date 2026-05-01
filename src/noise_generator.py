"""Synthetic character-level noise generation for chatbot messages."""

from __future__ import annotations

import random
import re
import string
from collections.abc import Callable


QWERTY_NEIGHBORS: dict[str, tuple[str, ...]] = {
    "q": ("w", "a"),
    "w": ("q", "e", "a", "s"),
    "e": ("w", "r", "s", "d"),
    "r": ("e", "t", "d", "f"),
    "t": ("r", "y", "f", "g"),
    "y": ("t", "u", "g", "h"),
    "u": ("y", "i", "h", "j"),
    "i": ("u", "o", "j", "k"),
    "o": ("i", "p", "k", "l"),
    "p": ("o", "l"),
    "a": ("q", "w", "s", "z"),
    "s": ("w", "e", "a", "d", "z", "x"),
    "d": ("e", "r", "s", "f", "x", "c"),
    "f": ("r", "t", "d", "g", "c", "v"),
    "g": ("t", "y", "f", "h", "v", "b"),
    "h": ("y", "u", "g", "j", "b", "n"),
    "j": ("u", "i", "h", "k", "n", "m"),
    "k": ("i", "o", "j", "l", "m"),
    "l": ("o", "p", "k"),
    "z": ("a", "s", "x"),
    "x": ("z", "s", "d", "c"),
    "c": ("x", "d", "f", "v"),
    "v": ("c", "f", "g", "b"),
    "b": ("v", "g", "h", "n"),
    "n": ("b", "h", "j", "m"),
    "m": ("n", "j", "k"),
}

NoiseOperation = Callable[[str, random.Random], str]


def delete_character(text: str, rng: random.Random) -> str:
    """Delete one non-space character while keeping a non-empty result."""
    if len(text.strip()) <= 1:
        return text

    indices = [idx for idx, char in enumerate(text) if not char.isspace()]
    if len(indices) <= 1:
        return text

    index = rng.choice(indices)
    noisy_text = text[:index] + text[index + 1 :]
    return noisy_text if noisy_text.strip() else text


def insert_character(text: str, rng: random.Random) -> str:
    """Insert one plausible extra character."""
    if not text:
        return text

    insertion = rng.choice(string.ascii_lowercase)
    index = rng.randrange(len(text) + 1)
    noisy_text = text[:index] + insertion + text[index:]
    return noisy_text if noisy_text.strip() else text


def swap_adjacent_characters(text: str, rng: random.Random) -> str:
    """Swap one adjacent pair inside a token."""
    candidates = [
        idx
        for idx in range(len(text) - 1)
        if not text[idx].isspace() and not text[idx + 1].isspace()
    ]
    if not candidates:
        return text

    index = rng.choice(candidates)
    chars = list(text)
    chars[index], chars[index + 1] = chars[index + 1], chars[index]
    noisy_text = "".join(chars)
    return noisy_text if noisy_text.strip() else text


def qwerty_typo(text: str, rng: random.Random) -> str:
    """Replace one character with a neighboring QWERTY key."""
    candidates = [
        idx for idx, char in enumerate(text) if char.lower() in QWERTY_NEIGHBORS
    ]
    if not candidates:
        return text

    index = rng.choice(candidates)
    original = text[index]
    replacement = rng.choice(QWERTY_NEIGHBORS[original.lower()])
    if original.isupper():
        replacement = replacement.upper()

    noisy_text = text[:index] + replacement + text[index + 1 :]
    return noisy_text if noisy_text.strip() else text


def mixed_noise(text: str, noise_level: float = 0.1, seed: int | None = None) -> str:
    """Apply a deterministic mixture of low-impact character corruptions."""
    if not text or noise_level <= 0:
        return text

    rng = random.Random(seed)
    spans = list(re.finditer(r"\S+", text))
    if not spans:
        return text

    operation_count = _operation_count(len(spans), noise_level, rng)
    operation_count = min(operation_count, len(spans))

    noisy_text = text
    operations: tuple[NoiseOperation, ...] = (
        delete_character,
        insert_character,
        swap_adjacent_characters,
        qwerty_typo,
    )

    for _ in range(operation_count):
        before = noisy_text
        operation = rng.choice(operations)
        noisy_text = operation(noisy_text, rng)
        if not noisy_text.strip():
            noisy_text = before

    return noisy_text


def generate_noisy_text(
    text: str, noise_level: float = 0.1, seed: int | None = None
) -> str:
    """Generate reproducible noisy text for one chatbot message."""
    return mixed_noise(text, noise_level=noise_level, seed=seed)


def _operation_count(token_count: int, noise_level: float, rng: random.Random) -> int:
    expected = max(0.0, token_count * noise_level)
    base_count = int(expected)
    if rng.random() < expected - base_count:
        base_count += 1

    if noise_level > 0 and base_count == 0:
        base_count = 1
    if noise_level < 0.5:
        base_count = min(base_count, max(1, token_count // 2))

    return base_count
