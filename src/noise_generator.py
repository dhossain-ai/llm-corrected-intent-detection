"""Synthetic character-level noise generation for chatbot messages."""

from __future__ import annotations

import argparse
import random
import re
import string
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.data_loader import load_clinc150
from src.utils import ensure_directory


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
SUPPORTED_NOISE_LEVELS = (0.05, 0.10, 0.20)


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


def build_noisy_test_frames(
    test_df: pd.DataFrame,
    seed: int = 42,
) -> dict[float, pd.DataFrame]:
    """Create clean and noisy CLINC150 test DataFrames keyed by noise level."""
    frames = {0.0: _format_output_frame(test_df, test_df["text"], 0.0)}

    for noise_level in SUPPORTED_NOISE_LEVELS:
        noisy_text = [
            generate_noisy_text(
                text,
                noise_level=noise_level,
                seed=seed + row_index + int(noise_level * 1000),
            )
            for row_index, text in enumerate(test_df["text"].astype(str))
        ]
        frames[noise_level] = _format_output_frame(test_df, noisy_text, noise_level)

    return frames


def export_noisy_test_files(
    output_dir: str | Path = PROCESSED_DATA_DIR,
    seed: int = 42,
    sample_size: int | None = None,
) -> dict[float, Path]:
    """Load CLINC150 test data and export clean/noisy CSV files."""
    _, _, test_df, _, _ = load_clinc150()
    if sample_size is not None:
        test_df = test_df.head(sample_size).copy()

    output_path = ensure_directory(output_dir)
    frames = build_noisy_test_frames(test_df, seed=seed)
    paths: dict[float, Path] = {}

    for noise_level, frame in frames.items():
        file_path = output_path / _output_filename(noise_level)
        frame.to_csv(file_path, index=False)
        paths[noise_level] = file_path

    return paths


def _format_output_frame(
    source_df: pd.DataFrame,
    noisy_text: pd.Series | list[str],
    noise_level: float,
) -> pd.DataFrame:
    frame = source_df[["text", "intent", "split"]].copy()
    frame["noisy_text"] = list(noisy_text)
    frame["noise_level"] = noise_level
    return frame[["text", "noisy_text", "intent", "split", "noise_level"]]


def _output_filename(noise_level: float) -> str:
    if noise_level == 0.0:
        return "clinc150_test_clean.csv"
    return f"clinc150_test_noisy_{int(noise_level * 100):02d}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clean and synthetic noisy CLINC150 test CSV files."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = export_noisy_test_files(
        output_dir=args.output_dir,
        seed=args.seed,
        sample_size=args.sample_size,
    )

    for noise_level, path in paths.items():
        print(f"wrote noise_level={noise_level:.2f}: {path}")

    noisy_sample = pd.read_csv(paths[0.10]).head(5)
    print("sample noisy rows:")
    print(noisy_sample[["text", "noisy_text", "intent"]].to_string(index=False))


if __name__ == "__main__":
    main()
