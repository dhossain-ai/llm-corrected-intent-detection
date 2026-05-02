import random

import pandas as pd

from src.noise_generator import (
    build_noisy_test_frames,
    delete_character,
    generate_noisy_text,
    insert_character,
    mixed_noise,
    qwerty_typo,
    swap_adjacent_characters,
)


def test_noise_generator_imports_successfully():
    import src.noise_generator  # noqa: F401


def test_same_seed_gives_same_noisy_output():
    text = "please transfer money from checking to savings"

    first = generate_noisy_text(text, noise_level=0.1, seed=123)
    second = generate_noisy_text(text, noise_level=0.1, seed=123)

    assert first == second


def test_zero_noise_level_returns_original_text():
    text = "what is my account balance"

    assert mixed_noise(text, noise_level=0, seed=123) == text
    assert generate_noisy_text(text, noise_level=0, seed=123) == text


def test_generated_noisy_text_is_non_empty():
    noisy_text = generate_noisy_text("book a flight tomorrow", noise_level=0.2, seed=7)

    assert noisy_text.strip()


def test_at_least_one_noise_function_changes_normal_sentence():
    text = "please book a taxi for tonight"
    operations = (
        delete_character,
        insert_character,
        swap_adjacent_characters,
        qwerty_typo,
    )

    changed = [operation(text, random.Random(3)) != text for operation in operations]

    assert any(changed)


def test_generate_noisy_text_handles_short_text_safely():
    noisy_text = generate_noisy_text("a", noise_level=0.2, seed=9)

    assert noisy_text.strip()


def test_noisy_frame_keeps_labels_unchanged():
    test_df = pd.DataFrame(
        {
            "text": ["pay my bill", "find a restaurant"],
            "intent": ["pay_bill", "restaurant_search"],
            "split": ["test", "test"],
        }
    )

    frames = build_noisy_test_frames(test_df, seed=42)

    assert list(frames[0.10]["intent"]) == list(test_df["intent"])
    assert list(frames[0.10].columns) == [
        "text",
        "noisy_text",
        "intent",
        "split",
        "noise_level",
    ]
