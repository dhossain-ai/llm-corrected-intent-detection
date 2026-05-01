"""CLINC150 dataset loading utilities."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import pandas as pd
from datasets import DatasetDict, load_dataset

from src.config import CLINC_DATASET_NAME, CLINC_PREFERRED_CONFIG


REQUIRED_SPLITS = ("train", "validation", "test")


def load_clinc150(
    dataset_name: str = CLINC_DATASET_NAME,
    preferred_config: str = CLINC_PREFERRED_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int], dict[int, str]]:
    """Load CLINC150 and return standardized splits plus label mappings."""
    dataset = _load_dataset_with_fallback(dataset_name, preferred_config)
    _validate_splits(dataset)

    label2id, id2label = _build_label_mappings(dataset)

    train_df = _standardize_split(dataset["train"], "train", id2label)
    validation_df = _standardize_split(dataset["validation"], "validation", id2label)
    test_df = _standardize_split(dataset["test"], "test", id2label)

    return train_df, validation_df, test_df, label2id, id2label


def _load_dataset_with_fallback(dataset_name: str, preferred_config: str) -> DatasetDict:
    errors: list[str] = []

    try:
        dataset = load_dataset(dataset_name, preferred_config)
        return _as_dataset_dict(dataset)
    except Exception as exc:
        errors.append(f"{dataset_name!r} with config {preferred_config!r}: {exc}")

    try:
        dataset = load_dataset(dataset_name)
        return _as_dataset_dict(dataset)
    except Exception as exc:
        errors.append(f"{dataset_name!r} without config: {exc}")

    joined_errors = "\n\n".join(errors)
    raise RuntimeError(
        "Unable to download the CLINC150 dataset from Hugging Face. "
        "Check your internet connection, Hugging Face availability, and whether "
        f"the dataset {dataset_name!r} is accessible.\n\nAttempts failed:\n"
        f"{joined_errors}"
    )


def _as_dataset_dict(dataset: Any) -> DatasetDict:
    if isinstance(dataset, DatasetDict):
        return dataset
    raise TypeError(
        "Expected Hugging Face load_dataset to return a DatasetDict with train, "
        f"validation, and test splits, but got {type(dataset).__name__}."
    )


def _validate_splits(dataset: DatasetDict) -> None:
    missing_splits = [split for split in REQUIRED_SPLITS if split not in dataset]
    if missing_splits:
        available = ", ".join(dataset.keys())
        missing = ", ".join(missing_splits)
        raise ValueError(
            f"CLINC150 dataset is missing required split(s): {missing}. "
            f"Available split(s): {available}."
        )


def _build_label_mappings(dataset: DatasetDict) -> tuple[dict[str, int], dict[int, str]]:
    intent_feature = dataset["train"].features.get("intent")
    label_names = getattr(intent_feature, "names", None)

    if label_names:
        label2id = {label: idx for idx, label in enumerate(label_names)}
        id2label = {idx: label for label, idx in label2id.items()}
        return label2id, id2label

    labels = sorted(
        {
            str(row["intent"])
            for split in REQUIRED_SPLITS
            for row in dataset[split]
            if "intent" in row
        }
    )
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def _standardize_split(
    split_data: Any,
    split_name: str,
    id2label: Mapping[int, str],
) -> pd.DataFrame:
    missing_columns = {"text", "intent"} - set(split_data.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Split {split_name!r} is missing required column(s): {missing}.")

    frame = split_data.to_pandas()[["text", "intent"]].copy()
    frame["intent"] = frame["intent"].map(lambda value: _format_intent(value, id2label))
    frame["split"] = split_name
    return frame[["text", "intent", "split"]]


def _format_intent(value: Any, id2label: Mapping[int, str]) -> str:
    if isinstance(value, Integral) and int(value) in id2label:
        return id2label[int(value)]
    return str(value)


def main() -> None:
    train_df, validation_df, test_df, label2id, _ = load_clinc150()

    print(f"train sample count: {len(train_df)}")
    print(f"validation sample count: {len(validation_df)}")
    print(f"test sample count: {len(test_df)}")
    print(f"number of intent labels: {len(label2id)}")
    print("5 sample rows:")
    print(train_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
