"""Evaluate trained intent classifiers on clean and noisy CLINC150 test sets."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.config import METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR
from src.metrics import compute_classification_metrics
from src.train_tfidf import MODEL_FILENAME, VECTORIZER_FILENAME
from src.utils import ensure_directory


EVALUATION_FILES = {
    "clean": ("clinc150_test_clean.csv", 0.0),
    "noisy_05": ("clinc150_test_noisy_05.csv", 0.05),
    "noisy_10": ("clinc150_test_noisy_10.csv", 0.10),
    "noisy_20": ("clinc150_test_noisy_20.csv", 0.20),
}


def evaluate_tfidf(
    processed_dir: str | Path = PROCESSED_DATA_DIR,
    models_dir: str | Path = MODELS_DIR,
    metrics_dir: str | Path = METRICS_DIR,
    predictions_dir: str | Path = PREDICTIONS_DIR,
    sample_predictions: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the TF-IDF baseline and save metric/prediction CSVs."""
    vectorizer, model = _load_tfidf_artifacts(models_dir)
    datasets = _load_evaluation_datasets(processed_dir)

    result_rows: list[dict[str, float | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    clean_accuracy: float | None = None

    for condition, (frame, noise_level) in datasets.items():
        prediction_text = frame["noisy_text"].astype(str)
        predictions = model.predict(vectorizer.transform(prediction_text))
        metric_values = compute_classification_metrics(frame["intent"], predictions)

        if condition == "clean":
            clean_accuracy = metric_values["accuracy"]
            robustness_score = 1.0
        else:
            robustness_score = (
                metric_values["accuracy"] / clean_accuracy
                if clean_accuracy
                else 0.0
            )

        result_rows.append(
            {
                "model": "tfidf",
                "condition": condition,
                "noise_level": noise_level,
                **metric_values,
                "robustness_score": float(robustness_score),
            }
        )

        prediction_frames.append(
            _build_prediction_frame(condition, frame, predictions)
        )

    results_df = pd.DataFrame(result_rows)
    sample_predictions_df = (
        pd.concat(prediction_frames, ignore_index=True)
        .head(sample_predictions)
        .copy()
    )

    ensure_directory(metrics_dir)
    ensure_directory(predictions_dir)
    results_df.to_csv(Path(metrics_dir) / "tfidf_results.csv", index=False)
    sample_predictions_df.to_csv(
        Path(predictions_dir) / "tfidf_sample_predictions.csv",
        index=False,
    )

    return results_df, sample_predictions_df


def _load_tfidf_artifacts(models_dir: str | Path):
    models_path = Path(models_dir)
    vectorizer_path = models_path / VECTORIZER_FILENAME
    model_path = models_path / MODEL_FILENAME
    missing = [
        str(path)
        for path in (vectorizer_path, model_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing TF-IDF artifact(s): "
            + ", ".join(missing)
            + ". Run python -m src.train_tfidf first."
        )
    return joblib.load(vectorizer_path), joblib.load(model_path)


def _load_evaluation_datasets(
    processed_dir: str | Path,
) -> dict[str, tuple[pd.DataFrame, float]]:
    processed_path = Path(processed_dir)
    missing = [
        str(processed_path / filename)
        for filename, _ in EVALUATION_FILES.values()
        if not (processed_path / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing processed CLINC150 test file(s): "
            + ", ".join(missing)
            + ". Run python -m src.noise_generator first."
        )

    datasets: dict[str, tuple[pd.DataFrame, float]] = {}
    for condition, (filename, noise_level) in EVALUATION_FILES.items():
        frame = pd.read_csv(processed_path / filename)
        _validate_evaluation_frame(frame, filename)
        datasets[condition] = (frame, noise_level)
    return datasets


def _validate_evaluation_frame(frame: pd.DataFrame, filename: str) -> None:
    required_columns = {"text", "noisy_text", "intent", "split", "noise_level"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{filename} is missing required column(s): {missing}.")


def _build_prediction_frame(
    condition: str,
    frame: pd.DataFrame,
    predictions,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "condition": condition,
            "text": frame["text"],
            "noisy_text": frame["noisy_text"],
            "true_intent": frame["intent"],
            "predicted_intent": predictions,
        }
    )
    output["correct"] = output["true_intent"] == output["predicted_intent"]
    return output[
        [
            "condition",
            "text",
            "noisy_text",
            "true_intent",
            "predicted_intent",
            "correct",
        ]
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained intent models.")
    parser.add_argument("--model", choices=["tfidf"], required=True)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_df, sample_predictions_df = evaluate_tfidf(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        metrics_dir=args.metrics_dir,
        predictions_dir=args.predictions_dir,
    )

    print("TF-IDF evaluation results:")
    print(results_df.to_string(index=False))
    print(f"sample predictions saved: {len(sample_predictions_df)} rows")


if __name__ == "__main__":
    main()
