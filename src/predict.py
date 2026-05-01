"""Prediction utilities for trained intent classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from src.config import MODELS_DIR
from src.train_tfidf import LABEL_MAPPING_FILENAME, MODEL_FILENAME, VECTORIZER_FILENAME


def load_tfidf_artifacts(models_dir: str | Path | None = None) -> dict:
    """Load TF-IDF baseline artifacts from disk."""
    artifact_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    vectorizer_path = artifact_dir / VECTORIZER_FILENAME
    model_path = artifact_dir / MODEL_FILENAME
    label_mapping_path = artifact_dir / LABEL_MAPPING_FILENAME

    missing = [
        str(path)
        for path in (vectorizer_path, model_path, label_mapping_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing TF-IDF artifact(s): "
            + ", ".join(missing)
            + ". Run python -m src.train_tfidf first."
        )

    return {
        "vectorizer": joblib.load(vectorizer_path),
        "model": joblib.load(model_path),
        "label_mapping": json.loads(label_mapping_path.read_text(encoding="utf-8")),
    }


def predict_intent(text: str, top_k: int = 3) -> dict:
    """Predict an intent and top-k class scores for one text string."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    artifacts = load_tfidf_artifacts()
    vectorizer = artifacts["vectorizer"]
    model = artifacts["model"]

    features = vectorizer.transform([text])
    class_labels = np.asarray(model.classes_)
    scores = _predict_scores(model, features)
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    top_predictions = [
        {
            "intent": str(class_labels[index]),
            "score": float(scores[index]),
        }
        for index in ranked_indices
    ]

    best = top_predictions[0]
    return {
        "text": text,
        "predicted_intent": best["intent"],
        "confidence": best["score"],
        "top_k": top_predictions,
    }


def _predict_scores(model, features) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[0]

    if hasattr(model, "decision_function"):
        decision_scores = np.asarray(model.decision_function(features))
        if decision_scores.ndim > 1:
            decision_scores = decision_scores[0]
        return _softmax(decision_scores)

    predicted_label = model.predict(features)[0]
    return np.asarray([1.0 if label == predicted_label else 0.0 for label in model.classes_])


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum()
