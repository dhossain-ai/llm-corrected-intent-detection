"""Train a TF-IDF + Logistic Regression intent baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import MODELS_DIR
from src.data_loader import load_clinc150
from src.utils import ensure_directory


VECTORIZER_FILENAME = "tfidf_vectorizer.joblib"
MODEL_FILENAME = "tfidf_model.joblib"
LABEL_MAPPING_FILENAME = "label_mapping.json"


def train_tfidf_baseline(
    max_features: int = 50000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_iter: int = 1000,
    class_weight: str | None = None,
) -> tuple[
    TfidfVectorizer,
    OneVsRestClassifier,
    dict[str, int],
    dict[int, str],
]:
    """Train the TF-IDF baseline on the CLINC150 train split."""
    train_df, _, _, label2id, id2label = load_clinc150()

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        lowercase=True,
        sublinear_tf=True,
    )
    model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            solver="liblinear",
            random_state=42,
        ),
        n_jobs=1,
    )

    features = vectorizer.fit_transform(train_df["text"])
    features.sort_indices()
    model.fit(features, train_df["intent"])

    return vectorizer, model, label2id, id2label


def save_tfidf_artifacts(
    vectorizer: TfidfVectorizer,
    model: OneVsRestClassifier,
    label2id: dict[str, int],
    id2label: dict[int, str],
    output_dir: str | Path = MODELS_DIR,
) -> dict[str, Path]:
    """Persist trained baseline artifacts."""
    model_dir = ensure_directory(output_dir)

    vectorizer_path = model_dir / VECTORIZER_FILENAME
    model_path = model_dir / MODEL_FILENAME
    label_mapping_path = model_dir / LABEL_MAPPING_FILENAME

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)

    label_mapping = {
        "label2id": label2id,
        "id2label": {str(label_id): label for label_id, label in id2label.items()},
        "model_classes": list(model.classes_),
    }
    label_mapping_path.write_text(json.dumps(label_mapping, indent=2), encoding="utf-8")

    return {
        "vectorizer": vectorizer_path,
        "model": model_path,
        "label_mapping": label_mapping_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a TF-IDF + Logistic Regression intent classifier."
    )
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--class-weight", choices=["balanced"], default=None)
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vectorizer, model, label2id, id2label = train_tfidf_baseline(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_iter=args.max_iter,
        class_weight=args.class_weight,
    )
    paths = save_tfidf_artifacts(
        vectorizer=vectorizer,
        model=model,
        label2id=label2id,
        id2label=id2label,
        output_dir=args.output_dir,
    )

    print("trained TF-IDF baseline")
    print(f"intent labels: {len(label2id)}")
    for artifact_name, artifact_path in paths.items():
        print(f"saved {artifact_name}: {artifact_path}")


if __name__ == "__main__":
    main()
