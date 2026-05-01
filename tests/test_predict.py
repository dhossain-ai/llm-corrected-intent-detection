import pytest


def test_predict_imports_cleanly():
    import src.predict  # noqa: F401


def test_missing_tfidf_artifacts_raise_helpful_error(tmp_path):
    from src import predict

    with pytest.raises(FileNotFoundError, match="Run python -m src.train_tfidf first"):
        predict.load_tfidf_artifacts(models_dir=tmp_path)
