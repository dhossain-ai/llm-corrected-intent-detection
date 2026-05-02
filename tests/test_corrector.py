from src.corrector import correct_text


def test_correct_text_none_returns_original():
    result = correct_text("i want chek my acount balnce", method="none")

    assert result["original_text"] == "i want chek my acount balnce"
    assert result["corrected_text"] == "i want chek my acount balnce"
    assert result["method"] == "none"
    assert result["changed"] is False
    assert result["error"] is None


def test_spellchecker_returns_non_empty_text():
    result = correct_text("i want chek my acount balnce", method="spellchecker")

    assert result["corrected_text"].strip()
    assert result["method"] == "spellchecker"
    assert result["error"] is None


def test_correct_text_handles_empty_string():
    result = correct_text("", method="spellchecker")

    assert result["original_text"] == ""
    assert result["corrected_text"] == ""
    assert result["changed"] is False
    assert result["error"] is None


def test_ollama_unavailable_returns_original_with_error(monkeypatch):
    def failing_correction(text):
        raise RuntimeError("Ollama is unavailable")

    monkeypatch.setattr("src.corrector.correct_with_ollama", failing_correction)

    result = correct_text("i want chek my acount balnce", method="ollama")

    assert result["corrected_text"] == "i want chek my acount balnce"
    assert result["method"] == "ollama"
    assert result["changed"] is False
    assert "Ollama is unavailable" in result["error"]


def test_streamlit_app_imports_cleanly():
    import app.streamlit_app  # noqa: F401
