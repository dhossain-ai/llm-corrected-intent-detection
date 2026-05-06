"""Milestone 2 beta Streamlit demo."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.corrector import correct_text
from src.predict import predict_intent


DEFAULT_INPUT = "i want chek my acount balnce"


def main() -> None:
    st.set_page_config(
        page_title="Robust Intent Detection",
        page_icon="🤖",
        layout="centered",
    )
    _render_sidebar()

    st.title("Robust Intent Detection for Noisy Chatbot Messages")
    st.write(
        "Milestone 2 beta demo: enter a noisy chatbot message, optionally correct "
        "it, then classify the intent with the TF-IDF baseline."
    )

    try:
        user_text = st.text_area("User message", value=DEFAULT_INPUT, height=120)
        correction_method = st.selectbox(
            "Correction method",
            options=["none", "spellchecker", "ollama"],
            index=1,
        )
        st.selectbox("Model", options=["TF-IDF baseline"], index=0)

        if st.button("Predict Intent", type="primary"):
            _run_prediction(user_text=user_text, correction_method=correction_method)
    except Exception as exc:
        st.error(f"UI rendering error: {exc}")
        import traceback
        st.code(traceback.format_exc(), language="python")


def _render_sidebar() -> None:
    with st.sidebar:
        st.subheader("Demo Info")
        st.write("Project phase: Milestone 2 Beta")
        st.write("Dataset: CLINC150 / CLINC-OOS")
        st.write("Current model: TF-IDF + Logistic Regression")
        st.write("Correction: None / Spellchecker / Ollama local LLM")


def _run_prediction(user_text: str, correction_method: str) -> None:
    try:
        correction = correct_text(user_text, method=correction_method)
        corrected_text = correction["corrected_text"]

        if correction["error"]:
            st.warning(f"Correction warning: {correction['error']}")

        try:
            prediction = predict_intent(corrected_text, top_k=3)
        except FileNotFoundError as exc:
            st.error(f"Run python -m src.train_tfidf first. Error: {exc}")
            return
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            import traceback
            st.code(traceback.format_exc(), language="python")
            return

        st.subheader("Correction")
        st.write("Original input")
        st.code(correction["original_text"] or "", language=None)
        st.write("Corrected input")
        st.code(corrected_text or "", language=None)
        st.write(f"Correction method: {correction['method']}")

        st.subheader("Prediction")
        st.metric("Predicted intent", prediction["predicted_intent"])
        st.metric("Confidence", f"{prediction['confidence']:.2%}")

        top_k_frame = pd.DataFrame(prediction["top_k"])
        st.table(top_k_frame)
    except Exception as exc:
        st.error(f"Prediction workflow error: {exc}")
        import traceback
        st.code(traceback.format_exc(), language="python")


if __name__ == "__main__":
    main()
