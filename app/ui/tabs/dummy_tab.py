"""Dummy classifier evaluation tab for Streamlit UI."""

import streamlit as st

from app.ui.components.api_client import make_api_request
from app.ui.components.metrics import _display_metrics_row


def _handle_theoretical_dummy_eval() -> None:
    """Render and execute the theoretical dummy evaluation form."""
    col1, col2 = st.columns(2)
    pixels = col1.number_input("Total Pixels", value=1_000_000, step=100_000)
    ratio = col2.slider("Synthetic Anomaly Ratio", min_value=0.001, max_value=0.05, value=0.015, step=0.001)

    if not st.button("Run Theoretical Evaluation"):
        return

    payload = {"mode": "theoretical", "pixels": pixels, "anomaly_ratio": ratio}
    data = make_api_request("/api/pipelines/dummy", payload)
    if data:
        st.success(data.get("message", "Success"))
        st.metric("Dummy Accuracy", f"{data.get('accuracy', 0) * 100:.2f}%")


def _handle_real_dummy_eval() -> None:
    """Render and execute the real dataset dummy evaluation form."""
    data_root = st.text_input("Dataset Root Path", value="data/raw/mvtec_ad")
    category = st.text_input("Category", value="bottle")

    if not st.button("Run Real Dataset Evaluation"):
        return

    payload = {"mode": "real", "data_root": data_root, "category": category}
    data = make_api_request("/api/pipelines/dummy", payload, timeout=60)
    if not data:
        return

    st.success(data.get("message", "Success"))
    results = data.get("results", {})

    if "error" in results:
        st.error(results["error"])
        return

    st.text_area("Evaluation Summary Output", value=results.get("summary", ""), height=160)
    _display_metrics_row(results)


def render_dummy_evaluation_tab() -> None:
    """Render the dummy classifier evaluation tab."""
    st.header("Dummy Classifier (Accuracy Paradox)")
    st.markdown("Evaluate a dummy classifier predicting all normal pixels on synthetic or real data.")

    mode = st.radio("Evaluation Mode", options=["theoretical", "real"], horizontal=True)

    if mode == "theoretical":
        _handle_theoretical_dummy_eval()
    else:
        _handle_real_dummy_eval()
