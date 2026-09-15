"""Dummy classifier evaluation tab for Streamlit UI."""

import streamlit as st

from app.domain.categories import discover_dataset_categories
from app.pipelines.modelling.dummy_classifier import run_dummy_evaluation, run_real_data_dummy
from app.ui.components.metrics import _display_metrics_row


def _handle_theoretical_dummy_eval() -> None:
    """Render and execute the theoretical dummy evaluation form."""
    col1, col2 = st.columns(2)
    pixels = col1.number_input("Total Pixels", value=1_000_000, step=100_000)
    ratio = col2.slider("Synthetic Anomaly Ratio", min_value=0.001, max_value=0.05, value=0.015, step=0.001)

    if not st.button("Run Theoretical Evaluation"):
        return

    with st.spinner("Running theoretical evaluation..."):
        try:
            accuracy = run_dummy_evaluation(total_pixels=int(pixels), anomaly_ratio=ratio)
            st.success("Success")
            st.metric("Dummy Accuracy", f"{accuracy * 100:.2f}%")
        except Exception as e:
            st.error(f"Pipeline error: {e}")


def _handle_real_dummy_eval() -> None:
    """Render and execute the real dataset dummy evaluation form."""
    col1, col2 = st.columns(2)
    data_root = col1.text_input("Dataset Root Path", value="data/raw/mvtec_ad")
    categories = discover_dataset_categories(data_root)
    default_idx = categories.index("bottle") if "bottle" in categories else 0
    category = col2.selectbox("Category", options=categories, index=default_idx)

    if not st.button("Run Real Dataset Evaluation"):
        return

    with st.spinner("Running evaluation on real data..."):
        try:
            results = run_real_data_dummy(data_root=data_root, category=category)
            if "error" in results:
                st.error(results["error"])
                return

            st.success("Success")
            st.text_area("Evaluation Summary Output", value=results.get("summary", ""), height=160)
            _display_metrics_row(results)
        except Exception as e:
            st.error(f"Pipeline error: {e}")


def render_dummy_evaluation_tab() -> None:
    """Render the dummy classifier evaluation tab."""
    st.header("Dummy Classifier (Accuracy Paradox)")
    st.markdown("Evaluate a dummy classifier predicting all normal pixels on synthetic or real data.")

    mode = st.radio("Evaluation Mode", options=["theoretical", "real"], horizontal=True)

    if mode == "theoretical":
        _handle_theoretical_dummy_eval()
    else:
        _handle_real_dummy_eval()
