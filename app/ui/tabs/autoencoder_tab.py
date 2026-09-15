"""Convolutional Autoencoder baseline evaluation tab for Streamlit UI."""

import streamlit as st

from app.pipelines.modelling.autoencoder import run_autoencoder_pipeline
from app.ui.components.metrics import _display_metrics_row
from app.ui.components.selectors import render_dataset_and_category_selector


def render_autoencoder_tab() -> None:
    """Render the Convolutional Autoencoder baseline evaluation tab."""
    st.header("Convolutional Autoencoder Anomaly Detection")
    st.markdown(
        "Train a minimal Convolutional autoencoder on normal samples and "
        "evaluate reconstruction error on test anomalies."
    )

    data_root, category = render_dataset_and_category_selector(key_prefix="ae")

    col_e, col_b, col_l, col_s = st.columns(4)
    st.session_state.setdefault("ae_epochs", 5)
    epochs = col_e.number_input("Epochs", min_value=1, max_value=50, step=1, key="ae_epochs")
    st.session_state.setdefault("ae_batch", 16)
    batch_size = col_b.number_input("Batch Size", min_value=1, max_value=64, step=4, key="ae_batch")
    st.session_state.setdefault("ae_latent", 64)
    latent_dim = col_l.number_input("Latent Dim", min_value=8, max_value=256, step=8, key="ae_latent")
    st.session_state.setdefault("ae_size", 64)
    img_size = col_s.number_input("Image Size", min_value=32, max_value=128, step=16, key="ae_size")

    if not st.button("Run Autoencoder Training & Evaluation"):
        return

    with st.spinner("Training Autoencoder and evaluating on multi-class anomalies..."):
        try:
            results = run_autoencoder_pipeline(
                data_root=data_root,
                category=category,
                epochs=epochs,
                batch_size=batch_size,
                latent_dim=latent_dim,
                img_size=img_size,
            )
            st.success("Autoencoder execution finished.")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            return

        if "classification_report" in results:
            st.text_area("Classification Report", value=results["classification_report"], height=200)

        _display_metrics_row(results)
