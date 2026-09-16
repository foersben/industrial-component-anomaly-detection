"""Main Streamlit dashboard application for Industrial Component Anomaly Detection."""

import warnings

import streamlit as st

from app.ui.presentation import render_defense_presentation

# Suppress timm deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*timm.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=".*timm.*")


def _render_technical_dashboard() -> None:
    """Load the heavier model tabs only after the dashboard is requested."""
    from app.ui.tabs.autoencoder_tab import render_autoencoder_tab
    from app.ui.tabs.dino_tab import render_dino_tab
    from app.ui.tabs.dummy_tab import render_dummy_evaluation_tab
    from app.ui.tabs.guide_tab import render_evaluation_guide_tab
    from app.ui.tabs.keras_cae_tab import render_keras_cae_tab
    from app.ui.tabs.patchcore_tab import render_baseline_patchcore_tab

    st.title("Industrial Component Anomaly Detection Dashboard")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Evaluation Guide",
            "Dummy Classifier Evaluation",
            "Convolutional Autoencoder Evaluation",
            "Patchcore Evaluation (Image & Pixel Level)",
            "Keras CAE (State-of-the-Art)",
            "DINO Vision Transformer Baselines",
        ]
    )

    with tab1:
        render_evaluation_guide_tab()
    with tab2:
        render_dummy_evaluation_tab()
    with tab3:
        render_autoencoder_tab()
    with tab4:
        render_baseline_patchcore_tab()
    with tab5:
        render_keras_cae_tab()
    with tab6:
        render_dino_tab()


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(page_title="Industrial Anomaly Detection", layout="wide")
    st.markdown(
        """
        <style>
        body:has([data-testid="stPopoverBody"]) [data-testid="stTooltipContent"],
        body:has([data-testid="stPopoverBody"]) [data-testid="stTooltipErrorContent"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state.setdefault("application_mode", "Defense Presentation")
    mode = st.session_state["application_mode"]
    if mode == "Defense Presentation":
        render_defense_presentation()
        return

    mode = st.segmented_control(
        "Application mode",
        ["Defense Presentation", "Technical Dashboard"],
        key="application_mode",
        label_visibility="collapsed",
        width="stretch",
    )
    if mode == "Defense Presentation":
        st.rerun()

    _render_technical_dashboard()


if __name__ == "__main__":
    main()
