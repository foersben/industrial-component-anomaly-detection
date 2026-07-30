"""Streamlit UI app."""

import warnings
from typing import Any

import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"

# Suppress timm deprecation warnings emitted by downstream libraries
warnings.filterwarnings("ignore", category=FutureWarning, message=".*timm.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=".*timm.*")


def make_api_request(endpoint: str, payload: dict[str, Any], timeout: int = 10) -> Any:
    """Helper to handle repetitive POST requests and error catching.

    Args:
        endpoint: API endpoint to call
        payload: Payload to send to the API
        timeout: Timeout for the request

    Returns:
        Response from the API
    """
    try:
        res = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=timeout)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.ReadTimeout:
        st.error("Request timed out. This process might take longer.")
    except requests.exceptions.ConnectionError:
        st.error("Backend unreachable. Ensure FastAPI server is running.")
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
    return None


def render_dummy_evaluation_tab() -> None:
    """Render the dummy classifier evaluation tab.

    The user can select a mode (theoretical or real) and run the dummy classifier.
    """
    st.header("Dummy Classifier (Accuracy Paradox)")
    st.markdown("Evaluate a dummy classifier predicting all normal pixels on synthetic or real data.")

    mode = st.radio("Evaluation Mode", options=["theoretical", "real"], horizontal=True)

    if mode == "theoretical":
        col1, col2 = st.columns(2)
        pixels = col1.number_input("Total Pixels", min_value=1000, value=1000000, step=100000)
        ratio = col2.slider("Anomaly Ratio", min_value=0.001, max_value=0.200, value=0.015, step=0.005)

        if st.button("Run Theoretical Evaluation"):
            data = make_api_request(
                "/api/pipelines/dummy", {"mode": "theoretical", "pixels": pixels, "anomaly_ratio": ratio}
            )
            if data:
                st.success(data.get("message", "Success"))
                st.metric("Dummy Accuracy", f"{data.get('accuracy', 0) * 100:.2f}%")

    else:
        data_root = st.text_input("Dataset Root Path", value="data/raw/mvtec_ad")
        category = st.text_input("Category", value="bottle")

        if st.button("Run Real Dataset Evaluation"):
            data = make_api_request(
                "/api/pipelines/dummy", {"mode": "real", "data_root": data_root, "category": category}, timeout=60
            )
            if data:
                st.success(data.get("message", "Success"))
                results = data.get("results", {})

                if "error" in results:
                    st.error(results["error"])
                else:
                    st.text_area("Evaluation Summary Output", value=results.get("summary", ""), height=160)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{results.get('accuracy', 0) * 100:.2f}%")
                    col2.metric("Precision", f"{results.get('precision', 0):.2f}")
                    col3.metric("Recall", f"{results.get('recall', 0):.2f}")


def render_baseline_patchcore_tab() -> None:
    """Render the baseline Patchcore evaluation tab.

    The user can select a category from the MVTec AD dataset and run the Patchcore
    model training and evaluation on the selected category.
    """
    st.header("Baseline Patchcore Anomaly Detection")
    st.markdown("Run Patchcore model training and evaluation on MVTec AD dataset.")

    data_root = st.text_input("Baseline Dataset Root", value="data/raw/mvtec_ad", key="b_root")
    category = st.text_input("Baseline Category", value="bottle", key="b_cat")

    if st.button("Run Patchcore Baseline Pipeline"):
        with st.spinner("Running Patchcore evaluation (this might take a while)..."):
            data = make_api_request(
                "/api/pipelines/baseline", {"data_root": data_root, "category": category}, timeout=300
            )
            if data:
                st.success(data.get("message", "Success"))
                st.text_area("Results Summary", value=data.get("results", ""), height=200)


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(page_title="Industrial Anomaly Detection", layout="wide")
    st.title("Industrial Component Anomaly Detection Dashboard")

    tab1, tab2 = st.tabs(["Dummy Classifier Evaluation", "Baseline Patchcore Evaluation"])

    with tab1:
        render_dummy_evaluation_tab()
    with tab2:
        render_baseline_patchcore_tab()


if __name__ == "__main__":
    main()
