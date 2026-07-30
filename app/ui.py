"""Streamlit UI app."""

import warnings
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from app.pipelines.evaluation.visualization import render_evaluation_curves

BACKEND_URL = "http://127.0.0.1:8000"

# Suppress timm deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*timm.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=".*timm.*")


def make_api_request(endpoint: str, payload: dict[str, Any], timeout: int = 10) -> Any:
    """Helper to handle repetitive POST requests and error catching.

    Args:
        endpoint: The API endpoint to call.
        payload: The payload to send to the API.
        timeout: The timeout for the request.

    Returns:
        The response from the API.
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


def _display_metrics_row(metrics: dict[str, Any]) -> None:
    """Helper to render 3-column metric cards (Accuracy, Precision, Recall).

    Args:
        metrics: The metrics to display.
    """
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f}%")
    col2.metric("Precision", f"{metrics.get('precision', 0):.2f}")
    col3.metric("Recall", f"{metrics.get('recall', 0):.2f}")


def _display_level_metrics(title: str, metrics: dict[str, Any]) -> None:
    """Helper to render level-specific metrics (Image or Pixel).

    Args:
        title: The title of the metrics.
        metrics: The metrics to display.
    """
    st.subheader(title)
    st.metric("AUROC", f"{metrics.get('auroc', 0.0):.4f}")
    st.metric("F1-Score", f"{metrics.get('f1_score', 0.0):.4f}")
    st.caption(f"Saved: `{metrics.get('metrics_path', '')}`")


def _find_metric_files() -> list[str]:
    """Search for saved metric .npz files in default results directories.

    Returns:
        A list of metric file paths.
    """
    found_files: list[str] = []
    for search_dir in [Path("results"), Path("data/external")]:
        if search_dir.exists():
            found_files.extend([str(p) for p in search_dir.rglob("*.npz") if not p.name.startswith(".")])
    return sorted(found_files)


def _handle_theoretical_dummy_eval() -> None:
    """Render and execute the theoretical dummy evaluation form."""
    col1, col2 = st.columns(2)
    pixels = col1.number_input("Total Pixels", min_value=1000, value=1000000, step=100000)
    ratio = col2.slider("Anomaly Ratio", min_value=0.001, max_value=0.200, value=0.015, step=0.005)

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


def render_baseline_patchcore_tab() -> None:
    """Render the Patchcore anomaly detection evaluation tab."""
    st.header("Patchcore Anomaly Detection & Evaluation")
    st.markdown("Run Patchcore model training and evaluation on MVTec AD dataset (Image & Pixel level).")

    data_root = st.text_input("Dataset Root Directory", value="data/raw/mvtec_ad", key="b_root")
    category = st.text_input("Category Name", value="bottle", key="b_cat")

    if not st.button("Run Patchcore Evaluation Pipeline"):
        return

    with st.spinner("Fitting model and evaluating Image & Pixel level metrics..."):
        payload = {"data_root": data_root, "category": category}
        data = make_api_request("/api/pipelines/baseline", payload, timeout=300)

        if not data:
            return

        st.success(data.get("message", "Success"))
        results = data.get("results", {})

        if isinstance(results, dict):
            col_img, col_pix = st.columns(2)
            with col_img:
                _display_level_metrics("Image-Level (Classification)", results.get("image_level", {}))
            with col_pix:
                _display_level_metrics("Pixel-Level (Localization)", results.get("pixel_level", {}))
        else:
            st.text_area("Results Summary", value=str(results), height=180)


def render_evaluation_curves_tab() -> None:
    """Render the evaluation tradeoff and PR curves tab."""
    st.header("Evaluation Curves")
    st.markdown("Render evaluation tradeoff and PR curves from saved metric `.npz` files in `results/`.")

    found_files = _find_metric_files()

    if found_files:
        selected_file = st.selectbox("Select saved metrics file from results/", options=found_files)
        data_path = st.text_input("Or enter custom path:", value=selected_file)
    else:
        st.info("No `.npz` files detected in `results/`. Enter path manually or generate one during pipeline runs.")
        data_path = st.text_input("Metrics File Path (.npz)", value="results/metrics.npz")

    if st.button("Render Curves"):
        render_evaluation_curves(data_path)


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(page_title="Industrial Anomaly Detection", layout="wide")
    st.title("Industrial Component Anomaly Detection Dashboard")

    tab1, tab2, tab3 = st.tabs(
        [
            "Dummy Classifier Evaluation",
            "Patchcore Evaluation (Image & Pixel Level)",
            "Evaluation Curves",
        ]
    )

    with tab1:
        render_dummy_evaluation_tab()
    with tab2:
        render_baseline_patchcore_tab()
    with tab3:
        render_evaluation_curves_tab()


if __name__ == "__main__":
    main()
