"""DINO vision foundation model evaluation tab for Streamlit UI."""

from pathlib import Path
from typing import Literal

import streamlit as st

from app.pipelines.evaluation.visualization import render_evaluation_curves
from app.ui.components.api_client import make_api_request
from app.ui.components.heatmaps import _render_heatmap_explorer
from app.ui.components.metrics import _render_evaluation_summary


def render_dino_tab() -> None:
    """Render the DINO vision transformer nearest-neighbor baselines tab (DINOv2 & DINOv3)."""
    st.header("DINO Vision Transformer Baselines")
    st.markdown(
        "Evaluate frozen **DINOv2** and **DINOv3** patch nearest-neighbor baselines on MVTec AD. "
        "Supports stock and enhanced multi-layer position/density-aware scorers."
    )

    col_arch, col_cat = st.columns(2)
    architecture = col_arch.radio("Architecture", ["DINOv2", "DINOv3"], horizontal=True)

    mvtec_categories = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
        "all",
    ]
    category = col_cat.selectbox("Category", options=mvtec_categories, key="dino_cat")

    st.subheader("Model Configuration")
    col1, col2, col3 = st.columns(3)
    num_neighbors = col1.slider("Patch Nearest Neighbors (k)", min_value=1, max_value=10, value=1, step=1)

    if architecture == "DINOv2":
        variant = col2.selectbox(
            "Scorer Variant", ["baseline", "enhanced"], help="Stock or enhanced position/density-aware scorer"
        )
        masking_options: list[Literal["off", "on", "published"]] = ["published", "off", "on"]
        masking = col3.selectbox("Foreground Masking Policy", options=masking_options, index=0)
    else:
        variant = "baseline"
        masking = "off"
        col2.info("DINOv3 uses frozen backbone with stock scorer.")

    st.subheader("Evaluation Settings")
    data_root = st.text_input("Dataset Root Directory", value="data/raw/mvtec_ad", key="dino_root")
    run_heatmap = st.checkbox("Generate Anomaly Prediction Heatmaps", value=False, key="dino_heatmap")

    if not st.button(f"Run {architecture} Baseline Evaluation", type="primary", key="btn_run_dino"):
        return

    endpoint = "/api/pipelines/dinov2" if architecture == "DINOv2" else "/api/pipelines/dinov3"
    payload = {
        "data_root": data_root,
        "category": category,
        "num_neighbors": num_neighbors,
        "masking": masking,
        "run_heatmap": run_heatmap,
    }
    if architecture == "DINOv2":
        payload["variant"] = variant

    with st.spinner(f"Running {architecture} nearest-neighbor evaluation on '{category}'..."):
        data = make_api_request(endpoint, payload, timeout=300)

    if not data:
        return

    st.success(data.get("message", "Success"))
    results = data.get("results", {})

    if category == "all" and "macro_average" in results:
        st.subheader("Macro Averages Across All Categories")
        macro = results.get("macro_average", {})
        m1, m2, m3 = st.columns(3)
        m1.metric("Image AUROC", f"{macro.get('image_auroc', 0.0):.4f}")
        m2.metric("Pixel AUROC", f"{macro.get('pixel_auroc', 0.0):.4f}")
        m3.metric("Pixel AUPIMO", f"{macro.get('pixel_aupimo', 0.0):.4f}")

        st.subheader("Category Breakdown")
        cat_data = []
        for cat_name, cat_res in results.get("categories", {}).items():
            img_m = cat_res.get("image_level", {})
            pix_m = cat_res.get("pixel_level", {})
            cat_data.append(
                {
                    "Category": cat_name,
                    "Image AUROC": img_m.get("auroc", 0.0),
                    "Pixel AUROC": pix_m.get("auroc", 0.0),
                    "Pixel AUPIMO": pix_m.get("aupimo", 0.0),
                }
            )
        st.dataframe(cat_data, width="stretch", hide_index=True)
    elif isinstance(results, dict):
        _render_evaluation_summary(results, model_type="patchcore")
        pixel_metrics = results.get("pixel_level", {})
        if metrics_path := pixel_metrics.get("metrics_path"):
            if Path(metrics_path).is_file():
                render_evaluation_curves(metrics_path)
        _render_heatmap_explorer(results)
    else:
        st.text_area("Results", value=str(results), height=180)
