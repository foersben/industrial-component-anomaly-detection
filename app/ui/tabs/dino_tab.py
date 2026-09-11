"""DINO vision foundation model evaluation tab for Streamlit UI."""

from pathlib import Path
from typing import Any, Literal, cast

import streamlit as st

from app.domain.categories import discover_dataset_categories
from app.pipelines.evaluation.visualization import render_evaluation_curves
from app.pipelines.modelling.dino import run_dinov2_baseline, run_dinov3_baseline
from app.ui.components.heatmaps import _render_heatmap_explorer
from app.ui.components.metrics import _render_evaluation_summary


def _render_dino_config_controls(architecture: str) -> tuple[int, str, Literal["off", "on", "published"]]:
    """Render slider and dropdown controls for DINO model hyperparameters.

    Args:
        architecture: Selected vision transformer model family ('DINOv2' or 'DINOv3').

    Returns:
        Tuple of (num_neighbors, variant, masking_policy).
    """
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

    return num_neighbors, variant, masking


def _execute_dino_run(
    architecture: str,
    data_root: str,
    category: str,
    num_neighbors: int,
    masking: Literal["off", "on", "published"],
    run_heatmap: bool,
    variant: str,
) -> dict[str, Any] | None:
    """Execute the selected DINO pipeline under a loading spinner.

    Args:
        architecture: Model architecture ('DINOv2' or 'DINOv3').
        data_root: Root dataset directory path.
        category: Component category name or 'all'.
        num_neighbors: Number of patch nearest neighbors.
        masking: Foreground masking policy.
        run_heatmap: Whether to compute heatmap overlays.
        variant: Scorer variant for DINOv2.

    Returns:
        Evaluation results dictionary if successful, None otherwise.
    """
    with st.spinner(f"Running {architecture} nearest-neighbor evaluation on '{category}'..."):
        try:
            if architecture == "DINOv2":
                results = run_dinov2_baseline(
                    data_root=Path(data_root),
                    category=category,
                    num_neighbors=num_neighbors,
                    masking=masking,
                    run_heatmap=run_heatmap,
                    variant=cast("Literal['baseline', 'enhanced']", variant),
                )
            else:
                results = run_dinov3_baseline(
                    data_root=Path(data_root),
                    category=category,
                    num_neighbors=num_neighbors,
                    masking=masking,
                    run_heatmap=run_heatmap,
                )
            st.success(f"{architecture} baseline execution finished.")
            return cast("dict[str, Any]", results)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            return None


def _render_dino_all_categories_summary(results_dict: dict[str, Any]) -> None:
    """Render macro averages and category breakdown table for multi-category runs.

    Args:
        results_dict: Multi-category evaluation results dictionary.
    """
    st.subheader("Macro Averages Across All Categories")
    macro = results_dict.get("macro_average", {})
    m1, m2, m3 = st.columns(3)
    m1.metric("Image AUROC", f"{macro.get('image_auroc', 0.0):.4f}")
    m2.metric("Pixel AUROC", f"{macro.get('pixel_auroc', 0.0):.4f}")
    m3.metric("Pixel AUPIMO", f"{macro.get('pixel_aupimo', 0.0):.4f}")

    st.subheader("Category Breakdown")
    cat_data = []
    for cat_name, cat_res in results_dict.get("categories", {}).items():
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


def _render_dino_results(results_dict: dict[str, Any], category: str) -> None:
    """Render metrics cards, curves, and heatmaps for completed DINO evaluation.

    Args:
        results_dict: Evaluation output dictionary.
        category: Selected component category name or 'all'.
    """
    if category == "all" and "macro_average" in results_dict:
        _render_dino_all_categories_summary(results_dict)
    elif isinstance(results_dict, dict):
        _render_evaluation_summary(results_dict, model_type="patchcore")
        pixel_metrics = results_dict.get("pixel_level", {})
        if metrics_path := pixel_metrics.get("metrics_path"):
            if Path(metrics_path).is_file():
                render_evaluation_curves(metrics_path)
        _render_heatmap_explorer(results_dict)
    else:
        st.text_area("Results", value=str(results_dict), height=180)


def render_dino_tab() -> None:
    """Render the DINO vision transformer nearest-neighbor baselines tab (DINOv2 & DINOv3)."""
    st.header("DINO Vision Transformer Baselines")
    st.markdown(
        "Evaluate frozen **DINOv2** and **DINOv3** patch nearest-neighbor baselines on MVTec AD. "
        "Supports stock and enhanced multi-layer position/density-aware scorers."
    )

    col_root, col_cat = st.columns(2)
    st.session_state.setdefault("dino_root", "data/raw/mvtec_ad")
    data_root = col_root.text_input("Dataset Root Directory", key="dino_root")
    categories = [*discover_dataset_categories(data_root), "all"]
    category = col_cat.selectbox("Category", options=categories, key="dino_cat")

    st.subheader("Model Configuration")
    architecture = st.radio("Architecture", ["DINOv2", "DINOv3"], horizontal=True, key="dino_arch")
    num_neighbors, variant, masking = _render_dino_config_controls(architecture)

    st.subheader("Evaluation Settings")
    run_heatmap = st.checkbox("Generate Anomaly Prediction Heatmaps", value=False, key="dino_heatmap")

    if not st.button(f"Run {architecture} Baseline Evaluation", type="primary", key="btn_run_dino"):
        return

    results = _execute_dino_run(
        architecture=architecture,
        data_root=data_root,
        category=category,
        num_neighbors=num_neighbors,
        masking=masking,
        run_heatmap=run_heatmap,
        variant=variant,
    )
    if results is not None:
        _render_dino_results(results, category)
