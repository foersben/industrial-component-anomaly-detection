"""Heatmap visualization components for Streamlit UI."""

from typing import Any

import numpy as np
import streamlit as st


def _render_heatmap_gallery(overlays: dict[Any, Any], anomalous_indices: list[int]) -> None:
    """Render a grid of Error heatmap overlays for all anomalous images.

    Displays two rectangular areas (grids) side-by-side:
    - Left Grid: Original image + Model Prediction Heatmap
    - Right Grid: Ground Truth Mask + Model Prediction Heatmap
    """
    st.success(f"Heatmaps computed for {len(overlays)} image(s).")

    indices = [i for i in anomalous_indices if str(i) in overlays or i in overlays]
    if not indices:
        return

    main_col1, main_col2 = st.columns(2)
    cols_per_row = 2

    with main_col1:
        st.subheader("Prediction Heatmap")
        for row_start in range(0, len(indices), cols_per_row):
            row_indices = indices[row_start : row_start + cols_per_row]
            cols = st.columns(len(row_indices))
            for col, idx in zip(cols, row_indices, strict=False):
                overlay_data = overlays.get(idx) or overlays.get(str(idx))
                if isinstance(overlay_data, dict) and "heatmap" in overlay_data:
                    hm_arr = np.array(overlay_data["heatmap"], dtype=np.uint8)
                    col.image(hm_arr, caption=f"Image #{idx}", width="stretch")
                elif isinstance(overlay_data, list):
                    col.image(np.array(overlay_data, dtype=np.uint8), caption=f"Image #{idx}")

    with main_col2:
        st.subheader("Ground Truth + Heatmap")
        for row_start in range(0, len(indices), cols_per_row):
            row_indices = indices[row_start : row_start + cols_per_row]
            cols = st.columns(len(row_indices))
            for col, idx in zip(cols, row_indices, strict=False):
                overlay_data = overlays.get(idx) or overlays.get(str(idx))
                if isinstance(overlay_data, dict):
                    if "gt_and_heatmap" in overlay_data:
                        gt_hm_arr = np.array(overlay_data["gt_and_heatmap"], dtype=np.uint8)
                        col.image(gt_hm_arr, caption=f"Image #{idx}", width="stretch")
                    elif "gt_overlay" in overlay_data:
                        gt_hm_arr = np.array(overlay_data["gt_overlay"], dtype=np.uint8)
                        col.image(gt_hm_arr, caption=f"Image #{idx}", width="stretch")

    st.caption(
        "Heatmaps derived directly from the per-pixel Mean Squared Error between the original "
        "image and the autoencoder's reconstruction, slightly smoothed with a Gaussian filter."
    )


def _render_heatmap_explorer(results: dict[str, Any]) -> None:
    """Render the Reconstruction Error Heatmap explorer below the evaluation results.

    Shows a gallery of error heatmap overlays for every detected anomalous image.
    The user can trigger computation by clicking a single button; results are
    shown in a responsive grid so all anomalous images are visible at once.
    """
    anomalous_indices: list[int] = results.get("anomalous_indices", [])
    if not anomalous_indices:
        return

    st.divider()
    with st.expander(
        f"Anomaly Heatmap Explorer — {len(anomalous_indices)} anomalous image(s) found",
        expanded=True,
    ):
        st.markdown("""
        The model processes test images and generates a pixel-wise **Anomaly Map**. High values indicate that the model
        believes those specific pixels are defective based on what it learned from normal components.

        **Red / warm** — high anomaly score → likely a defect
        **Blue / cool** — low anomaly score → looks normal to the model
        """)

        existing_overlays: dict[Any, Any] = results.get("heatmap_overlays", {})

        if existing_overlays:
            _render_heatmap_gallery(existing_overlays, anomalous_indices)
        else:
            st.info("No heatmaps were computed.")
