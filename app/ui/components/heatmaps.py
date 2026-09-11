"""Heatmap visualization components for Streamlit UI."""

from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st


def _render_prediction_heatmap_grid(
    indices: list[int],
    overlays: dict[Any, Any],
    cols_per_row: int = 2,
) -> None:
    """Render a grid of prediction heatmaps for selected image indices.

    Args:
        indices: Image indices to render.
        overlays: Heatmap overlays dictionary.
        cols_per_row: Number of grid columns per row.
    """
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


def _extract_gt_overlay_array(overlay_data: Any) -> np.ndarray | None:
    """Extract ground truth overlay image array from overlay dictionary or list.

    Args:
        overlay_data: Dictionary or list containing ground truth overlay image data.

    Returns:
        NumPy array representation of the overlay image, or None if missing.
    """
    if not isinstance(overlay_data, dict):
        return None
    if "gt_and_heatmap" in overlay_data:
        return np.array(overlay_data["gt_and_heatmap"], dtype=np.uint8)
    if "gt_overlay" in overlay_data:
        return np.array(overlay_data["gt_overlay"], dtype=np.uint8)
    return None


def _render_gt_heatmap_grid(
    indices: list[int],
    overlays: dict[Any, Any],
    cols_per_row: int = 2,
) -> None:
    """Render a grid of ground truth and prediction overlay heatmaps.

    Args:
        indices: Image indices to render.
        overlays: Heatmap overlays dictionary.
        cols_per_row: Number of grid columns per row.
    """
    st.subheader("Ground Truth + Heatmap")
    for row_start in range(0, len(indices), cols_per_row):
        row_indices = indices[row_start : row_start + cols_per_row]
        cols = st.columns(len(row_indices))
        for col, idx in zip(cols, row_indices, strict=False):
            overlay_data = overlays.get(idx) or overlays.get(str(idx))
            gt_hm_arr = _extract_gt_overlay_array(overlay_data)
            if gt_hm_arr is not None:
                col.image(gt_hm_arr, caption=f"Image #{idx}", width="stretch")


def _render_heatmap_gallery(overlays: dict[Any, Any], anomalous_indices: list[int]) -> None:
    """Render a grid of Error heatmap overlays for all anomalous images.

    Displays two rectangular areas (grids) side-by-side:
    - Left Grid: Original image + Model Prediction Heatmap
    - Right Grid: Ground Truth Mask + Model Prediction Heatmap

    Args:
        overlays: Dictionary mapping image index to overlay dictionaries.
        anomalous_indices: List of detected anomalous image indices.
    """
    indices = [i for i in anomalous_indices if str(i) in overlays or i in overlays]
    if not indices:
        return

    main_col1, main_col2 = st.columns(2)
    with main_col1:
        _render_prediction_heatmap_grid(indices, overlays)
    with main_col2:
        _render_gt_heatmap_grid(indices, overlays)

    st.caption(
        "Heatmaps derived directly from the per-pixel Mean Squared Error between the original "
        "image and the autoencoder's reconstruction, slightly smoothed with a Gaussian filter."
    )


def _find_four_panel_images(results: dict[str, Any]) -> list[Path]:
    """Find four-panel images linked to an evaluation result."""
    metadata = results.get("metadata", {})
    raw_path = metadata.get("four_panel_images_path") if isinstance(metadata, dict) else None
    if not raw_path:
        return []
    panel_dir = Path(str(raw_path))
    if not panel_dir.is_absolute():
        metrics_path = results.get("pixel_level", {}).get("metrics_path")
        if metrics_path:
            panel_dir = Path(str(metrics_path)).parent / panel_dir
    if not panel_dir.is_dir():
        return []
    return sorted(
        path
        for path in panel_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"} and "good" not in path.parts
    )


def _resolve_cached_heatmap_archive(results: dict[str, Any]) -> Path | None:
    """Resolve a cached heatmap archive without reading its image arrays."""
    metadata = results.get("metadata", {})
    raw_path = metadata.get("heatmap_overlays_path") if isinstance(metadata, dict) else None
    metrics_path = results.get("pixel_level", {}).get("metrics_path")
    if not raw_path or not metrics_path:
        return None
    archive_path = Path(str(raw_path))
    if not archive_path.is_absolute():
        archive_path = Path(str(metrics_path)).parent / archive_path.name
    return archive_path if archive_path.is_file() else None


def _cached_heatmap_indices(archive_path: Path) -> list[int]:
    """Read only the entry names from a compressed heatmap archive."""
    with np.load(archive_path, allow_pickle=False) as archive:
        indices = {
            int(raw_index)
            for key in archive.files
            if (parts := key.partition("__"))[1] and (raw_index := parts[2]).isdigit()
        }
    return sorted(indices)


def _load_cached_heatmap_page(archive_path: Path, indices: list[int]) -> dict[int, dict[str, np.ndarray]]:
    """Deserialize only the heatmap arrays visible on the current page."""
    overlays: dict[int, dict[str, np.ndarray]] = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        for index in indices:
            prediction_key = f"prediction__{index}"
            ground_truth_key = f"ground_truth__{index}"
            if prediction_key in archive:
                overlays.setdefault(index, {})["heatmap"] = archive[prediction_key]
            if ground_truth_key in archive:
                overlays.setdefault(index, {})["gt_and_heatmap"] = archive[ground_truth_key]
    return overlays


def _select_gallery_page(total_items: int, model_hash: str, view: str, page_size: int = 6) -> slice:
    """Render a compact page selector and return the visible item slice."""
    page_count = max(1, (total_items + page_size - 1) // page_size)
    if page_count == 1:
        return slice(0, page_size)
    page = st.number_input(
        "Page",
        min_value=1,
        max_value=page_count,
        value=1,
        step=1,
        key=f"visualization_page_{model_hash}_{view}",
    )
    start = (int(page) - 1) * page_size
    st.caption(f"Page {page} of {page_count} · {total_items} images")
    return slice(start, start + page_size)


def _render_four_panel_gallery(images: list[Path], cols_per_row: int = 2) -> None:
    """Render cached four-panel comparison images in a responsive grid."""
    st.subheader("Four-Panel Prediction Comparison")
    st.caption("Each image shows: input, ground-truth mask, anomaly-map overlay, and thresholded prediction mask.")
    for row_start in range(0, len(images), cols_per_row):
        row_images = images[row_start : row_start + cols_per_row]
        columns = st.columns(len(row_images))
        for column, image_path in zip(columns, row_images, strict=False):
            column.image(str(image_path), caption=image_path.stem, width="stretch")


def _render_heatmap_explorer(results: dict[str, Any]) -> None:
    """Render heatmap and four-panel visualizations below evaluation results.

    Shows a gallery of error heatmap overlays for every detected anomalous image.
    The user can trigger computation by clicking a single button; results are
    shown in a responsive grid so all anomalous images are visible at once.
    """
    anomalous_indices: list[int] = results.get("anomalous_indices", [])
    four_panel_images = _find_four_panel_images(results)
    existing_overlays: dict[Any, Any] = results.get("heatmap_overlays", {})
    heatmap_archive = _resolve_cached_heatmap_archive(results) if not existing_overlays else None
    cached_indices = _cached_heatmap_indices(heatmap_archive) if heatmap_archive else []
    heatmap_indices = anomalous_indices or cached_indices
    if not heatmap_indices and not four_panel_images:
        return

    st.divider()
    with st.expander(
        f"Anomaly Visualization Explorer — {max(len(anomalous_indices), len(four_panel_images))} anomalous image(s)",
        expanded=True,
    ):
        st.markdown("""
        The model processes test images and generates a pixel-wise **Anomaly Map**. High values indicate that the model
        believes those specific pixels are defective based on what it learned from normal components.

        **Red / warm** — high anomaly score → likely a defect
        **Blue / cool** — low anomaly score → looks normal to the model
        """)

        view = "Heatmap Images"
        if (existing_overlays or heatmap_archive) and four_panel_images:
            view = st.radio(
                "Visualization",
                ["Heatmap Images", "Four-Panel Images"],
                horizontal=True,
                key=f"visualization_view_{results.get('model_hash', 'current')}",
            )
        elif existing_overlays or heatmap_archive:
            st.caption(
                "Four-panel images are not linked to this legacy cache entry. They will be available for newly "
                "created caches."
            )

        if view == "Four-Panel Images":
            page_slice = _select_gallery_page(
                len(four_panel_images), str(results.get("model_hash", "current")), "four_panel"
            )
            _render_four_panel_gallery(four_panel_images[page_slice])
        elif existing_overlays or heatmap_archive:
            page_slice = _select_gallery_page(
                len(heatmap_indices), str(results.get("model_hash", "current")), "heatmap"
            )
            visible_indices = heatmap_indices[page_slice]
            visible_overlays = (
                _load_cached_heatmap_page(heatmap_archive, visible_indices) if heatmap_archive else existing_overlays
            )
            st.success(f"Heatmaps available for {len(heatmap_indices)} image(s).")
            _render_heatmap_gallery(visible_overlays, visible_indices)
        elif four_panel_images:
            page_slice = _select_gallery_page(
                len(four_panel_images), str(results.get("model_hash", "current")), "four_panel"
            )
            _render_four_panel_gallery(four_panel_images[page_slice])
        else:
            st.info("No heatmap or four-panel visualizations were computed.")
