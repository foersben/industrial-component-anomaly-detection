"""Heatmap visualization components for Streamlit UI."""

from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

# ==========================================
# CELL & GRID RENDERERS
# ==========================================


def _get_overlay_item(overlays: dict[Any, Any], idx: int) -> Any:
    """Retrieve overlay data by integer or string key.

    Args:
        overlays: Heatmap overlays dictionary mapping image index to heatmap data.
        idx: Image index to retrieve.

    Returns:
        Overlay data for the specified image index.
    """
    return overlays.get(idx) if idx in overlays else overlays.get(str(idx))


def _extract_gt_overlay_array(overlay_data: Any) -> np.ndarray | None:
    """Extract ground truth overlay image array from overlay dictionary or list.

    Args:
        overlay_data: Overlay data structure potentially containing ground truth.

    Returns:
        NumPy array of the ground truth overlay if found, else None.
    """
    if not isinstance(overlay_data, dict):
        return None
    if "gt_and_heatmap" in overlay_data:
        return np.array(overlay_data["gt_and_heatmap"], dtype=np.uint8)
    if "gt_overlay" in overlay_data:
        return np.array(overlay_data["gt_overlay"], dtype=np.uint8)
    return None


def _render_prediction_cell(col: Any, idx: int, overlay_data: Any) -> None:
    """Render a single prediction heatmap cell.

    Args:
        col: Streamlit column to render the heatmap in.
        idx: Image index.
        overlay_data: Overlay data dictionary containing heatmap array.
    """
    if isinstance(overlay_data, dict) and "heatmap" in overlay_data:
        hm_arr = np.array(overlay_data["heatmap"], dtype=np.uint8)
        col.image(hm_arr, caption=f"Image #{idx}", width="stretch")
        return
    if isinstance(overlay_data, list):
        col.image(np.array(overlay_data, dtype=np.uint8), caption=f"Image #{idx}")


def _render_gt_cell(col: Any, idx: int, overlay_data: Any) -> None:
    """Render a single ground truth heatmap cell.

    Args:
        col: Streamlit column to render the heatmap in.
        idx: Image index.
        overlay_data: Overlay data structure potentially containing ground truth.
    """
    gt_hm_arr = _extract_gt_overlay_array(overlay_data)
    if gt_hm_arr is not None:
        col.image(gt_hm_arr, caption=f"Image #{idx}", width="stretch")


def _render_prediction_heatmap_grid(
    indices: list[int],
    overlays: dict[Any, Any],
    cols_per_row: int = 2,
) -> None:
    """Render a grid of prediction heatmaps for selected image indices.

    Args:
        indices: Image indices to render.
        overlays: Heatmap overlays dictionary mapping image index to heatmap data.
        cols_per_row: Number of grid columns per row.
    """
    st.subheader("Prediction Heatmap")
    for row_start in range(0, len(indices), cols_per_row):
        row_indices = indices[row_start : row_start + cols_per_row]
        cols = st.columns(len(row_indices))
        for col, idx in zip(cols, row_indices, strict=False):
            overlay_data = _get_overlay_item(overlays, idx)
            _render_prediction_cell(col, idx, overlay_data)


def _render_gt_heatmap_grid(
    indices: list[int],
    overlays: dict[Any, Any],
    cols_per_row: int = 2,
) -> None:
    """Render a grid of ground truth and prediction overlay heatmaps.

    Args:
        indices: Image indices to render.
        overlays: Heatmap overlays dictionary containing ground truth overlays.
        cols_per_row: Number of columns in the grid layout.
    """
    st.subheader("Ground Truth + Heatmap")
    for row_start in range(0, len(indices), cols_per_row):
        row_indices = indices[row_start : row_start + cols_per_row]
        cols = st.columns(len(row_indices))
        for col, idx in zip(cols, row_indices, strict=False):
            overlay_data = _get_overlay_item(overlays, idx)
            _render_gt_cell(col, idx, overlay_data)


def _render_heatmap_gallery(overlays: dict[Any, Any], anomalous_indices: list[int]) -> None:
    """Render side-by-side grids of error heatmaps for anomalous images.

    Args:
        overlays: Heatmap overlays dictionary mapping image index to heatmap data.
        anomalous_indices: List of detected anomalous image indices.
    """
    indices = [i for i in anomalous_indices if i in overlays or str(i) in overlays]
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


def _render_four_panel_gallery(images: list[Path], cols_per_row: int = 2) -> None:
    """Render cached four-panel comparison images in a responsive grid.

    Args:
        images: List of four-panel image paths to render.
        cols_per_row: Number of columns in the grid layout.
    """
    st.subheader("Four-Panel Prediction Comparison")
    st.caption("Each image shows: input, ground-truth mask, anomaly-map overlay, and thresholded prediction mask.")
    for row_start in range(0, len(images), cols_per_row):
        row_images = images[row_start : row_start + cols_per_row]
        columns = st.columns(len(row_images))
        for column, image_path in zip(columns, row_images, strict=False):
            column.image(str(image_path), caption=image_path.stem, width="stretch")


# ==========================================
# FILE SYSTEM & IO CACHING RESOLVERS
# ==========================================


def _is_valid_panel_image(path: Path) -> bool:
    """Validate whether a path points to a non-good anomalous panel image.

    Args:
        path: Path to the image file.

    Returns:
        True if the path points to a valid panel image, False otherwise.
    """
    if not path.is_file():
        return False
    if "good" in path.parts:
        return False
    return path.suffix.lower() in {".png", ".jpg", ".jpeg"}


def _resolve_panel_dir(results: dict[str, Any]) -> Path | None:
    """Resolve directory containing four-panel comparison images.

    Args:
        results: Evaluation results dictionary.

    Returns:
        Path to the directory containing four-panel images.
    """
    metadata = results.get("metadata")
    if not isinstance(metadata, dict):
        return None

    raw_path = metadata.get("four_panel_images_path")
    if not raw_path:
        return None

    panel_dir = Path(str(raw_path))
    if panel_dir.is_absolute():
        return panel_dir if panel_dir.is_dir() else None

    metrics_path = results.get("pixel_level", {}).get("metrics_path")
    if not metrics_path:
        return None

    resolved_dir = Path(str(metrics_path)).parent / panel_dir
    return resolved_dir if resolved_dir.is_dir() else None


def _find_four_panel_images(results: dict[str, Any]) -> list[Path]:
    """Find four-panel images linked to an evaluation result.

    Args:
        results: Evaluation results dictionary.

    Returns:
        List of four-panel image paths.
    """
    panel_dir = _resolve_panel_dir(results)
    if panel_dir is None:
        return []
    return sorted(path for path in panel_dir.rglob("*") if _is_valid_panel_image(path))


def _resolve_cached_heatmap_archive(results: dict[str, Any]) -> Path | None:
    """Resolve a cached heatmap archive without reading its image arrays.

    Args:
        results: Evaluation results dictionary.

    Returns:
        Path to the cached heatmap archive if found, else None.
    """
    metadata = results.get("metadata")
    if not isinstance(metadata, dict):
        return None

    raw_path = metadata.get("heatmap_overlays_path")
    metrics_path = results.get("pixel_level", {}).get("metrics_path")
    if not raw_path or not metrics_path:
        return None

    archive_path = Path(str(raw_path))
    if not archive_path.is_absolute():
        archive_path = Path(str(metrics_path)).parent / archive_path.name
    return archive_path if archive_path.is_file() else None


def _extract_archive_index(key: str) -> int | None:
    """Extract numeric image index from archive key.

    Args:
        key: Archive key to extract index from.

    Returns:
        Image index as integer if found, else None.
    """
    _, sep, raw_index = key.partition("__")
    if sep and raw_index.isdigit():
        return int(raw_index)
    return None


def _cached_heatmap_indices(archive_path: Path) -> list[int]:
    """Read only the entry names from a compressed heatmap archive.

    Args:
        archive_path: Path to the cached heatmap archive.

    Returns:
        List of heatmap indices.
    """
    with np.load(archive_path, allow_pickle=False) as archive:
        indices = {idx for key in archive.files if (idx := _extract_archive_index(key)) is not None}
    return sorted(indices)


def _load_cached_heatmap_page(archive_path: Path, indices: list[int]) -> dict[int, dict[str, np.ndarray]]:
    """Deserialize only the heatmap arrays visible on the current page.

    Args:
        archive_path: Path to the cached heatmap archive.
        indices: List of heatmap indices to load.

    Returns:
        Dictionary of heatmap data.
    """
    overlays: dict[int, dict[str, np.ndarray]] = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        for index in indices:
            pred_key = f"prediction__{index}"
            gt_key = f"ground_truth__{index}"
            if pred_key in archive:
                overlays.setdefault(index, {})["heatmap"] = archive[pred_key]
            if gt_key in archive:
                overlays.setdefault(index, {})["gt_and_heatmap"] = archive[gt_key]
    return overlays


def _select_gallery_page(total_items: int, model_hash: str, view: str, page_size: int = 6) -> slice:
    """Render a compact page selector and return the visible item slice.

    Args:
        total_items: Total number of items.
        model_hash: Model hash for unique widget key.
        view: View identifier for unique widget key.
        page_size: Number of items per page.

    Returns:
        Slice of items to display.
    """
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


def _display_four_panel_gallery(four_panel_images: list[Path], model_hash: str) -> None:
    """Handle layout slicing and presentation execution for Four-Panel view.

    Args:
        four_panel_images: List of paths to four-panel images.
        model_hash: Model hash for unique widget key.
    """
    page_slice = _select_gallery_page(len(four_panel_images), model_hash, "four_panel")
    _render_four_panel_gallery(four_panel_images[page_slice])


def _display_heatmap_gallery(
    heatmap_indices: list[int],
    heatmap_archive: Path | None,
    existing_overlays: dict[Any, Any],
    model_hash: str,
) -> None:
    """Handle target page loading arrays and rendering for Heatmap view.

    Args:
        heatmap_indices: List of heatmap indices.
        heatmap_archive: Path to the cached heatmap archive.
        existing_overlays: Existing heatmap overlays.
        model_hash: Model hash for unique widget key.
    """
    page_slice = _select_gallery_page(len(heatmap_indices), model_hash, "heatmap")
    visible_indices = heatmap_indices[page_slice]
    visible_overlays = (
        _load_cached_heatmap_page(heatmap_archive, visible_indices) if heatmap_archive else existing_overlays
    )
    st.success(f"Heatmaps available for {len(heatmap_indices)} image(s).")
    _render_heatmap_gallery(visible_overlays, visible_indices)


def _determine_visualization_view(has_heatmaps: bool, has_four_panel: bool, model_hash: str) -> str:
    """Isolate view selection into a clean linear condition tree.

    Args:
        has_heatmaps: Whether heatmaps are available.
        has_four_panel: Whether four-panel images are available.
        model_hash: Model hash for unique widget key.

    Returns:
        Visualization view selection as a string.
    """
    if has_heatmaps and has_four_panel:
        return str(
            st.radio(
                "Visualization",
                ["Heatmap Images", "Four-Panel Images"],
                horizontal=True,
                key=f"visualization_view_{model_hash}",
            )
        )
    if has_heatmaps:
        st.caption(
            "Four-panel images are not linked to this legacy cache entry. "
            "They will be available for newly created caches."
        )
        return "Heatmap Images"
    if has_four_panel:
        return "Four-Panel Images"
    return "None"


def _render_explorer_content(
    results: dict[str, Any],
    heatmap_indices: list[int],
    heatmap_archive: Path | None,
    existing_overlays: dict[Any, Any],
    four_panel_images: list[Path],
) -> None:
    """Render visualizer body content inside the expander.

    Args:
        results: Evaluation results dictionary.
        heatmap_indices: List of heatmap indices.
        heatmap_archive: Path to the cached heatmap archive.
        existing_overlays: Existing heatmap overlays.
        four_panel_images: List of paths to four-panel images.
    """
    st.markdown("""
    The model processes test images and generates a pixel-wise **Anomaly Map**. High values indicate that the model
    believes those specific pixels are defective based on what it learned from normal components.

    **Red / warm** — high anomaly score → likely a defect
    **Blue / cool** — low anomaly score → looks normal to the model
    """)

    model_hash = str(results.get("model_hash", "current"))
    has_heatmaps = bool(existing_overlays or heatmap_archive)
    has_four_panel = bool(four_panel_images)

    view = _determine_visualization_view(has_heatmaps, has_four_panel, model_hash)

    if view == "Four-Panel Images":
        _display_four_panel_gallery(four_panel_images, model_hash)
    elif view == "Heatmap Images":
        _display_heatmap_gallery(heatmap_indices, heatmap_archive, existing_overlays, model_hash)
    else:
        st.info("No heatmap or four-panel visualizations were computed.")


def _render_heatmap_explorer(results: dict[str, Any]) -> None:
    """Render heatmap and four-panel visualizations below evaluation results.

    Args:
        results: Evaluation results dictionary.
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
    max_images = max(len(anomalous_indices), len(four_panel_images))

    with st.expander(
        f"Anomaly Visualization Explorer — {max_images} anomalous image(s)",
        expanded=True,
    ):
        _render_explorer_content(
            results=results,
            heatmap_indices=heatmap_indices,
            heatmap_archive=heatmap_archive,
            existing_overlays=existing_overlays,
            four_panel_images=four_panel_images,
        )
