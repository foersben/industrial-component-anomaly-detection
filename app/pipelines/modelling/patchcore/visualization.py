"""Visualization and visual evaluation utilities for the PatchCore pipeline."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from anomalib.utils.path import generate_output_filename
from anomalib.visualization import ImageVisualizer
from anomalib.visualization.image.item_visualizer import visualize_image_item
from PIL import Image, ImageDraw, ImageFont

from app.core.logger import logger


def _add_panel_headers(
    grid: Image.Image,
    labels: list[str],
    panel_width: int = 256,
    header_height: int = 34,
) -> Image.Image:
    """Place unobstructed labels in a header above a horizontal image grid.

    Args:
        grid: Horizontal grid image containing concatenated visualization panels.
        labels: Text labels for each panel column.
        panel_width: Width in pixels of each panel column.
        header_height: Height in pixels of the newly created top header.

    Returns:
        New image with a clean top header strip containing centered panel labels.
    """
    labelled = Image.new("RGB", (grid.width, grid.height + header_height), "white")
    labelled.paste(grid.convert("RGB"), (0, header_height))
    draw = ImageDraw.Draw(labelled)
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for index, label in enumerate(labels):
        left = index * panel_width
        if left >= grid.width:
            break
        right = min(left + panel_width, grid.width)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        x = left + max(0, (right - left - text_width) // 2)
        y = max(0, (header_height - text_height) // 2 - text_box[1])
        draw.text((x, y), label, fill="black", font=font)
    return labelled


class _RawScoreImageVisualizer(ImageVisualizer):
    """Render raw continuous anomaly maps with optional calibrated masks and unobstructed headers."""

    def __init__(
        self,
        pixel_threshold: float | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """Configure four-panel rendering without drawing labels over image data.

        Args:
            pixel_threshold: Optional calibrated threshold for binarizing predicted masks.
            output_dir: Directory where generated four-panel visualizer images are saved.
        """
        super().__init__(text_config={"enable": False}, output_dir=output_dir)
        self.pixel_threshold = pixel_threshold
        self.render_enabled = True

    def on_test_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Render raw anomaly maps without mutating the batch used for evaluation."""
        del pl_module, outputs, batch_idx, dataloader_idx
        if not self.render_enabled:
            return

        anomaly_map = getattr(batch, "anomaly_map", None)
        if not isinstance(anomaly_map, torch.Tensor) or anomaly_map.ndim < 2:
            return

        flattened = anomaly_map.reshape(anomaly_map.shape[0], -1)
        minimum = flattened.min(dim=1).values
        maximum = flattened.max(dim=1).values
        view_shape = (anomaly_map.shape[0],) + (1,) * (anomaly_map.ndim - 1)
        minimum = minimum.reshape(view_shape)
        score_range = (maximum - flattened.min(dim=1).values).reshape(view_shape)
        normalized_map = torch.where(
            score_range > torch.finfo(anomaly_map.dtype).eps,
            (anomaly_map - minimum) / score_range,
            torch.zeros_like(anomaly_map),
        )
        updates: dict[str, Any] = {"anomaly_map": normalized_map}
        if self.pixel_threshold is not None:
            updates["pred_mask"] = anomaly_map > self.pixel_threshold
        visualization_batch = batch.update(in_place=False, **updates)

        if self.output_dir is None:
            self.output_dir = Path(trainer.default_root_dir) / "images"

        labels = ["Image", "Ground-Truth Mask", "Anomaly Map Overlay", "Predicted Mask"]
        for item in visualization_batch:
            image = visualize_image_item(
                item,
                fields=self.fields,
                overlay_fields=self.overlay_fields,
                field_size=self.field_size,
                fields_config=self.fields_config,
                overlay_fields_config=self.overlay_fields_config,
                text_config={"enable": False},
            )
            if image is None:
                continue
            image = _add_panel_headers(image, labels, panel_width=self.field_size[0])
            datamodule = getattr(trainer, "datamodule", None)
            dataset_name = getattr(datamodule, "name", None) if datamodule else None
            category = getattr(datamodule, "category", None) if datamodule else None
            filename = generate_output_filename(
                input_path=item.image_path or "",
                output_path=self.output_dir,
                dataset_name=dataset_name,
                category=category,
            )
            image.save(filename)


def _print_patchcore_results_table(results: Mapping[str, float]) -> None:
    """Print corrected PatchCore metrics in Anomalib's table layout."""
    from rich.console import Console
    from rich.table import Table

    table = Table(show_header=True, header_style="bold")
    table.add_column("Test metric")
    table.add_column("DataLoader 0", justify="right")
    for name, value in results.items():
        table.add_row(name, f"{value:.6f}")
    Console().print(table)


def _save_heatmap_overlays(
    overlays: dict[int, dict[str, list[Any]]],
    output_path: Path,
) -> Path | None:
    """Store dense heatmap images in a compressed binary archive instead of JSON.

    Serializes RGB heatmap overlays and ground-truth boundary comparison arrays
    into a compressed `.npz` container for lightweight disk persistence
    and fast UI visualization.

    Args:
        overlays: Dictionary mapping test sample indices to overlay visualization maps.
        output_path: Filesystem destination path to write the compressed `.npz` archive.

    Returns:
        Path to output file if successfully saved, or None if the input overlay map was empty.
    """
    if not overlays:
        return None

    arrays: dict[str, np.ndarray[Any, Any]] = {}
    for index, overlay in overlays.items():
        if "heatmap" in overlay:
            arrays[f"prediction__{index}"] = np.asarray(overlay["heatmap"], dtype=np.uint8)
        if "gt_and_heatmap" in overlay:
            arrays[f"ground_truth__{index}"] = np.asarray(overlay["gt_and_heatmap"], dtype=np.uint8)

    if not arrays:
        return None

    np.savez_compressed(output_path, **arrays)  # type: ignore[arg-type]
    return output_path


def _load_heatmap_overlays(input_path: Path) -> dict[int, dict[str, list[Any]]]:
    """Load heatmap images from a compressed archive into the standard UI schema.

    Deserializes binary arrays keyed by `prediction__{idx}` and `ground_truth__{idx}`
    back into a structured nested dictionary suitable for JSON transport and Streamlit rendering.

    Args:
        input_path: Filesystem path to the existing `.npz` overlay archive.

    Returns:
        Dictionary mapping integer sample indices to heatmaps and ground truth overlays.
    """
    overlays: dict[int, dict[str, list[Any]]] = {}
    with np.load(input_path, allow_pickle=False) as archive:
        for key in archive.files:
            kind, separator, raw_index = key.partition("__")
            if not separator or kind not in {"prediction", "ground_truth"}:
                continue
            try:
                index = int(raw_index)
            except ValueError:
                continue
            field = "heatmap" if kind == "prediction" else "gt_and_heatmap"
            overlays.setdefault(index, {})[field] = archive[key].astype(np.uint8, copy=False).tolist()
    return overlays


def _normalize_overlay_image(img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Convert tensor-like image arrays into HWC uint8 format for rendering.

    Transposes CHW dimensions if necessary and rescales float inputs in [0, 1] to
    the standard [0, 255] uint8 color space.

    Args:
        img: Input image array (can be CHW or HWC, float or uint8).

    Returns:
        HWC-formatted NumPy array with uint8 dtype.
    """
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    if img.dtype != np.uint8:
        scale_val = 255.0 if float(img.max()) <= 1.0 else 1.0
        return (img * scale_val).astype(np.uint8)
    return img


def _normalize_percentile_amap(amap: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Normalize continuous anomaly map between 1st and 99th percentiles into [0, 1].

    Applies percentile clipping to suppress extreme outlier responses and produce
    visually informative colormap heatmaps.

    Args:
        amap: Raw continuous anomaly map array.

    Returns:
        Normalized 2D anomaly map clipped strictly to [0.0, 1.0].
    """
    amap_sq = amap.squeeze()
    p_low = float(np.percentile(amap_sq, 1))
    p_high = float(np.percentile(amap_sq, 99))
    if abs(p_high - p_low) > 1e-8:
        return np.clip((amap_sq - p_low) / (p_high - p_low), 0.0, 1.0)
    return np.zeros_like(amap_sq)


def _create_sample_heatmap_overlay(
    img: np.ndarray[Any, Any],
    amap: np.ndarray[Any, Any],
    gt_mask_arr: Any,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Create resized heatmap overlay and ground-truth comparison for one sample.

    Generates alpha-blended colormapped anomaly heatmaps on top of the original image
    and overlays ground-truth defect contours for direct visual verification.

    Args:
        img: Raw input image array.
        amap: Continuous anomaly heatmap array.
        gt_mask_arr: Optional ground-truth defect segmentation mask.

    Returns:
        Tuple of (hm_overlay_small, gt_and_heatmap_small) resized to fit canonical UI display.
    """
    import cv2

    from app.pipelines.evaluation.heatmaps import overlay_ground_truth, overlay_heatmap

    orig_img = _normalize_overlay_image(img)
    amap_norm = _normalize_percentile_amap(amap)
    hm_overlay = overlay_heatmap(orig_img, amap_norm.astype(np.float32), alpha=0.4)

    gt_mask_img = None
    if gt_mask_arr is not None:
        gt_mask_img = gt_mask_arr.squeeze() if hasattr(gt_mask_arr, "squeeze") else gt_mask_arr
        if gt_mask_img.shape[:2] != orig_img.shape[:2]:
            gt_mask_img = cv2.resize(
                gt_mask_img.astype(np.float32),
                (orig_img.shape[1], orig_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

    gt_and_heatmap = overlay_ground_truth(hm_overlay, gt_mask_img)

    max_dim = 256
    if max(hm_overlay.shape[0], hm_overlay.shape[1]) > max_dim:
        scale_factor = max_dim / max(hm_overlay.shape[0], hm_overlay.shape[1])
        new_size = (int(hm_overlay.shape[1] * scale_factor), int(hm_overlay.shape[0] * scale_factor))
        hm_overlay_small = cv2.resize(hm_overlay, new_size, interpolation=cv2.INTER_AREA)
        gt_and_heatmap_small = cv2.resize(gt_and_heatmap, new_size, interpolation=cv2.INTER_AREA)
    else:
        hm_overlay_small = hm_overlay
        gt_and_heatmap_small = gt_and_heatmap

    return hm_overlay_small, gt_and_heatmap_small


def _process_batch_heatmaps(
    batch: Any,
    start_idx: int,
    heatmap_overlays: dict[int, dict[str, list[Any]]],
    anomalous_indices: list[int],
) -> int:
    """Process a single prediction batch and record overlays for anomalous samples.

    Extracts images, anomaly heatmaps, masks, and binary ground-truth labels from
    the Anomalib prediction batch, generating visualization overlays exclusively for
    confirmed defective samples (label == 1).

    Args:
        batch: Anomalib prediction batch namespace containing prediction tensors.
        start_idx: Current global sample index across all processed batches.
        heatmap_overlays: Output dictionary accumulator storing visualization payloads.
        anomalous_indices: Output list accumulator storing indices of anomalous samples.

    Returns:
        Updated global sample index after processing all batch elements.
    """
    images_t = getattr(batch, "image", None)
    anomaly_maps_t = getattr(batch, "anomaly_map", None)
    gt_masks_t = getattr(batch, "gt_mask", None)
    gt_labels_t = getattr(batch, "gt_label", None)

    if images_t is None or anomaly_maps_t is None or gt_labels_t is None:
        return start_idx

    images_np = images_t.detach().cpu().numpy()
    anomaly_maps_np = anomaly_maps_t.detach().cpu().numpy()
    gt_masks_np = gt_masks_t.detach().cpu().numpy() if gt_masks_t is not None else None
    gt_labels_np = gt_labels_t.detach().cpu().numpy()

    idx = start_idx
    for i, label in enumerate(gt_labels_np):
        if int(label) == 1:
            mask_i = gt_masks_np[i] if gt_masks_np is not None else None
            hm_small, gt_hm_small = _create_sample_heatmap_overlay(images_np[i], anomaly_maps_np[i], mask_i)
            anomalous_indices.append(idx)
            heatmap_overlays[idx] = {
                "heatmap": hm_small.tolist(),
                "gt_and_heatmap": gt_hm_small.tolist(),
            }
        idx += 1
    return idx


def _generate_test_heatmaps(
    predictions: list[Any],
) -> tuple[dict[int, dict[str, list[Any]]], list[int]]:
    """Iterate through predictions and generate visual heatmaps for anomalous test items.

    Coordinates batch iteration, defensive exception trapping, and sample indexing
    to assemble a serializable collection of heatmap overlays for downstream review.

    Args:
        predictions: Sequence of prediction batch items returned by Anomalib Engine.predict.

    Returns:
        Tuple of (heatmap_overlays, anomalous_indices) ready for JSON/NPZ persistence.
    """
    heatmap_overlays: dict[int, dict[str, list[Any]]] = {}
    anomalous_indices: list[int] = []

    try:
        global_idx = 0
        for batch in predictions:
            global_idx = _process_batch_heatmaps(batch, global_idx, heatmap_overlays, anomalous_indices)
    except Exception as e:
        logger.warning("Failed to compute heatmap overlays: %s", e)

    return heatmap_overlays, anomalous_indices
