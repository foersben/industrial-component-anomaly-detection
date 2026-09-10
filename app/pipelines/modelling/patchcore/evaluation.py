"""Evaluation metrics extraction, heatmap overlay serialization, and result formatting for PatchCore."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from anomalib.engine import Engine
from sklearn.metrics import f1_score, roc_auc_score

from app.core.logger import logger
from app.pipelines.evaluation.metrics import (
    AUPIMO_FPR_BOUNDS,
    AUPIMO_NUM_THRESHOLDS,
    CANONICAL_MAP_SIZE,
    compute_and_save_pr_metrics,
    compute_image_confusion_metrics,
    compute_shared_pixel_metrics,
)
from app.pipelines.evaluation.scoring import compute_adaptive_threshold
from app.pipelines.modelling.patchcore.types import (
    PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
    PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
    BaselineResult,
)
from app.pipelines.modelling.patchcore.visualization import _RawScoreImageVisualizer


def _to_float(val: Any) -> float:
    """Safely convert a scalar, PyTorch tensor, or numeric value to a float.

    Args:
        val: Value to convert to float.

    Returns:
        Float representation of the value.
    """
    if hasattr(val, "item") and callable(val.item):
        try:
            return float(val.item())
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _save_heatmap_overlays(
    overlays: dict[int, dict[str, list[Any]]],
    output_path: Path,
) -> Path | None:
    """Store dense heatmap images in a compressed binary archive instead of JSON.

    Args:
        overlays: Dictionary mapping test indices to overlay dictionaries.
        output_path: Path to write the .npz archive to.

    Returns:
        Path to output file if saved, None if overlays was empty.
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
    """Load heatmap images from the compressed archive into the existing result schema.

    Args:
        input_path: Path to the .npz archive.

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


def _tensor_to_numpy(val: Any) -> np.ndarray[Any, Any] | None:
    """Safely extract a 1D NumPy array from a PyTorch tensor attribute.

    Args:
        val: Potential tensor object.

    Returns:
        1D float NumPy array if val is a tensor, else None.
    """
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy().reshape(-1)
    return None


def _collect_batch_tensors(
    batch: Any,
    score_attr: str,
    label_attr: str,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None:
    """Extract score and label arrays from a batch if both attributes exist and are tensors.

    Args:
        batch: Anomalib prediction batch object.
        score_attr: Attribute name containing anomaly scores or maps.
        label_attr: Attribute name containing ground truth labels or masks.

    Returns:
        Tuple of (score_array, label_array) or None if either attribute is missing.
    """
    scores = _tensor_to_numpy(getattr(batch, score_attr, None))
    labels = _tensor_to_numpy(getattr(batch, label_attr, None))

    if scores is not None and labels is not None:
        return scores, labels
    return None


def _process_and_save_level(
    scores_list: list[np.ndarray[Any, Any]],
    labels_list: list[np.ndarray[Any, Any]],
    output_path: Path,
    level: str,
    aupimo: float | None = None,
    fpr_bounds: tuple[float, float] | None = None,
) -> None:
    """Concatenate prediction lists and save precision-recall metrics if data is present.

    Args:
        scores_list: List of anomaly score arrays.
        labels_list: List of ground-truth label arrays.
        output_path: Output .npz destination path.
        level: Evaluation level tag ("image" or "pixel").
        aupimo: Genuine AUPIMO score computed on full 2D maps, when available.
        fpr_bounds: False Positive Rate integration bounds, when available.
    """
    if not (scores_list and labels_list):
        return

    y_score = np.concatenate(scores_list)
    y_true = np.concatenate(labels_list)

    compute_and_save_pr_metrics(
        y_true,
        y_score,
        output_path,
        level=level,
        aupimo=aupimo,
        fpr_bounds=fpr_bounds,
    )
    logger.info("Saved %s-level PR metrics to %s", level, output_path)


def extract_and_save_pr_metrics(
    engine: Engine,
    model: Any,
    validation_dataloader: Any,
    test_dataloader: Any,
    base_dir: Path,
    run_heatmap: bool = False,
    model_name: str = "PatchCore",
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    dict[int, dict[str, list[Any]]],
    list[int],
    int,
    int,
    int,
    int,
    float,
    float,
]:
    """Extract model predictions and persist Precision-Recall metrics for visual analysis.

    Args:
        engine: Anomalib engine instance.
        model: Trained model.
        validation_dataloader: Loader containing only shared normal validation images.
        test_dataloader: Loader containing the unchanged official test partition.
        base_dir: Output directory for metrics.
        run_heatmap: Whether to compute heatmap overlays.
        model_name: Human-readable model name used in diagnostics.

    Returns:
        18-tuple of evaluated metrics and overlay dictionaries.
    """
    try:
        logger.info("Extracting predictions for PR curve metrics...")
        visualizer = getattr(model, "visualizer", None)
        if isinstance(visualizer, _RawScoreImageVisualizer):
            visualizer.render_enabled = False
        validation_predictions = engine.predict(model=model, dataloaders=validation_dataloader)
        if not validation_predictions:
            raise RuntimeError(f"{model_name} prediction returned no validation batches")
        validation_scores: list[np.ndarray[Any, Any]] = []
        validation_pixel_scores: list[np.ndarray[Any, Any]] = []
        for batch in validation_predictions:
            scores = _tensor_to_numpy(getattr(batch, "pred_score", None))
            maps = _tensor_to_numpy(getattr(batch, "anomaly_map", None))
            if scores is not None:
                validation_scores.append(scores)
            if maps is not None:
                validation_pixel_scores.append(maps)
        if not validation_scores:
            raise RuntimeError(f"{model_name} validation predictions did not contain image scores")
        if not validation_pixel_scores:
            raise RuntimeError(f"{model_name} validation predictions did not contain anomaly maps")

        # Freeze deployment thresholds using only the shared normal validation partition.
        img_threshold = compute_adaptive_threshold(
            np.concatenate(validation_scores),
            method="quantile",
            quantile=PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
        )
        pix_threshold = compute_adaptive_threshold(
            np.concatenate(validation_pixel_scores),
            method="quantile",
            quantile=PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
        )
        if isinstance(visualizer, _RawScoreImageVisualizer):
            visualizer.pixel_threshold = pix_threshold
            visualizer.render_enabled = True

        raw_predictions = engine.predict(model=model, dataloaders=test_dataloader)
        if not raw_predictions:
            raise RuntimeError(f"{model_name} prediction returned no test batches")
        predictions = raw_predictions

        pixel_scores, pixel_labels = [], []
        image_scores, image_labels = [], []
        anomaly_maps: list[np.ndarray] = []
        ground_truth_masks: list[np.ndarray | None] = []

        for batch in predictions:
            if px := _collect_batch_tensors(batch, "anomaly_map", "gt_mask"):
                pixel_scores.append(px[0])
                pixel_labels.append(px[1])

                batch_item: Any = batch
                map_tensor = batch_item.anomaly_map
                mask_tensor = batch_item.gt_mask
                maps = map_tensor.detach().cpu().numpy()
                masks = mask_tensor.detach().cpu().numpy()
                if maps.ndim == 4 and maps.shape[1] == 1:
                    maps = maps[:, 0]
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks[:, 0]
                if maps.ndim != 3 or masks.ndim != 3 or maps.shape != masks.shape:
                    raise ValueError(
                        "AUPIMO requires matching PatchCore maps and masks with shape (N, H, W); "
                        f"got maps={maps.shape}, masks={masks.shape}"
                    )
                anomaly_maps.extend(np.asarray(item, dtype=np.float32) for item in maps)
                ground_truth_masks.extend(np.asarray(item, dtype=np.uint8) for item in masks)

            if img := _collect_batch_tensors(batch, "pred_score", "gt_label"):
                image_scores.append(img[0])
                image_labels.append(img[1])

        if not anomaly_maps or not ground_truth_masks:
            raise RuntimeError(f"{model_name} predictions did not contain full anomaly maps and ground-truth masks")

        image_scores_np = np.concatenate(image_scores)
        image_labels_np = np.concatenate(image_labels).astype(np.uint8)
        if len(np.unique(image_labels_np)) < 2:
            raise ValueError(f"{model_name} image AUROC requires both normal and anomalous test labels")
        image_auroc = float(roc_auc_score(image_labels_np, image_scores_np))
        shared_pixel_metrics, canonical_maps, canonical_masks = compute_shared_pixel_metrics(
            anomaly_maps, ground_truth_masks, image_labels_np
        )
        fpr_bounds = AUPIMO_FPR_BOUNDS
        pixel_aupimo = float(shared_pixel_metrics["pixel_aupimo"])
        pixel_auroc = float(shared_pixel_metrics["pixel_auroc"])
        stacked_maps = canonical_maps
        anomaly_map_min = float(stacked_maps.min())
        anomaly_map_max = float(stacked_maps.max())
        anomaly_map_range = anomaly_map_max - anomaly_map_min
        logger.info(
            "%s full-map AUPIMO: %.6f at FPR bounds %s | maps min=%.8f max=%.8f range=%.8f",
            model_name,
            pixel_aupimo,
            fpr_bounds,
            anomaly_map_min,
            anomaly_map_max,
            anomaly_map_range,
        )

        _process_and_save_level(
            [canonical_maps.reshape(-1)],
            [canonical_masks.reshape(-1)],
            base_dir / "pixel_metrics.npz",
            level="pixel",
            aupimo=pixel_aupimo,
            fpr_bounds=fpr_bounds,
        )
        _process_and_save_level(image_scores, image_labels, base_dir / "image_metrics.npz", level="image")

        confusion = compute_image_confusion_metrics(image_labels_np, image_scores_np, img_threshold)
        manual_image_f1 = float(confusion["f1_score"])
        manual_image_prec = float(confusion["precision"])
        manual_image_rec = float(confusion["recall"])

        pixel_scores_np = canonical_maps.reshape(-1)
        pixel_labels_np = canonical_masks.reshape(-1)
        pix_preds = (pixel_scores_np > pix_threshold).astype(int)
        manual_pixel_f1 = float(f1_score(pixel_labels_np, pix_preds))

        heatmap_overlays: dict[int, dict[str, list[Any]]] = {}
        anomalous_indices: list[int] = []

        if run_heatmap:
            try:
                import cv2

                from app.pipelines.evaluation.heatmaps import overlay_ground_truth, overlay_heatmap

                global_idx = 0
                for batch in predictions:
                    images_t = getattr(batch, "image", None)
                    anomaly_maps_t = getattr(batch, "anomaly_map", None)
                    gt_masks_t = getattr(batch, "gt_mask", None)
                    gt_labels_t = getattr(batch, "gt_label", None)

                    if images_t is None or anomaly_maps_t is None or gt_labels_t is None:
                        continue

                    images_np = images_t.detach().cpu().numpy()
                    anomaly_maps_np = anomaly_maps_t.detach().cpu().numpy()
                    gt_masks_np = gt_masks_t.detach().cpu().numpy() if gt_masks_t is not None else None
                    gt_labels_np = gt_labels_t.detach().cpu().numpy()

                    for i in range(len(gt_labels_np)):
                        if int(gt_labels_np[i]) == 1:
                            img = images_np[i]
                            if img.ndim == 3 and img.shape[0] in (1, 3):
                                img = np.transpose(img, (1, 2, 0))

                            if img.dtype != np.uint8:
                                if img.max() <= 1.0:
                                    orig_img = (img * 255).astype(np.uint8)
                                else:
                                    orig_img = img.astype(np.uint8)
                            else:
                                orig_img = img

                            amap = anomaly_maps_np[i].squeeze()
                            p_low = float(np.percentile(amap, 1))
                            p_high = float(np.percentile(amap, 99))
                            if abs(p_high - p_low) > 1e-8:
                                amap_norm = np.clip((amap - p_low) / (p_high - p_low), 0.0, 1.0)
                            else:
                                amap_norm = np.zeros_like(amap)

                            hm_overlay = overlay_heatmap(orig_img, amap_norm.astype(np.float32), alpha=0.4)

                            gt_mask_img = None
                            if gt_masks_np is not None and gt_masks_np[i] is not None:
                                gt_mask_arr = gt_masks_np[i]
                                if hasattr(gt_mask_arr, "squeeze"):
                                    gt_mask_img = gt_mask_arr.squeeze()
                                    if gt_mask_img.shape[:2] != orig_img.shape[:2]:
                                        gt_mask_img = cv2.resize(
                                            gt_mask_img.astype(np.float32),
                                            (orig_img.shape[1], orig_img.shape[0]),
                                            interpolation=cv2.INTER_NEAREST,
                                        )

                            gt_and_heatmap = overlay_ground_truth(hm_overlay, gt_mask_img)

                            max_dim = 256
                            if hm_overlay.shape[0] > max_dim or hm_overlay.shape[1] > max_dim:
                                scale = max_dim / max(hm_overlay.shape[0], hm_overlay.shape[1])
                                new_size = (int(hm_overlay.shape[1] * scale), int(hm_overlay.shape[0] * scale))
                                hm_overlay_small = cv2.resize(hm_overlay, new_size, interpolation=cv2.INTER_AREA)
                                gt_and_heatmap_small = cv2.resize(
                                    gt_and_heatmap, new_size, interpolation=cv2.INTER_AREA
                                )
                            else:
                                hm_overlay_small = hm_overlay
                                gt_and_heatmap_small = gt_and_heatmap

                            anomalous_indices.append(global_idx)
                            heatmap_overlays[global_idx] = {
                                "heatmap": hm_overlay_small.tolist(),
                                "gt_and_heatmap": gt_and_heatmap_small.tolist(),
                            }
                        global_idx += 1
            except Exception as e:
                logger.warning("Failed to compute heatmap overlays: %s", e)

        return (
            manual_image_f1,
            manual_pixel_f1,
            manual_image_prec,
            manual_image_rec,
            img_threshold,
            pixel_auroc,
            pixel_aupimo,
            anomaly_map_min,
            anomaly_map_max,
            anomaly_map_range,
            heatmap_overlays,
            anomalous_indices,
            int(confusion["true_positives"]),
            int(confusion["false_positives"]),
            int(confusion["false_negatives"]),
            int(confusion["true_negatives"]),
            pix_threshold,
            image_auroc,
        )

    except Exception as e:
        logger.exception("Could not compute %s evaluation metrics: %s", model_name, e)
        raise


def format_results(
    test_results: list[Mapping[str, float]] | None,
    category: str,
    base_dir: Path,
    manual_image_f1: float,
    manual_pixel_f1: float,
    manual_image_prec: float,
    manual_image_rec: float,
    img_threshold: float,
    pixel_threshold: float,
    pixel_auroc: float,
    pixel_aupimo: float,
    anomaly_map_min: float,
    anomaly_map_max: float,
    anomaly_map_range: float,
    heatmap_overlays: dict[int, dict[str, list[Any]]],
    anomalous_indices: list[int],
    fpr_limit: float = 1e-4,
    preprocessing_steps: list[dict[str, Any]] | None = None,
    hyperparameters: dict[str, Any] | None = None,
    dataset_split: dict[str, Any] | None = None,
    model_hash: str = "",
    metadata: dict[str, Any] | None = None,
    true_positives: int = 0,
    false_positives: int = 0,
    false_negatives: int = 0,
    true_negatives: int = 0,
) -> BaselineResult:
    """Format Anomalib engine evaluation output into a structured response schema.

    Args:
        test_results: A list of metric mappings from Anomalib.
        category: The component category name.
        base_dir: Base directory to save metrics to.
        manual_image_f1: Manually calculated image-level F1 score.
        manual_pixel_f1: Manually calculated pixel-level F1 score.
        manual_image_prec: Manually calculated image-level Precision score.
        manual_image_rec: Manually calculated image-level Recall score.
        img_threshold: Manually calculated image-level classification threshold.
        pixel_threshold: Normal-validation threshold used to create predicted masks.
        pixel_auroc: Pixel AUROC from the shared canonical metric path.
        pixel_aupimo: Full-map AUPIMO computed by Anomalib.
        anomaly_map_min: Minimum PatchCore anomaly-map value.
        anomaly_map_max: Maximum PatchCore anomaly-map value.
        anomaly_map_range: Range of PatchCore anomaly-map values.
        heatmap_overlays: Dictionary of precomputed heatmap overlays.
        anomalous_indices: List of image indices corresponding to anomalies.
        fpr_limit: Maximum allowable False Positive Rate for AUPIMO threshold.
        preprocessing_steps: Optional list of active preprocessing steps.
        hyperparameters: Optional dictionary of model hyperparameters.
        dataset_split: Optional dataset partition sample counts.
        model_hash: Unique 12-char model hash.
        metadata: Full metadata dictionary.
        true_positives: Image-level true-positive count.
        false_positives: Image-level false-positive count.
        false_negatives: Image-level false-negative count.
        true_negatives: Image-level true-negative count.

    Returns:
        A dictionary containing structured image_level and pixel_level results.
    """
    if not np.isclose(fpr_limit, AUPIMO_FPR_BOUNDS[1]):
        raise ValueError(f"fair-eval-v1 requires fpr_limit={AUPIMO_FPR_BOUNDS[1]}")

    res_dict: Mapping[str, float] = test_results[0] if test_results else {}

    return {
        "category": category,
        "image_level": {
            "auroc": _to_float(res_dict.get("image_AUROC", 0.0)),
            "f1_score": manual_image_f1,
            "precision": manual_image_prec,
            "recall": manual_image_rec,
            "threshold": img_threshold,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "metrics_path": str(base_dir / "image_metrics.npz"),
        },
        "pixel_level": {
            "auroc": pixel_auroc,
            "f1_score": manual_pixel_f1,
            "threshold": pixel_threshold,
            "aupimo_score": pixel_aupimo,
            "fpr_lower_bound": 1e-5,
            "fpr_upper_bound": fpr_limit,
            "aupimo_num_thresholds": AUPIMO_NUM_THRESHOLDS,
            "canonical_height": CANONICAL_MAP_SIZE[0],
            "canonical_width": CANONICAL_MAP_SIZE[1],
            "aupimo": pixel_aupimo,
            "anomaly_map_min": anomaly_map_min,
            "anomaly_map_max": anomaly_map_max,
            "anomaly_map_range": anomaly_map_range,
            "metrics_path": str(base_dir / "pixel_metrics.npz"),
        },
        "raw_results": {k: _to_float(v) for k, v in res_dict.items()},
        "heatmap_overlays": heatmap_overlays,
        "anomalous_indices": anomalous_indices,
        "preprocessing_steps": preprocessing_steps or [],
        "hyperparameters": hyperparameters or {},
        "dataset_split": dataset_split or {},
        "model_hash": model_hash,
        "metadata": metadata or {},
    }


__all__ = [
    "_collect_batch_tensors",
    "_load_heatmap_overlays",
    "_process_and_save_level",
    "_save_heatmap_overlays",
    "_tensor_to_numpy",
    "_to_float",
    "extract_and_save_pr_metrics",
    "format_results",
]
