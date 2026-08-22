"""Runs the baseline Patchcore model on the MVTec AD dataset.

This serves as a functional baseline with PR-AUC evaluation.
"""

import hashlib
import json
import shutil
import warnings
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore
from sklearn.metrics import f1_score, precision_score, recall_score

from app.core.logger import logger
from app.pipelines.evaluation.metrics import compute_and_save_pr_metrics
from app.pipelines.evaluation.scoring import compute_adaptive_threshold
from app.pipelines.preprocessing.adapter import PreprocessingTransformAdapter
from app.pipelines.preprocessing.base import PreprocessingPipeline
from app.pipelines.preprocessing.factory import build_pipeline_from_configs

# Suppress the timm deprecation warning caused by anomalib
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")


class MetricLevelResult(TypedDict, total=False):
    """Schema for individual evaluation level metrics (image or pixel).

    Attributes:
        auroc: Area Under the Receiver Operating Characteristic Curve.
        f1_score: F1 score for the given metric level.
        precision: Precision score.
        recall: Recall score.
        threshold: Decision threshold for classification.
        threshold_limit: Anomaly score threshold at the lower bound.
        t_aupimo_min: Minimum AUPIMO threshold bound (for pixel localization).
        tpr_at_limit: Recall/catch rate at threshold_limit.
        aupimo_score: Integrated AUPIMO score.
        fpr_lower_bound: Lower bound for FPR integration.
        fpr_upper_bound: Upper bound for FPR integration.
        aupimo: AUPIMO score.
        metrics_path: Path to the .npz file containing precision, recall, and thresholds.
    """

    auroc: float
    f1_score: float
    precision: float
    recall: float
    threshold: float
    threshold_limit: float
    t_aupimo_min: float
    tpr_at_limit: float
    aupimo_score: float
    fpr_lower_bound: float
    fpr_upper_bound: float
    aupimo: float
    metrics_path: str


class BaselineResult(TypedDict, total=False):
    """Schema for overall baseline execution results.

    Attributes:
        category: The specific category being evaluated.
        image_level: Image-level evaluation metrics.
        pixel_level: Pixel-level evaluation metrics.
        raw_results: Raw evaluation results from the anomalib engine.
        heatmap_overlays: Dictionary of generated heatmap overlays.
        anomalous_indices: List of test dataset indices that are anomalous.
        preprocessing_steps: List of active preprocessing step configurations.
        hyperparameters: Dictionary of model hyperparameters.
        dataset_split: Dictionary of dataset partition sample counts.
        model_hash: Unique 12-char model hash.
        metadata: Full metadata dictionary.
    """

    category: str
    image_level: MetricLevelResult
    pixel_level: MetricLevelResult
    raw_results: dict[str, float]
    heatmap_overlays: dict[int, dict[str, list[Any]]]
    anomalous_indices: list[int]
    preprocessing_steps: list[dict[str, Any]]
    hyperparameters: dict[str, Any]
    dataset_split: dict[str, Any]
    model_hash: str
    metadata: dict[str, Any]


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


def _tensor_to_numpy(val: Any) -> np.ndarray[Any, Any] | None:
    """Helper to safely extract a 1D NumPy array from a PyTorch tensor attribute.

    Args:
        val: Value to convert to NumPy array.

    Returns:
        NumPy array representation of the value if it is a PyTorch tensor, None otherwise.
    """
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy().reshape(-1)
    return None


def _collect_batch_tensors(
    batch: Any,
    score_attr: str,
    label_attr: str,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None:
    """Extracts score and label arrays from a batch if both attributes exist and are tensors.

    Args:
        batch: Batch to extract tensors from.
        score_attr: Attribute name for scores.
        label_attr: Attribute name for labels.

    Returns:
        Tuple of score and label arrays if both attributes exist and are tensors, None otherwise.
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
    fpr_limit: float = 1e-4,
) -> None:
    """Concatenates prediction lists and saves metrics if data is present.

    Args:
        scores_list: List of anomaly scores.
        labels_list: List of ground truth labels.
        output_path: Path to save metrics to.
        level: Level of metrics (image or pixel).
        fpr_limit: Maximum allowable False Positive Rate for AUPIMO threshold.
    """
    if not (scores_list and labels_list):
        return

    y_score = np.concatenate(scores_list)
    y_true = np.concatenate(labels_list)

    compute_and_save_pr_metrics(y_true, y_score, output_path, level=level, fpr_limit=fpr_limit)
    logger.info("Saved %s-level PR metrics to %s", level, output_path)


def extract_and_save_pr_metrics(
    engine: Engine,
    model: Patchcore,
    datamodule: MVTecAD,
    base_dir: Path,
    fpr_limit: float = 1e-4,
    run_heatmap: bool = False,
) -> tuple[float, float, float, float, float, dict[int, dict[str, list[Any]]], list[int]]:
    """Extract model predictions and persist Precision-Recall metrics for visual analysis.

    Args:
        engine: Anomalib engine instance.
        model: Trained model.
        datamodule: Dataset object.
        base_dir: Output directory for metrics.
        fpr_limit: Maximum allowable False Positive Rate for AUPIMO threshold.
        run_heatmap: Whether to compute heatmap overlays.
    """
    try:
        logger.info("Extracting predictions for PR curve metrics...")
        raw_predictions = engine.predict(model=model, dataloaders=datamodule.test_dataloader())
        if not raw_predictions:
            return 0.0, 0.0, 0.0, 0.0, 0.0, {}, []
        predictions = raw_predictions

        pixel_scores, pixel_labels = [], []
        image_scores, image_labels = [], []

        for batch in predictions:
            if px := _collect_batch_tensors(batch, "anomaly_map", "gt_mask"):
                pixel_scores.append(px[0])
                pixel_labels.append(px[1])

            if img := _collect_batch_tensors(batch, "pred_score", "gt_label"):
                image_scores.append(img[0])
                image_labels.append(img[1])

        _process_and_save_level(
            pixel_scores, pixel_labels, base_dir / "pixel_metrics.npz", level="pixel", fpr_limit=fpr_limit
        )
        _process_and_save_level(
            image_scores, image_labels, base_dir / "image_metrics.npz", level="image", fpr_limit=fpr_limit
        )

        # Compute manual thresholds and F1 scores strictly on normal data
        image_scores_np = np.concatenate(image_scores)
        image_labels_np = np.concatenate(image_labels)
        normal_image_scores = image_scores_np[image_labels_np == 0]

        if len(normal_image_scores) > 0:
            img_threshold = compute_adaptive_threshold(normal_image_scores, method="quantile", quantile=0.95)
            img_preds = (image_scores_np > img_threshold).astype(int)
            manual_image_f1 = float(f1_score(image_labels_np, img_preds))
            manual_image_prec = float(precision_score(image_labels_np, img_preds, zero_division=0))
            manual_image_rec = float(recall_score(image_labels_np, img_preds, zero_division=0))
        else:
            manual_image_f1 = 0.0
            manual_image_prec = 0.0
            manual_image_rec = 0.0

        pixel_scores_np = np.concatenate(pixel_scores)
        pixel_labels_np = np.concatenate(pixel_labels)
        normal_pixel_scores = pixel_scores_np[pixel_labels_np == 0]

        if len(normal_pixel_scores) > 0:
            pix_threshold = compute_adaptive_threshold(normal_pixel_scores, method="quantile", quantile=0.95)
            pix_preds = (pixel_scores_np > pix_threshold).astype(int)
            manual_pixel_f1 = float(f1_score(pixel_labels_np.flatten(), pix_preds.flatten()))
        else:
            manual_pixel_f1 = 0.0

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
            heatmap_overlays,
            anomalous_indices,
        )

    except Exception as e:
        logger.warning("Could not auto-save evaluation metrics.npz: %s", e)
        return 0.0, 0.0, 0.0, 0.0, 0.0, {}, []


def _normalize_preprocessing_steps(steps: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize preprocessing step configurations for deterministic comparison and hashing."""
    if not steps:
        return []
    normalized: list[dict[str, Any]] = []
    for s in steps:
        name = str(s.get("name", ""))
        params = dict(s.get("params", {})) if isinstance(s.get("params"), dict) else {}
        normalized.append({"name": name, "params": params})
    normalized.sort(key=lambda x: x["name"])
    return normalized


def find_cached_patchcore_model(
    category: str,
    backbone: str = "resnet18",
    feature_layers: tuple[str, ...] = ("layer2", "layer3"),
    coreset_sampling_ratio: float = 0.1,
    num_neighbors: int = 9,
    fpr_limit: float = 1e-4,
    preprocessing_steps: list[dict[str, Any]] | None = None,
    target_hash: str | None = None,
    registry_base: Path | str = "data/models/patchcore",
) -> tuple[Path, dict[str, Any]] | None:
    """Find the newest cached Patchcore model matching either a specific hash or the given parameters.

    Args:
        category: Component category name.
        backbone: Feature extractor backbone name.
        feature_layers: Layers to extract features from.
        coreset_sampling_ratio: Ratio for coreset subsampling.
        num_neighbors: Number of nearest neighbors for scoring.
        fpr_limit: Max allowable False Positive Rate.
        preprocessing_steps: Optional preprocessing step configurations.
        target_hash: Optional exact model hash to search for.
        registry_base: Path to the patchcore model registry.

    Returns:
        Tuple of (model_dir, metadata_dict) if found, else None.
    """
    base_path = Path(registry_base)
    if not base_path.exists():
        return None

    if target_hash:
        target_dir = base_path / target_hash
        meta_file = target_dir / "metadata.json"
        if target_dir.exists() and meta_file.exists():
            try:
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
                return target_dir, meta
            except Exception:
                pass
        return None

    norm_req_prep = _normalize_preprocessing_steps(preprocessing_steps)
    candidates: list[tuple[float, Path, dict[str, Any]]] = []

    for meta_file in base_path.rglob("metadata.json"):
        if ".trash" in meta_file.parts:
            continue
        model_dir = meta_file.parent
        try:
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        if meta.get("category") != category:
            continue
        if meta.get("backbone", "resnet18") != backbone:
            continue
        if tuple(meta.get("feature_layers", ["layer2", "layer3"])) != tuple(feature_layers):
            continue
        if abs(float(meta.get("coreset_sampling_ratio", 0.1)) - float(coreset_sampling_ratio)) > 1e-5:
            continue
        if int(meta.get("num_neighbors", 9)) != int(num_neighbors):
            continue
        if abs(float(meta.get("fpr_limit", 1e-4)) - float(fpr_limit)) > 1e-6:
            continue

        meta_prep = _normalize_preprocessing_steps(meta.get("preprocessing_steps"))
        if meta_prep != norm_req_prep:
            continue

        ts_str = meta.get("timestamp", "")
        try:
            if ts_str:
                ts = datetime.fromisoformat(ts_str).timestamp()
            else:
                ts = meta_file.stat().st_mtime
        except Exception:
            ts = meta_file.stat().st_mtime

        candidates.append((ts, model_dir, meta))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, newest_dir, newest_meta = candidates[0]
    return newest_dir, newest_meta


def delete_cached_patchcore_model(
    model_hash: str,
    registry_base: str | Path = "data/models/patchcore",
    soft_delete: bool = True,
) -> bool:
    """Safely delete a cached Patchcore model directory from the model registry.

    Args:
        model_hash: The unique 12-character hex hash of the model to delete.
        registry_base: Base directory path for the model registry.
        soft_delete: If True, moves the model to .trash/; if False, permanently deletes.

    Returns:
        True if the model was found and successfully deleted/trashed, False otherwise.
    """
    if not model_hash or not isinstance(model_hash, str) or len(model_hash) < 4:
        return False

    base_path = Path(registry_base).resolve()
    if not base_path.exists():
        return False

    target_dir = (base_path / model_hash).resolve()
    if not target_dir.is_relative_to(base_path) or target_dir == base_path or target_dir.name == ".trash":
        logger.warning("Attempted invalid model deletion outside registry: %s", target_dir)
        return False

    if not (target_dir.exists() and target_dir.is_dir()):
        return False

    if soft_delete:
        trash_dir = base_path / ".trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        dest_dir = trash_dir / model_hash
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.move(str(target_dir), str(dest_dir))
        logger.info("Moved cached Patchcore model directory to trash: %s -> %s", target_dir, dest_dir)
        return True

    shutil.rmtree(target_dir)
    logger.info("Permanently deleted cached Patchcore model directory: %s", target_dir)
    return True


def restore_cached_patchcore_model(
    model_hash: str,
    registry_base: str | Path = "data/models/patchcore",
) -> bool:
    """Restore a previously soft-deleted Patchcore model from the .trash/ recovery directory.

    Args:
        model_hash: The unique 12-character hex hash of the model to restore.
        registry_base: Base directory path for the model registry.

    Returns:
        True if the model was found in .trash and restored, False otherwise.
    """
    if not model_hash or not isinstance(model_hash, str) or len(model_hash) < 4:
        return False

    base_path = Path(registry_base).resolve()
    trash_dir = base_path / ".trash"
    source_dir = (trash_dir / model_hash).resolve()
    dest_dir = (base_path / model_hash).resolve()

    if not source_dir.exists() or not source_dir.is_dir():
        return False

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.move(str(source_dir), str(dest_dir))
    logger.info("Restored Patchcore model directory from trash: %s -> %s", source_dir, dest_dir)
    return True


def list_trashed_patchcore_models(registry_base: str | Path = "data/models/patchcore") -> list[dict[str, Any]]:
    """List all Patchcore models currently held in the .trash/ recovery directory.

    Args:
        registry_base: Base directory path for the model registry.

    Returns:
        List of metadata dictionaries for all trashed models.
    """
    base_path = Path(registry_base).resolve()
    trash_dir = base_path / ".trash"
    trashed: list[dict[str, Any]] = []
    if not trash_dir.exists():
        return trashed

    for meta_file in trash_dir.rglob("metadata.json"):
        try:
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
                meta["hash"] = meta.get("hash", meta_file.parent.name)
                trashed.append(meta)
        except Exception:
            trashed.append({"hash": meta_file.parent.name})
    return trashed


def purge_patchcore_trash(
    registry_base: str | Path = "data/models/patchcore",
    model_hash: str | None = None,
) -> int:
    """Permanently delete models from the .trash/ recovery directory.

    Args:
        registry_base: Base directory path for the model registry.
        model_hash: Optional specific model hash to purge. If None, empties the entire trash.

    Returns:
        Number of model directories permanently deleted.
    """
    base_path = Path(registry_base).resolve()
    trash_dir = base_path / ".trash"
    if not trash_dir.exists():
        return 0

    purged_count = 0
    if model_hash:
        target = trash_dir / model_hash
        if target.exists() and target.is_dir():
            shutil.rmtree(target)
            purged_count = 1
            logger.info("Purged model %s from trash.", model_hash)
    else:
        for child in trash_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
                purged_count += 1
        logger.info("Emptied Patchcore trash: purged %d models.", purged_count)

    return purged_count


def format_results(
    test_results: list[Mapping[str, float]] | None,
    category: str,
    base_dir: Path,
    manual_image_f1: float,
    manual_pixel_f1: float,
    manual_image_prec: float,
    manual_image_rec: float,
    img_threshold: float,
    heatmap_overlays: dict[int, dict[str, list[Any]]],
    anomalous_indices: list[int],
    fpr_limit: float = 1e-4,
    preprocessing_steps: list[dict[str, Any]] | None = None,
    hyperparameters: dict[str, Any] | None = None,
    dataset_split: dict[str, Any] | None = None,
    model_hash: str = "",
    metadata: dict[str, Any] | None = None,
) -> BaselineResult:
    """Format anomalib engine evaluation output into a structured response schema.

    Args:
        test_results: A list of metric mappings from Anomalib.
        category: The component category name.
        base_dir: Base directory to save metrics to.
        manual_image_f1: Manually calculated image-level F1 score.
        manual_pixel_f1: Manually calculated pixel-level F1 score.
        manual_image_prec: Manually calculated image-level Precision score.
        manual_image_rec: Manually calculated image-level Recall score.
        img_threshold: Manually calculated image-level classification threshold.
        heatmap_overlays: Dictionary of precomputed heatmap overlays.
        anomalous_indices: List of image indices corresponding to anomalies.
        fpr_limit: Maximum allowable False Positive Rate for AUPIMO threshold.
        preprocessing_steps: Optional list of active preprocessing steps.
        hyperparameters: Optional dictionary of model hyperparameters.
        dataset_split: Optional dataset partition sample counts.
        model_hash: Unique 12-char model hash.
        metadata: Full metadata dictionary.

    Returns:
        A dictionary containing structured image_level and pixel_level results.
    """
    res_dict: Mapping[str, float] = test_results[0] if test_results else {}
    t_aupimo_min = 0.0
    aupimo = 0.0

    pixel_file = base_dir / "pixel_metrics.npz"
    if pixel_file.exists():
        try:
            data = np.load(pixel_file)
            if "t_aupimo_min" in data:
                t_aupimo_min = float(data["t_aupimo_min"])
            if "aupimo" in data:
                aupimo = float(data["aupimo"])
        except Exception:
            pass

    return {
        "category": category,
        "image_level": {
            "auroc": _to_float(res_dict.get("image_AUROC", 0.0)),
            "f1_score": manual_image_f1,
            "precision": manual_image_prec,
            "recall": manual_image_rec,
            "threshold": img_threshold,
            "metrics_path": str(base_dir / "image_metrics.npz"),
        },
        "pixel_level": {
            "auroc": _to_float(res_dict.get("pixel_AUROC", 0.0)),
            "f1_score": manual_pixel_f1,
            "aupimo_score": _to_float(res_dict.get("pixel_AUPIMO", 0.0)),
            "threshold_limit": t_aupimo_min,
            "tpr_at_limit": aupimo,
            "fpr_lower_bound": 1e-5,
            "fpr_upper_bound": fpr_limit,
            "t_aupimo_min": t_aupimo_min,
            "aupimo": _to_float(res_dict.get("pixel_AUPIMO", 0.0)),
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


def run_baseline(
    data_root: Path | str = "data/raw/mvtec_ad",
    category: str = "bottle",
    pipeline: list[dict[str, Any]] | PreprocessingPipeline | None = None,
    fpr_limit: float = 1e-4,
    backbone: str = "resnet18",
    feature_layers: tuple[str, ...] = ("layer2", "layer3"),
    coreset_sampling_ratio: float = 0.1,
    num_neighbors: int = 9,
    run_heatmap: bool = False,
    preprocessing_steps: list[dict[str, Any]] | None = None,
    force_retrain: bool = False,
    model_hash: str | None = None,
    registry_base: Path | str = "data/models/patchcore",
) -> BaselineResult:
    """Run the baseline Patchcore model on the MVTec AD dataset.

    Args:
        data_root: Root directory of MVTec AD.
        category: Category to evaluate.
        pipeline: Optional list of preprocessing step configurations or pipeline.
        fpr_limit: Maximum allowable False Positive Rate.
        backbone: Feature extractor backbone (e.g. 'resnet18', 'wide_resnet50_2').
        feature_layers: Layers to extract features from.
        coreset_sampling_ratio: Ratio for coreset subsampling.
        num_neighbors: Number of nearest neighbors for scoring.
        run_heatmap: Whether to compute heatmap overlays.
        preprocessing_steps: Deprecated alias for pipeline configuration list.
        force_retrain: If True, ignores cache and forces a full re-fit.
        model_hash: Optional target model hash to search for.
        registry_base: Base directory path for Patchcore model registry.

    Returns:
        Structured evaluation metrics.
    """
    steps_config = pipeline if pipeline is not None else preprocessing_steps
    if isinstance(steps_config, PreprocessingPipeline):
        proc_pipeline = steps_config
        raw_prep_list = []
    else:
        proc_pipeline = build_pipeline_from_configs(steps_config)
        raw_prep_list = _normalize_preprocessing_steps(steps_config)

    norm_prep_str = json.dumps(raw_prep_list, sort_keys=True)
    layer_str = "_".join(feature_layers)
    hp_string = (
        f"{category}_{backbone}_{layer_str}_{coreset_sampling_ratio}_{num_neighbors}_{fpr_limit}_{norm_prep_str}"
    )
    computed_hash = hashlib.sha256(hp_string.encode()).hexdigest()[:12]

    cached = find_cached_patchcore_model(
        category=category,
        backbone=backbone,
        feature_layers=feature_layers,
        coreset_sampling_ratio=coreset_sampling_ratio,
        num_neighbors=num_neighbors,
        fpr_limit=fpr_limit,
        preprocessing_steps=raw_prep_list,
        target_hash=model_hash,
        registry_base=registry_base,
    )

    if cached is not None and not force_retrain:
        cached_dir, meta = cached
        logger.info("Found cached Patchcore model in %s. Loading evaluation metrics...", cached_dir)
        pixel_file = cached_dir / "pixel_metrics.npz"
        image_file = cached_dir / "image_metrics.npz"

        t_aupimo_min = 0.0
        aupimo = 0.0
        if pixel_file.exists():
            try:
                data = np.load(pixel_file)
                if "t_aupimo_min" in data:
                    t_aupimo_min = float(data["t_aupimo_min"])
                if "aupimo" in data:
                    aupimo = float(data["aupimo"])
            except Exception:
                pass

        img_auroc = 0.0
        if image_file.exists():
            try:
                data_img = np.load(image_file)
                if "auroc" in data_img:
                    img_auroc = float(data_img["auroc"])
            except Exception:
                pass

        res_dict = meta.get("raw_results", {})
        manual_image_f1 = float(meta.get("manual_image_f1", 0.0))
        manual_pixel_f1 = float(meta.get("manual_pixel_f1", 0.0))
        manual_image_prec = float(meta.get("manual_image_prec", 0.0))
        manual_image_rec = float(meta.get("manual_image_rec", 0.0))
        img_threshold = float(meta.get("img_threshold", 0.0))
        heatmap_overlays = meta.get("heatmap_overlays", {})
        anomalous_indices = meta.get("anomalous_indices", [])
        split_info = meta.get("dataset_split", {})
        cached_prep = meta.get("preprocessing_steps", raw_prep_list)
        hyperparams = meta.get(
            "hyperparameters",
            {
                "backbone": meta.get("backbone", backbone),
                "feature_layers": meta.get("feature_layers", feature_layers),
                "coreset_sampling_ratio": meta.get("coreset_sampling_ratio", coreset_sampling_ratio),
                "num_neighbors": meta.get("num_neighbors", num_neighbors),
                "fpr_limit": meta.get("fpr_limit", fpr_limit),
                "train_batch_size": 16,
                "eval_batch_size": 16,
            },
        )

        return {
            "category": category,
            "image_level": {
                "auroc": img_auroc or _to_float(res_dict.get("image_AUROC", 0.0)),
                "f1_score": manual_image_f1,
                "precision": manual_image_prec,
                "recall": manual_image_rec,
                "threshold": img_threshold,
                "metrics_path": str(image_file),
            },
            "pixel_level": {
                "auroc": aupimo or _to_float(res_dict.get("pixel_AUROC", 0.0)),
                "f1_score": manual_pixel_f1,
                "aupimo_score": aupimo or _to_float(res_dict.get("pixel_AUPIMO", 0.0)),
                "threshold_limit": t_aupimo_min,
                "tpr_at_limit": aupimo,
                "fpr_lower_bound": 1e-5,
                "fpr_upper_bound": fpr_limit,
                "t_aupimo_min": t_aupimo_min,
                "aupimo": aupimo or _to_float(res_dict.get("pixel_AUPIMO", 0.0)),
                "metrics_path": str(pixel_file),
            },
            "raw_results": {k: _to_float(v) for k, v in res_dict.items()},
            "heatmap_overlays": heatmap_overlays,
            "anomalous_indices": anomalous_indices,
            "preprocessing_steps": cached_prep,
            "hyperparameters": hyperparams,
            "dataset_split": split_info,
            "model_hash": meta.get("hash", cached_dir.name),
            "metadata": meta,
        }

    effective_hash = model_hash or computed_hash
    logger.info("Configured preprocessing pipeline with %d steps.", len(proc_pipeline))
    base_dir = Path(registry_base) / effective_hash
    base_dir.mkdir(parents=True, exist_ok=True)
    transform_adapter = PreprocessingTransformAdapter(proc_pipeline)

    # 1. Initialize dataset, model, and engine
    datamodule = MVTecAD(
        root=data_root,
        category=category,
        train_batch_size=16,
        eval_batch_size=16,
    )

    if len(proc_pipeline) > 0:
        # Setup the datamodule datasets so train_data and test_data are instantiated
        datamodule.setup()

        # Assign the adapter transform to the underlying datasets
        train_data = getattr(datamodule, "train_data", None)
        if train_data is not None:
            train_data.transform = transform_adapter

        test_data = getattr(datamodule, "test_data", None)
        if test_data is not None:
            test_data.transform = transform_adapter

    model = Patchcore(
        backbone=backbone,
        layers=feature_layers,
        coreset_sampling_ratio=coreset_sampling_ratio,
        num_neighbors=num_neighbors,
    )
    engine = Engine(accelerator="gpu", devices=1)

    # 2. Fit and Test
    logger.info("Fitting Patchcore model on %s category (Hash: %s)...", category, effective_hash)
    engine.fit(model, datamodule)

    logger.info("Testing Patchcore model...")
    test_results = engine.test(model=model, datamodule=datamodule)

    # 3. Extract PR metrics and build summary
    (
        manual_image_f1,
        manual_pixel_f1,
        manual_image_prec,
        manual_image_rec,
        img_threshold,
        heatmap_overlays,
        anomalous_indices,
    ) = extract_and_save_pr_metrics(engine, model, datamodule, base_dir, fpr_limit, run_heatmap)

    # Extract dataset split counts if datamodule was setup
    split_info = {}
    train_ds = getattr(datamodule, "train_data", None)
    if train_ds is not None and hasattr(train_ds, "__len__"):
        split_info["train_normal"] = len(train_ds)
    test_ds = getattr(datamodule, "test_data", None)
    if test_ds is not None and hasattr(test_ds, "__len__"):
        split_info["test_total"] = len(test_ds)

    hyperparams = {
        "backbone": backbone,
        "feature_layers": feature_layers,
        "coreset_sampling_ratio": coreset_sampling_ratio,
        "num_neighbors": num_neighbors,
        "fpr_limit": fpr_limit,
        "train_batch_size": 16,
        "eval_batch_size": 16,
    }

    raw_results_dict = {k: _to_float(v) for k, v in (test_results[0] if test_results else {}).items()}

    metadata = {
        "hash": effective_hash,
        "model_type": "patchcore",
        "category": category,
        "backbone": backbone,
        "feature_layers": feature_layers,
        "coreset_sampling_ratio": coreset_sampling_ratio,
        "num_neighbors": num_neighbors,
        "fpr_limit": fpr_limit,
        "preprocessing_steps": raw_prep_list,
        "hyperparameters": hyperparams,
        "dataset_split": split_info,
        "manual_image_f1": manual_image_f1,
        "manual_pixel_f1": manual_pixel_f1,
        "manual_image_prec": manual_image_prec,
        "manual_image_rec": manual_image_rec,
        "img_threshold": img_threshold,
        "heatmap_overlays": heatmap_overlays,
        "anomalous_indices": anomalous_indices,
        "raw_results": raw_results_dict,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    try:
        with open(base_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        logger.info("Saved Patchcore model metadata to %s", base_dir / "metadata.json")
    except Exception as e:
        logger.warning("Could not save Patchcore metadata.json: %s", e)

    return format_results(
        test_results=test_results,
        category=category,
        base_dir=base_dir,
        manual_image_f1=manual_image_f1,
        manual_pixel_f1=manual_pixel_f1,
        manual_image_prec=manual_image_prec,
        manual_image_rec=manual_image_rec,
        img_threshold=img_threshold,
        heatmap_overlays=heatmap_overlays,
        anomalous_indices=anomalous_indices,
        fpr_limit=fpr_limit,
        preprocessing_steps=raw_prep_list,
        hyperparameters=hyperparams,
        dataset_split=split_info,
        model_hash=effective_hash,
        metadata=metadata,
    )


if __name__ == "__main__":
    run_baseline()
