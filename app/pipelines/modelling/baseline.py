"""Runs the baseline Patchcore model on the MVTec AD dataset.

This serves as a functional baseline with PR-AUC evaluation.
"""

import hashlib
import json
import shutil
import warnings
from collections.abc import Mapping
from copy import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.visualization import ImageVisualizer
from lightning import seed_everything
from sklearn.metrics import f1_score, roc_auc_score

from app.core.logger import logger
from app.domain.data import (
    FAIR_EVALUATION_PROTOCOL,
    FairEvaluationSplit,
    build_fair_evaluation_split,
    build_mvtec_manifest,
)
from app.pipelines.evaluation.metrics import (
    AUPIMO_FPR_BOUNDS,
    AUPIMO_NUM_THRESHOLDS,
    CANONICAL_MAP_SIZE,
    PIXEL_METRICS_VERSION,
    compute_and_save_pr_metrics,
    compute_image_confusion_metrics,
    compute_shared_pixel_metrics,
    fair_metric_evidence,
)
from app.pipelines.evaluation.scoring import compute_adaptive_threshold
from app.pipelines.preprocessing.adapter import PreprocessingTransformAdapter
from app.pipelines.preprocessing.base import PreprocessingPipeline
from app.pipelines.preprocessing.factory import build_pipeline_from_configs

# Suppress the timm deprecation warning caused by anomalib
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")

PATCHCORE_MODEL_SEED = 42
PATCHCORE_SCORE_SPACE = "raw"
PATCHCORE_IMAGE_THRESHOLD_QUANTILE = 0.95
PATCHCORE_PIXEL_THRESHOLD_QUANTILE = 0.99


def _seed_patchcore_run(seed: int) -> None:
    """Seed every random source used by PatchCore and its data loaders."""
    seed_everything(seed, workers=True, verbose=False)


class _RawScoreImageVisualizer(ImageVisualizer):
    """Normalize a display copy of raw maps before Anomalib converts them to 8-bit images."""

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
        anomaly_map = getattr(batch, "anomaly_map", None)
        if not isinstance(anomaly_map, torch.Tensor) or anomaly_map.ndim < 2:
            super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
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
        visualization_batch = batch.update(in_place=False, anomaly_map=normalized_map)
        super().on_test_batch_end(
            trainer,
            pl_module,
            outputs,
            visualization_batch,
            batch_idx,
            dataloader_idx,
        )


def _print_patchcore_results_table(results: Mapping[str, float]) -> None:
    """Print corrected PatchCore metrics in Anomalib's former table layout."""
    from rich.console import Console
    from rich.table import Table

    table = Table(show_header=True, header_style="bold")
    table.add_column("Test metric")
    table.add_column("DataLoader 0", justify="right")
    for name, value in results.items():
        table.add_row(name, f"{value:.6f}")
    Console().print(table)


def _dataset_with_ordered_paths(dataset: Any, ordered_paths: list[str]) -> Any:
    """Copy an Anomalib dataset and restrict it to an exact ordered path list."""
    samples = getattr(dataset, "samples", None)
    if samples is None or "image_path" not in samples.columns:
        raise TypeError("PatchCore dataset must expose a samples frame with image_path")
    indexed = {
        str(Path(path).expanduser().resolve()): index for index, path in enumerate(samples["image_path"].astype(str))
    }
    requested = [str(Path(path).expanduser().resolve()) for path in ordered_paths]
    missing = [path for path in requested if path not in indexed]
    if missing:
        raise ValueError(f"PatchCore dataset is missing {len(missing)} protocol paths")
    restricted = copy(dataset)
    restricted.samples = samples.iloc[[indexed[path] for path in requested]].reset_index(drop=True).copy()
    return restricted


def _configure_patchcore_partitions(
    datamodule: MVTecAD,
    fair_split: FairEvaluationSplit,
    transform_adapter: PreprocessingTransformAdapter | None = None,
) -> None:
    """Replace Anomalib's implicit partitions with the shared fair split."""
    datamodule.setup()
    source_train = datamodule.train_data
    source_test = datamodule.test_data
    train_data = _dataset_with_ordered_paths(source_train, fair_split.fitting_paths)
    validation_data = _dataset_with_ordered_paths(source_train, fair_split.validation_paths)
    test_data = _dataset_with_ordered_paths(source_test, fair_split.test_paths)
    if transform_adapter is not None:
        from app.pipelines.preprocessing.adapter import PreprocessedAnomalibDataset

        train_data = PreprocessedAnomalibDataset(train_data, transform_adapter)
        validation_data = PreprocessedAnomalibDataset(validation_data, transform_adapter)
        test_data = PreprocessedAnomalibDataset(test_data, transform_adapter)
    datamodule.train_data = train_data
    datamodule.val_data = validation_data
    datamodule.test_data = test_data


class MetricLevelResult(TypedDict, total=False):
    """Schema for individual evaluation level metrics (image or pixel).

    Attributes:
        auroc: Area Under the Receiver Operating Characteristic Curve.
        f1_score: F1 score for the given metric level.
        precision: Precision score.
        recall: Recall score.
        threshold: Decision threshold for classification.
        aupimo_score: Integrated AUPIMO score.
        fpr_lower_bound: Lower bound for FPR integration.
        fpr_upper_bound: Upper bound for FPR integration.
        aupimo: AUPIMO score.
        anomaly_map_min: Minimum anomaly-map score over the test set.
        anomaly_map_max: Maximum anomaly-map score over the test set.
        anomaly_map_range: Difference between maximum and minimum map scores.
        metrics_path: Path to the .npz file containing precision, recall, and thresholds.
    """

    auroc: float
    f1_score: float
    precision: float
    recall: float
    threshold: float
    aupimo_score: float
    fpr_lower_bound: float
    fpr_upper_bound: float
    aupimo: float
    anomaly_map_min: float
    anomaly_map_max: float
    anomaly_map_range: float
    metrics_path: str
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    aupimo_num_thresholds: int
    canonical_height: int
    canonical_width: int


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


def _save_heatmap_overlays(
    overlays: dict[int, dict[str, list[Any]]],
    output_path: Path,
) -> Path | None:
    """Store dense heatmap images in a compressed binary archive instead of JSON."""
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
    """Load heatmap images from the compressed archive into the existing result schema."""
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
    aupimo: float | None = None,
    fpr_bounds: tuple[float, float] | None = None,
) -> None:
    """Concatenates prediction lists and saves metrics if data is present.

    Args:
        scores_list: List of anomaly scores.
        labels_list: List of ground truth labels.
        output_path: Path to save metrics to.
        level: Level of metrics (image or pixel).
        aupimo: Genuine AUPIMO computed from full 2D maps, when available.
        fpr_bounds: FPR integration bounds used for AUPIMO, when available.
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
    model: Patchcore,
    validation_dataloader: Any,
    test_dataloader: Any,
    base_dir: Path,
    run_heatmap: bool = False,
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
]:
    """Extract model predictions and persist Precision-Recall metrics for visual analysis.

    Args:
        engine: Anomalib engine instance.
        model: Trained model.
        validation_dataloader: Loader containing only shared normal validation images.
        test_dataloader: Loader containing the unchanged official test partition.
        base_dir: Output directory for metrics.
        run_heatmap: Whether to compute heatmap overlays.
    """
    try:
        logger.info("Extracting predictions for PR curve metrics...")
        validation_predictions = engine.predict(model=model, dataloaders=validation_dataloader)
        if not validation_predictions:
            raise RuntimeError("PatchCore prediction returned no validation batches")
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
            raise RuntimeError("PatchCore validation predictions did not contain image scores")

        raw_predictions = engine.predict(model=model, dataloaders=test_dataloader)
        if not raw_predictions:
            raise RuntimeError("PatchCore prediction returned no test batches")
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
            raise RuntimeError("PatchCore predictions did not contain full anomaly maps and ground-truth masks")

        image_scores_np = np.concatenate(image_scores)
        image_labels_np = np.concatenate(image_labels).astype(np.uint8)
        if len(np.unique(image_labels_np)) < 2:
            raise ValueError("PatchCore image AUROC requires both normal and anomalous test labels")
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
            "PatchCore full-map AUPIMO: %.6f at FPR bounds %s | maps min=%.8f max=%.8f range=%.8f",
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

        # Freeze thresholds using only the shared normal validation partition.
        img_threshold = compute_adaptive_threshold(
            np.concatenate(validation_scores),
            method="quantile",
            quantile=PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
        )
        confusion = compute_image_confusion_metrics(image_labels_np, image_scores_np, img_threshold)
        manual_image_f1 = float(confusion["f1_score"])
        manual_image_prec = float(confusion["precision"])
        manual_image_rec = float(confusion["recall"])

        if not validation_pixel_scores:
            raise RuntimeError("PatchCore validation predictions did not contain anomaly maps")
        pix_threshold = compute_adaptive_threshold(
            np.concatenate(validation_pixel_scores),
            method="quantile",
            quantile=PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
        )
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
            image_auroc,
        )

    except Exception as e:
        logger.exception("Could not compute PatchCore evaluation metrics: %s", e)
        raise


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
    expected_split_evidence: dict[str, Any] | None = None,
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
        expected_split_evidence: Required fair-protocol split evidence, when evaluating a cache hit.

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
                if expected_split_evidence is None or all(
                    meta.get("dataset_split", {}).get(key) == value for key, value in expected_split_evidence.items()
                ):
                    return target_dir, meta
                return None
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

        if expected_split_evidence is not None and not all(
            meta.get("dataset_split", {}).get(key) == value for key, value in expected_split_evidence.items()
        ):
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
        pixel_auroc: Pixel AUROC from the shared canonical metric path.
        pixel_aupimo: Full-map AUPIMO computed by anomalib.
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
    model_seed: int = PATCHCORE_MODEL_SEED,
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
        model_seed: Seed controlling PatchCore coreset sampling and data-loader workers.

    Returns:
        Structured evaluation metrics.
    """
    if not np.isclose(fpr_limit, AUPIMO_FPR_BOUNDS[1]):
        raise ValueError(f"fair-eval-v1 requires fpr_limit={AUPIMO_FPR_BOUNDS[1]}")

    steps_config = pipeline if pipeline is not None else preprocessing_steps
    if isinstance(steps_config, PreprocessingPipeline):
        proc_pipeline = steps_config
        raw_prep_list = []
    else:
        proc_pipeline = build_pipeline_from_configs(steps_config)
        raw_prep_list = _normalize_preprocessing_steps(steps_config)

    manifest = build_mvtec_manifest(data_root)
    fair_split = build_fair_evaluation_split(manifest, category)
    split_evidence = fair_split.evidence()
    cache_evidence = {
        **split_evidence,
        **fair_metric_evidence(),
        "model_seed": model_seed,
        "score_space": PATCHCORE_SCORE_SPACE,
        "image_threshold_quantile": PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
        "pixel_threshold_quantile": PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
    }
    norm_prep_str = json.dumps(raw_prep_list, sort_keys=True)
    layer_str = "_".join(feature_layers)
    hp_string = (
        f"{category}_{backbone}_{layer_str}_{coreset_sampling_ratio}_{num_neighbors}_{fpr_limit}_{norm_prep_str}_"
        f"{json.dumps(cache_evidence, sort_keys=True)}_{CANONICAL_MAP_SIZE}_{AUPIMO_FPR_BOUNDS}_"
        f"{AUPIMO_NUM_THRESHOLDS}_{PIXEL_METRICS_VERSION}"
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
        expected_split_evidence=cache_evidence,
    )

    if cached is not None and not force_retrain:
        cached_dir, meta = cached
        logger.info("Found cached Patchcore model in %s. Loading evaluation metrics...", cached_dir)
        pixel_file = cached_dir / "pixel_metrics.npz"
        image_file = cached_dir / "image_metrics.npz"

        if not pixel_file.exists() or not image_file.exists():
            raise FileNotFoundError("Fair PatchCore cache is missing required metric artifacts")
        with np.load(pixel_file, allow_pickle=False) as pixel_data:
            if "aupimo" not in pixel_data:
                raise ValueError("Fair PatchCore pixel metrics are missing genuine AUPIMO")
            aupimo = float(pixel_data["aupimo"])

        required_metric_keys = {
            "image_auroc",
            "pixel_auroc",
            "manual_image_f1",
            "manual_pixel_f1",
            "manual_image_prec",
            "manual_image_rec",
            "img_threshold",
            "true_positives",
            "false_positives",
            "false_negatives",
            "true_negatives",
        }
        if missing_keys := required_metric_keys.difference(meta):
            raise ValueError(f"Fair PatchCore metadata is missing metrics: {sorted(missing_keys)}")
        res_dict = meta.get("raw_results", {})
        manual_image_f1 = float(meta["manual_image_f1"])
        manual_pixel_f1 = float(meta["manual_pixel_f1"])
        manual_image_prec = float(meta["manual_image_prec"])
        manual_image_rec = float(meta["manual_image_rec"])
        img_threshold = float(meta["img_threshold"])
        heatmap_overlays_path = meta.get("heatmap_overlays_path")
        if heatmap_overlays_path:
            try:
                heatmap_overlays = _load_heatmap_overlays(cached_dir / Path(heatmap_overlays_path).name)
            except (OSError, ValueError) as e:
                logger.warning("Could not load cached PatchCore heatmaps: %s", e)
                heatmap_overlays = {}
        else:
            # Backward compatibility for caches written before heatmaps moved to NPZ.
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
                "auroc": float(meta["image_auroc"]),
                "f1_score": manual_image_f1,
                "precision": manual_image_prec,
                "recall": manual_image_rec,
                "threshold": img_threshold,
                "true_positives": int(meta["true_positives"]),
                "false_positives": int(meta["false_positives"]),
                "false_negatives": int(meta["false_negatives"]),
                "true_negatives": int(meta["true_negatives"]),
                "metrics_path": str(image_file),
            },
            "pixel_level": {
                "auroc": float(meta["pixel_auroc"]),
                "f1_score": manual_pixel_f1,
                "aupimo_score": aupimo,
                "fpr_lower_bound": 1e-5,
                "fpr_upper_bound": fpr_limit,
                "aupimo": aupimo,
                "anomaly_map_min": float(meta.get("anomaly_map_min", 0.0)),
                "anomaly_map_max": float(meta.get("anomaly_map_max", 0.0)),
                "anomaly_map_range": float(meta.get("anomaly_map_range", 0.0)),
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

    # A rejected or explicitly bypassed cache must never be overwritten. New
    # fair-protocol artifacts always use their protocol-aware computed identity.
    effective_hash = computed_hash
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
        val_split_mode="none",
    )
    _configure_patchcore_partitions(
        datamodule,
        fair_split,
        transform_adapter if len(proc_pipeline) > 0 else None,
    )

    # PatchCore's coreset is sampled stochastically. Seed immediately before
    # construction and fitting so identical protocol inputs produce the same model.
    _seed_patchcore_run(model_seed)
    model = Patchcore(
        backbone=backbone,
        layers=feature_layers,
        coreset_sampling_ratio=coreset_sampling_ratio,
        num_neighbors=num_neighbors,
        post_processor=False,
        evaluator=False,
        visualizer=_RawScoreImageVisualizer(),
    )
    engine = Engine(accelerator="gpu", devices=1, deterministic=True)

    # 2. Fit. Evaluation below uses one explicit raw-score prediction path;
    # engine.test() is intentionally avoided because it invokes model callbacks.
    logger.info("Fitting Patchcore model on %s category (Hash: %s)...", category, effective_hash)
    train_dataloader = datamodule.train_dataloader()
    validation_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    engine.fit(model, train_dataloaders=train_dataloader)

    # 3. Extract PR metrics and build summary
    (
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
        true_positives,
        false_positives,
        false_negatives,
        true_negatives,
        image_auroc,
    ) = extract_and_save_pr_metrics(
        engine,
        model,
        validation_dataloader,
        test_dataloader,
        base_dir,
        run_heatmap,
    )

    # Extract dataset split counts if datamodule was setup
    split_info = {
        **cache_evidence,
        "test_normal": int((fair_split.test["is_anomaly"] == 0).sum()),
        "test_anomalous": int(fair_split.test["is_anomaly"].astype(bool).sum()),
    }

    hyperparams = {
        "backbone": backbone,
        "feature_layers": feature_layers,
        "coreset_sampling_ratio": coreset_sampling_ratio,
        "num_neighbors": num_neighbors,
        "fpr_limit": fpr_limit,
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "model_seed": model_seed,
        "score_space": PATCHCORE_SCORE_SPACE,
    }

    raw_results_dict = {
        "image_AUROC": image_auroc,
        "image_F1Score": manual_image_f1,
        "image_Precision": manual_image_prec,
        "image_Recall": manual_image_rec,
        "pixel_AUROC": pixel_auroc,
        "pixel_F1Score": manual_pixel_f1,
        "pixel_AUPIMO": pixel_aupimo,
    }
    test_results: list[Mapping[str, float]] = [raw_results_dict]
    _print_patchcore_results_table(raw_results_dict)

    logger.info(
        "PatchCore evaluation summary | image: AUROC=%.6f F1=%.6f precision=%.6f recall=%.6f threshold=%.6f",
        image_auroc,
        manual_image_f1,
        manual_image_prec,
        manual_image_rec,
        img_threshold,
    )
    logger.info(
        "PatchCore evaluation summary | confusion: TP=%d FP=%d FN=%d TN=%d",
        true_positives,
        false_positives,
        false_negatives,
        true_negatives,
    )
    logger.info(
        "PatchCore evaluation summary | pixel: AUROC=%.6f F1=%.6f AUPIMO=%.6f",
        pixel_auroc,
        manual_pixel_f1,
        pixel_aupimo,
    )

    heatmap_archive = _save_heatmap_overlays(heatmap_overlays, base_dir / "heatmap_overlays.npz")
    if heatmap_archive is not None:
        logger.info("Saved compressed PatchCore heatmaps to %s", heatmap_archive)

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
        "protocol": FAIR_EVALUATION_PROTOCOL,
        "threshold_source": "normal_validation",
        "image_threshold_quantile": PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
        "pixel_threshold_quantile": PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
        "model_seed": model_seed,
        "score_space": PATCHCORE_SCORE_SPACE,
        "pixel_metrics_version": PIXEL_METRICS_VERSION,
        "image_auroc": raw_results_dict["image_AUROC"],
        "pixel_auroc": pixel_auroc,
        "manual_image_f1": manual_image_f1,
        "manual_pixel_f1": manual_pixel_f1,
        "manual_image_prec": manual_image_prec,
        "manual_image_rec": manual_image_rec,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "img_threshold": img_threshold,
        "pixel_aupimo": pixel_aupimo,
        "aupimo_fpr_bounds": [1e-5, fpr_limit],
        "aupimo_num_thresholds": AUPIMO_NUM_THRESHOLDS,
        "canonical_height": CANONICAL_MAP_SIZE[0],
        "canonical_width": CANONICAL_MAP_SIZE[1],
        "anomaly_map_min": anomaly_map_min,
        "anomaly_map_max": anomaly_map_max,
        "anomaly_map_range": anomaly_map_range,
        "heatmap_overlays_path": heatmap_archive.name if heatmap_archive is not None else None,
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
        pixel_auroc=pixel_auroc,
        pixel_aupimo=pixel_aupimo,
        anomaly_map_min=anomaly_map_min,
        anomaly_map_max=anomaly_map_max,
        anomaly_map_range=anomaly_map_range,
        heatmap_overlays=heatmap_overlays,
        anomalous_indices=anomalous_indices,
        fpr_limit=fpr_limit,
        preprocessing_steps=raw_prep_list,
        hyperparameters=hyperparams,
        dataset_split=split_info,
        model_hash=effective_hash,
        metadata=metadata,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
    )


if __name__ == "__main__":
    run_baseline()
