"""Model registry, disk caching, and soft-delete lifecycle management for PatchCore models."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from app.pipelines.preprocessing.base import PreprocessingPipeline
from app.pipelines.preprocessing.factory import normalize_preprocessing_steps

logger = logging.getLogger(__name__)

# Re-export for internal and test compatibility
_normalize_preprocessing_steps = normalize_preprocessing_steps


def _find_target_hash_cached_dir(
    base_path: Path,
    target_hash: str,
    expected_split_evidence: dict[str, Any] | None,
) -> tuple[Path, dict[str, Any]] | None:
    """Locate a cached PatchCore model directory by its exact hash identifier.

    Args:
        base_path: Root registry path.
        target_hash: Exact directory name hash to find.
        expected_split_evidence: Optional dictionary of expected split partition fingerprints.

    Returns:
        Tuple of (model_dir, metadata_dict) if found and valid, otherwise None.
    """
    target_dir = base_path / target_hash
    meta_file = target_dir / "metadata.json"
    if not (target_dir.exists() and meta_file.exists()):
        return None

    try:
        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)
        if expected_split_evidence is None or all(
            meta.get("dataset_split", {}).get(key) == value for key, value in expected_split_evidence.items()
        ):
            return target_dir, meta
    except Exception:
        pass
    return None


def _matches_patchcore_metadata(
    meta: dict[str, Any],
    category: str,
    backbone: str,
    feature_layers: tuple[str, ...],
    coreset_sampling_ratio: float,
    num_neighbors: int,
    fpr_limit: float,
    norm_req_prep: list[dict[str, Any]],
    expected_split_evidence: dict[str, Any] | None,
) -> bool:
    """Verify whether a candidate model's metadata matches the desired PatchCore parameters.

    Args:
        meta: Parsed metadata dictionary from a candidate directory.
        category: Component category name.
        backbone: Feature extractor backbone name.
        feature_layers: Tuple of layer names to extract features from.
        coreset_sampling_ratio: Coreset subsampling fraction.
        num_neighbors: Number of nearest neighbors.
        fpr_limit: Max allowable False Positive Rate.
        norm_req_prep: Normalized list of preprocessing steps.
        expected_split_evidence: Optional dictionary of expected split partition fingerprints.

    Returns:
        True if all model parameters, preprocessing, and split evidence match; False otherwise.
    """
    if expected_split_evidence is not None and not all(
        meta.get("dataset_split", {}).get(key) == value for key, value in expected_split_evidence.items()
    ):
        return False

    if meta.get("category") != category:
        return False
    if meta.get("backbone", "resnet18") != backbone:
        return False
    if tuple(meta.get("feature_layers", ["layer2", "layer3"])) != tuple(feature_layers):
        return False
    if abs(float(meta.get("coreset_sampling_ratio", 0.1)) - coreset_sampling_ratio) > 1e-5:
        return False
    if int(meta.get("num_neighbors", 9)) != num_neighbors:
        return False
    if abs(float(meta.get("fpr_limit", 1e-4)) - fpr_limit) > 1e-6:
        return False

    meta_prep = normalize_preprocessing_steps(meta.get("preprocessing_steps"))
    return meta_prep == norm_req_prep


def _extract_metadata_timestamp(meta: dict[str, Any], meta_file: Path) -> float:
    """Extract a sortable timestamp from model metadata or filesystem stats.

    Args:
        meta: Metadata dictionary loaded from disk.
        meta_file: Path to the metadata.json file.

    Returns:
        POSIX timestamp as float.
    """
    ts_str = meta.get("timestamp", "")
    if ts_str:
        try:
            return datetime.fromisoformat(ts_str).timestamp()
        except Exception:
            pass
    try:
        return meta_file.stat().st_mtime
    except Exception:
        return 0.0


def find_cached_patchcore_model(
    category: str,
    backbone: str = "resnet18",
    feature_layers: tuple[str, ...] = ("layer2", "layer3"),
    coreset_sampling_ratio: float = 0.1,
    num_neighbors: int = 9,
    fpr_limit: float = 1e-4,
    pipeline: list[dict[str, Any]] | None = None,
    target_hash: str | None = None,
    registry_base: Path | str = "data/models/patchcore",
    expected_split_evidence: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]] | None:
    """Find the newest cached PatchCore model matching either a specific hash or the given parameters.

    Args:
        category: Component category name.
        backbone: Feature extractor backbone name.
        feature_layers: Layers to extract features from.
        coreset_sampling_ratio: Ratio for coreset subsampling.
        num_neighbors: Number of nearest neighbors for scoring.
        fpr_limit: Max allowable False Positive Rate.
        pipeline: Optional preprocessing pipeline configuration.
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
        return _find_target_hash_cached_dir(base_path, target_hash, expected_split_evidence)

    if isinstance(pipeline, PreprocessingPipeline):
        norm_req_prep: list[dict[str, Any]] = []
    else:
        norm_req_prep = _normalize_preprocessing_steps(pipeline)
    candidates: list[tuple[float, Path, dict[str, Any]]] = []

    for meta_file in base_path.rglob("metadata.json"):
        if ".trash" in meta_file.parts:
            continue
        try:
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        if not _matches_patchcore_metadata(
            meta=meta,
            category=category,
            backbone=backbone,
            feature_layers=feature_layers,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
            fpr_limit=fpr_limit,
            norm_req_prep=norm_req_prep,
            expected_split_evidence=expected_split_evidence,
        ):
            continue

        ts = _extract_metadata_timestamp(meta, meta_file)
        candidates.append((ts, meta_file.parent, meta))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, newest_dir, newest_meta = candidates[0]
    return newest_dir, newest_meta
