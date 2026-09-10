"""Model registry, disk caching, and soft-delete lifecycle management for PatchCore models."""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from app.pipelines.preprocessing.factory import normalize_preprocessing_steps

logger = logging.getLogger(__name__)

# Re-export for internal and test compatibility
_normalize_preprocessing_steps = normalize_preprocessing_steps


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
    """Find the newest cached PatchCore model matching either a specific hash or the given parameters.

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

    norm_req_prep = normalize_preprocessing_steps(preprocessing_steps)
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
        if abs(float(meta.get("coreset_sampling_ratio", 0.1)) - coreset_sampling_ratio) > 1e-5:
            continue
        if int(meta.get("num_neighbors", 9)) != num_neighbors:
            continue
        if abs(float(meta.get("fpr_limit", 1e-4)) - fpr_limit) > 1e-6:
            continue

        meta_prep = normalize_preprocessing_steps(meta.get("preprocessing_steps"))
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
    """Safely delete a cached PatchCore model directory from the model registry.

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
    """Restore a previously soft-deleted PatchCore model from the .trash/ recovery directory.

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


def list_trashed_patchcore_models(
    registry_base: str | Path = "data/models/patchcore",
) -> list[dict[str, Any]]:
    """List all PatchCore models currently held in the .trash/ recovery directory.

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


__all__ = [
    "_normalize_preprocessing_steps",
    "delete_cached_patchcore_model",
    "find_cached_patchcore_model",
    "list_trashed_patchcore_models",
    "normalize_preprocessing_steps",
    "purge_patchcore_trash",
    "restore_cached_patchcore_model",
]
