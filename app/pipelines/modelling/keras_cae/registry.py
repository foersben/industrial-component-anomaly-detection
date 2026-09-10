"""Model registry, disk caching, and soft-delete lifecycle management for Keras CAE models."""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from app.pipelines.preprocessing.factory import normalize_preprocessing_steps

logger = logging.getLogger(__name__)

# Re-export for internal and backward compatibility
_normalize_preprocessing_steps = normalize_preprocessing_steps


def find_cached_model(
    category: str,
    img_size: int,
    crop_size: int = 64,
    crop_stride: int = 32,
    latent_channels: int = 32,
    epochs: int = 20,
    batch_size: int = 16,
    mask_ratio: float = 0.25,
    mask_patch_size: int = 8,
    preprocessing_steps: list[dict[str, Any]] | None = None,
    target_hash: str | None = None,
    registry_base: Path | str = "data/models/keras_cae",
    expected_split_evidence: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]] | None:
    """Find the newest cached model matching either a specific hash or the given hyperparameters.

    Args:
        category: Component category name.
        img_size: Spatial image dimension.
        crop_size: Crop dimension.
        crop_stride: Crop sliding stride.
        latent_channels: Bottleneck latent channels.
        epochs: Number of epochs.
        batch_size: Batch size.
        mask_ratio: Mask ratio for MIM.
        mask_patch_size: Mask patch size for MIM.
        preprocessing_steps: Optional preprocessing step configurations.
        target_hash: Optional exact model hash to search for.
        registry_base: Path to the keras_cae model registry.
        expected_split_evidence: Required fair-protocol split evidence, when evaluating a cache hit.

    Returns:
        Tuple of (model_dir, metadata_dict) if found, else None.
    """
    base_path = Path(registry_base)
    if not base_path.exists():
        return None

    # 1. If explicit target_hash is given, look for exact directory
    if target_hash:
        target_dir = base_path / target_hash
        model_file = target_dir / "model.keras"
        meta_file = target_dir / "metadata.json"
        if model_file.exists():
            meta: dict[str, Any] = {}
            if meta_file.exists():
                try:
                    with open(meta_file, encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass
            if expected_split_evidence is None or all(
                meta.get("dataset_split", {}).get(key) == value for key, value in expected_split_evidence.items()
            ):
                return target_dir, meta
            return None

    norm_req_prep = normalize_preprocessing_steps(preprocessing_steps)

    # 2. Otherwise, search all models and find those matching hyperparameters
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    for meta_file in base_path.rglob("metadata.json"):
        if ".trash" in meta_file.parts:
            continue
        model_dir = meta_file.parent
        model_file = model_dir / "model.keras"
        if not model_file.exists():
            continue

        try:
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        if expected_split_evidence is not None and not all(
            meta.get("dataset_split", {}).get(key) == value for key, value in expected_split_evidence.items()
        ):
            continue

        # Check parameter compatibility
        if meta.get("category") != category:
            continue
        if meta.get("img_size") != img_size:
            continue
        meta_latent = meta.get("latent_channels", meta.get("latent_dim", 32))
        if meta_latent != latent_channels:
            continue
        if meta.get("epochs") != epochs:
            continue
        if meta.get("batch_size") != batch_size:
            continue
        if abs(float(meta.get("mask_ratio", 0.25)) - mask_ratio) > 1e-3:
            continue
        if meta.get("crop_size", 64) != crop_size:
            continue
        if meta.get("crop_stride", 32) != crop_stride:
            continue
        if meta.get("mask_patch_size", 8) != mask_patch_size:
            continue

        # Check preprocessing steps compatibility
        meta_prep = normalize_preprocessing_steps(meta.get("preprocessing_steps"))
        if meta_prep != norm_req_prep:
            continue

        # Determine timestamp / mtime for sorting newest first
        ts_str = meta.get("timestamp", "")
        try:
            if ts_str:
                dt = datetime.fromisoformat(ts_str)
                ts = dt.timestamp()
            else:
                ts = model_file.stat().st_mtime
        except Exception:
            ts = model_file.stat().st_mtime

        candidates.append((ts, model_dir, meta))

    if not candidates:
        return None

    # Sort descending by timestamp (newest first)
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, newest_dir, newest_meta = candidates[0]
    return newest_dir, newest_meta


def delete_cached_model(
    model_hash: str,
    registry_base: str | Path = "data/models/keras_cae",
    soft_delete: bool = True,
) -> bool:
    """Safely delete a cached model directory from the model registry.

    When soft_delete is True (default), moves the model to a .trash/ recovery directory,
    enabling non-destructive undo and restore operations.

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
        logger.info("Moved cached model directory to trash: %s -> %s", target_dir, dest_dir)
        return True

    shutil.rmtree(target_dir)
    logger.info("Permanently deleted cached model directory: %s", target_dir)
    return True


def restore_cached_model(
    model_hash: str,
    registry_base: str | Path = "data/models/keras_cae",
) -> bool:
    """Restore a previously soft-deleted model from the .trash/ recovery directory.

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
    logger.info("Restored model directory from trash: %s -> %s", source_dir, dest_dir)
    return True


def list_trashed_models(
    registry_base: str | Path = "data/models/keras_cae",
) -> list[dict[str, Any]]:
    """List all models currently held in the .trash/ recovery directory.

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


def purge_trash(
    registry_base: str | Path = "data/models/keras_cae",
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
            purged_count += 1
    else:
        for child in list(trash_dir.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
                purged_count += 1
    return purged_count


__all__ = [
    "delete_cached_model",
    "find_cached_model",
    "list_trashed_models",
    "normalize_preprocessing_steps",
    "purge_trash",
    "restore_cached_model",
]
