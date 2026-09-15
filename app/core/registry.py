"""Safe lifecycle operations for cached model evaluation artifacts."""

import json
import shutil
from pathlib import Path
from typing import Any

from app.core.logger import logger


def delete_cached_model(model_hash: str, registry_base: str | Path, soft_delete: bool = True) -> bool:
    """Delete or move one cached model artifact directory to trash."""
    if not isinstance(model_hash, str) or len(model_hash) < 4:
        return False
    base_path = Path(registry_base).resolve()
    target_dir = (base_path / model_hash).resolve()
    if (
        not base_path.exists()
        or not target_dir.is_relative_to(base_path)
        or target_dir == base_path
        or target_dir.name == ".trash"
        or not target_dir.is_dir()
    ):
        return False
    if soft_delete:
        trash_dir = base_path / ".trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        destination = trash_dir / model_hash
        if destination.exists():
            logger.warning("Refusing to overwrite existing trash entry: %s", destination)
            return False
        shutil.move(str(target_dir), str(destination))
        logger.info("Moved cached artifacts to trash: %s -> %s", target_dir, destination)
    else:
        shutil.rmtree(target_dir)
        logger.info("Permanently deleted cached artifacts: %s", target_dir)
    return True


def restore_cached_model(model_hash: str, registry_base: str | Path) -> bool:
    """Restore one soft-deleted artifact directory."""
    if not isinstance(model_hash, str) or len(model_hash) < 4:
        return False
    base_path = Path(registry_base).resolve()
    source = (base_path / ".trash" / model_hash).resolve()
    destination = (base_path / model_hash).resolve()
    if not source.is_relative_to(base_path / ".trash") or not source.is_dir():
        return False
    if destination.exists():
        logger.warning("Refusing to overwrite active run while restoring: %s", destination)
        return False
    shutil.move(str(source), str(destination))
    logger.info("Restored cached artifacts from trash: %s -> %s", source, destination)
    return True


def list_trashed_models(registry_base: str | Path) -> list[dict[str, Any]]:
    """Return metadata for all artifact directories in trash."""
    trash_dir = Path(registry_base).resolve() / ".trash"
    if not trash_dir.exists():
        return []
    trashed: list[dict[str, Any]] = []
    for metadata_path in trash_dir.rglob("metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["hash"] = metadata.get("hash", metadata_path.parent.name)
            trashed.append(metadata)
        except (OSError, json.JSONDecodeError):
            trashed.append({"hash": metadata_path.parent.name})
    return trashed


def purge_trash(registry_base: str | Path, model_hash: str | None = None) -> int:
    """Permanently remove cached artifact directorie(s) in trash.

    Args:
        registry_base: Base directory for the registry.
        model_hash: Optional specific model hash to purge. If None, all trash is purged.
    """
    trash_dir = Path(registry_base).resolve() / ".trash"
    if not trash_dir.exists():
        return 0
    purged = 0
    if model_hash is not None:
        target = trash_dir / model_hash
        if target.exists() and target.is_dir():
            shutil.rmtree(target)
            purged += 1
            logger.info("Purged specific cached run from trash: %s", model_hash)
    else:
        for child in trash_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
                purged += 1
        logger.info("Emptied trash: purged %d cached run(s).", purged)
    return purged


__all__ = [
    "delete_cached_model",
    "list_trashed_models",
    "purge_trash",
    "restore_cached_model",
]
