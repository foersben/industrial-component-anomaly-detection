"""Safe lifecycle operations for cached DINO evaluation artifacts."""

import json
import shutil
from pathlib import Path
from typing import Any

from app.core.logger import logger


def delete_cached_dino_model(model_hash: str, registry_base: str | Path, soft_delete: bool = True) -> bool:
    """Delete or move one cached DINO artifact directory to trash."""
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
            logger.warning("Refusing to overwrite existing DINO trash entry: %s", destination)
            return False
        shutil.move(str(target_dir), str(destination))
        logger.info("Moved cached DINO artifacts to trash: %s -> %s", target_dir, destination)
    else:
        shutil.rmtree(target_dir)
        logger.info("Permanently deleted cached DINO artifacts: %s", target_dir)
    return True


def restore_cached_dino_model(model_hash: str, registry_base: str | Path) -> bool:
    """Restore one soft-deleted DINO artifact directory."""
    if not isinstance(model_hash, str) or len(model_hash) < 4:
        return False
    base_path = Path(registry_base).resolve()
    source = (base_path / ".trash" / model_hash).resolve()
    destination = (base_path / model_hash).resolve()
    if not source.is_relative_to(base_path / ".trash") or not source.is_dir():
        return False
    if destination.exists():
        logger.warning("Refusing to overwrite active DINO run while restoring: %s", destination)
        return False
    shutil.move(str(source), str(destination))
    logger.info("Restored cached DINO artifacts from trash: %s -> %s", source, destination)
    return True


def list_trashed_dino_models(registry_base: str | Path) -> list[dict[str, Any]]:
    """Return metadata for all DINO artifact directories in trash."""
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


def purge_dino_trash(registry_base: str | Path) -> int:
    """Permanently remove every cached DINO artifact directory in trash."""
    trash_dir = Path(registry_base).resolve() / ".trash"
    if not trash_dir.exists():
        return 0
    purged = 0
    for child in trash_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
            purged += 1
    logger.info("Emptied DINO trash: purged %d cached run(s).", purged)
    return purged


__all__ = [
    "delete_cached_dino_model",
    "list_trashed_dino_models",
    "purge_dino_trash",
    "restore_cached_dino_model",
]
