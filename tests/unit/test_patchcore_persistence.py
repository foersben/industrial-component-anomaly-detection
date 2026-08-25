"""Unit tests verifying Patchcore model persistence, caching, and trash lifecycle."""

import json
from pathlib import Path

import numpy as np

from app.pipelines.modelling.baseline import (
    delete_cached_patchcore_model,
    find_cached_patchcore_model,
    list_trashed_patchcore_models,
    purge_patchcore_trash,
    restore_cached_patchcore_model,
    run_baseline,
)


def test_find_cached_patchcore_model_resolution(tmp_path: Path) -> None:
    """Verify that find_cached_patchcore_model accurately identifies matching models in registry.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    model_dir = tmp_path / "patch_123"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez(model_dir / "image_metrics.npz", auroc=0.98)
    np.savez(model_dir / "pixel_metrics.npz", aupimo=0.85, t_aupimo_min=0.4)

    meta = {
        "hash": "patch_123",
        "category": "bottle",
        "backbone": "resnet18",
        "coreset_sampling_ratio": 0.1,
        "fpr_limit": 1e-4,
        "preprocessing_steps": [{"name": "clahe", "params": {}}],
        "timestamp": "2026-08-22T10:00:00",
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # 1. Search with matching parameters
    hit = find_cached_patchcore_model(
        category="bottle",
        backbone="resnet18",
        coreset_sampling_ratio=0.1,
        fpr_limit=1e-4,
        preprocessing_steps=[{"name": "clahe", "params": {}}],
        registry_base=tmp_path,
    )
    assert hit is not None
    found_dir, found_meta = hit
    assert found_dir == model_dir
    assert found_meta["hash"] == "patch_123"

    # 2. Search with target_hash
    hit_hash = find_cached_patchcore_model(
        category="bottle",
        target_hash="patch_123",
        registry_base=tmp_path,
    )
    assert hit_hash is not None
    assert hit_hash[1]["hash"] == "patch_123"

    # 3. Search with non-matching parameters
    miss = find_cached_patchcore_model(
        category="leather",
        backbone="resnet18",
        registry_base=tmp_path,
    )
    assert miss is None


def test_patchcore_model_soft_delete_and_restore_lifecycle(tmp_path: Path) -> None:
    """Verify soft-delete moves model to .trash/ and restore recovers it.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    model_dir = tmp_path / "model_del"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metadata.json").write_text(json.dumps({"hash": "model_del", "category": "bottle"}))

    # 1. Soft delete
    assert delete_cached_patchcore_model("model_del", registry_base=tmp_path, soft_delete=True)
    assert not model_dir.exists()
    assert (tmp_path / ".trash" / "model_del" / "metadata.json").exists()

    # 2. List trashed models
    trashed = list_trashed_patchcore_models(registry_base=tmp_path)
    assert len(trashed) == 1
    assert trashed[0]["hash"] == "model_del"

    # 3. Restore
    assert restore_cached_patchcore_model("model_del", registry_base=tmp_path)
    assert model_dir.exists()
    assert (model_dir / "metadata.json").exists()
    assert not (tmp_path / ".trash" / "model_del").exists()


def test_patchcore_purge_trash(tmp_path: Path) -> None:
    """Verify purge_patchcore_trash permanently empties the trash.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    trash_dir = tmp_path / ".trash" / "model_trash_1"
    trash_dir.mkdir(parents=True, exist_ok=True)
    (trash_dir / "metadata.json").write_text(json.dumps({"hash": "model_trash_1"}))

    assert len(list_trashed_patchcore_models(registry_base=tmp_path)) == 1

    purged = purge_patchcore_trash(registry_base=tmp_path)
    assert purged == 1
    assert len(list_trashed_patchcore_models(registry_base=tmp_path)) == 0


def test_run_baseline_cached_loading(tmp_path: Path) -> None:
    """Verify run_baseline directly loads cached results when matching model exists.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    model_dir = tmp_path / "cached_run"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez(model_dir / "image_metrics.npz", auroc=0.99)
    np.savez(model_dir / "pixel_metrics.npz", aupimo=0.92, t_aupimo_min=0.35)

    meta = {
        "hash": "cached_run",
        "category": "bottle",
        "backbone": "resnet18",
        "coreset_sampling_ratio": 0.1,
        "fpr_limit": 1e-4,
        "preprocessing_steps": [],
        "dataset_split": {"train_normal": 209, "test_total": 83},
        "manual_image_f1": 0.95,
        "manual_pixel_f1": 0.88,
        "img_threshold": 0.42,
        "timestamp": "2026-08-22T12:00:00",
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    result = run_baseline(
        data_root=tmp_path,
        category="bottle",
        backbone="resnet18",
        coreset_sampling_ratio=0.1,
        fpr_limit=1e-4,
        preprocessing_steps=[],
        registry_base=tmp_path,
    )

    assert result["category"] == "bottle"
    assert result["model_hash"] == "cached_run"
    assert result["image_level"]["f1_score"] == 0.95
    assert result["pixel_level"]["f1_score"] == 0.88
    assert result["dataset_split"]["train_normal"] == 209
