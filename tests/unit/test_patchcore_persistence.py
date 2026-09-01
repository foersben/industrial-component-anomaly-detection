"""Unit tests verifying Patchcore model persistence, caching, and trash lifecycle."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from app.pipelines.modelling.baseline import (
    _load_heatmap_overlays,
    _save_heatmap_overlays,
    delete_cached_patchcore_model,
    extract_and_save_pr_metrics,
    find_cached_patchcore_model,
    list_trashed_patchcore_models,
    purge_patchcore_trash,
    restore_cached_patchcore_model,
    run_baseline,
)


def test_patchcore_heatmaps_use_compressed_binary_archive(tmp_path: Path) -> None:
    """Verify dense heatmap arrays round-trip through NPZ without JSON serialization."""
    overlays = {
        7: {
            "heatmap": np.arange(24, dtype=np.uint8).reshape(2, 4, 3).tolist(),
            "gt_and_heatmap": np.full((2, 4, 3), 127, dtype=np.uint8).tolist(),
        }
    }

    archive_path = tmp_path / "heatmap_overlays.npz"
    assert _save_heatmap_overlays(overlays, archive_path) == archive_path
    assert _load_heatmap_overlays(archive_path) == overlays

    with np.load(archive_path, allow_pickle=False) as archive:
        assert archive["prediction__7"].dtype == np.uint8
        assert archive["ground_truth__7"].dtype == np.uint8


def test_patchcore_aupimo_receives_full_maps(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Verify PatchCore sends full 2D maps and masks through the shared AUPIMO path."""
    anomaly_maps = torch.tensor(
        [
            [[[0.0, 0.1], [0.2, 0.3]]],
            [[[0.4, 0.5], [0.6, 0.7]]],
        ],
        dtype=torch.float32,
    )
    gt_masks = torch.tensor(
        [
            [[[0, 0], [0, 0]]],
            [[[0, 0], [1, 1]]],
        ],
        dtype=torch.uint8,
    )
    batch = SimpleNamespace(
        anomaly_map=anomaly_maps,
        gt_mask=gt_masks,
        pred_score=torch.tensor([0.2, 0.8]),
        gt_label=torch.tensor([0, 1]),
    )
    engine = SimpleNamespace(predict=lambda **_kwargs: [batch])
    datamodule = SimpleNamespace(test_dataloader=lambda: object())
    captured: dict[str, Any] = {}

    def fake_compute_aupimo(maps: list[np.ndarray], masks: list[np.ndarray | None], fpr_bounds: Any) -> float:
        captured["maps"] = maps
        captured["masks"] = masks
        captured["bounds"] = fpr_bounds
        return 0.42

    monkeypatch.setattr("app.pipelines.modelling.baseline.compute_aupimo", fake_compute_aupimo)

    result = extract_and_save_pr_metrics(engine, object(), datamodule, tmp_path)

    assert [item.shape for item in captured["maps"]] == [(2, 2), (2, 2)]
    assert [item.shape for item in captured["masks"]] == [(2, 2), (2, 2)]
    assert captured["bounds"] == (1e-5, 1e-4)
    assert np.allclose(result[5:9], (0.42, 0.0, 0.7, 0.7))
    with np.load(tmp_path / "pixel_metrics.npz") as pixel_metrics:
        assert float(pixel_metrics["aupimo"]) == 0.42
        assert np.array_equal(pixel_metrics["aupimo_fpr_bounds"], np.array([1e-5, 1e-4]))


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
    overlays = {3: {"heatmap": np.zeros((2, 2, 3), dtype=np.uint8).tolist()}}
    _save_heatmap_overlays(overlays, model_dir / "heatmap_overlays.npz")

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
        "heatmap_overlays_path": "heatmap_overlays.npz",
        "anomalous_indices": [3],
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
    assert result["heatmap_overlays"] == overlays


def test_run_baseline_loads_legacy_inline_heatmaps(tmp_path: Path) -> None:
    """Verify caches from before the NPZ migration retain their heatmap behavior."""
    model_dir = tmp_path / "legacy_run"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez(model_dir / "image_metrics.npz", auroc=0.99)
    np.savez(model_dir / "pixel_metrics.npz", aupimo=0.92)
    legacy_overlays = {"3": {"heatmap": np.zeros((2, 2, 3), dtype=np.uint8).tolist()}}
    metadata = {
        "hash": "legacy_run",
        "category": "bottle",
        "backbone": "resnet18",
        "feature_layers": ["layer2", "layer3"],
        "coreset_sampling_ratio": 0.1,
        "num_neighbors": 9,
        "fpr_limit": 1e-4,
        "preprocessing_steps": [],
        "heatmap_overlays": legacy_overlays,
        "anomalous_indices": [3],
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    result = run_baseline(
        data_root=tmp_path,
        category="bottle",
        model_hash="legacy_run",
        registry_base=tmp_path,
    )

    assert result["heatmap_overlays"] == legacy_overlays
    assert result["anomalous_indices"] == [3]
