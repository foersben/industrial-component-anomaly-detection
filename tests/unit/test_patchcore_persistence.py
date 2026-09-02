"""Unit tests verifying Patchcore model persistence, caching, and trash lifecycle."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from app.domain.data import build_fair_evaluation_split
from app.pipelines.evaluation.metrics import fair_metric_evidence
from app.pipelines.modelling.baseline import (
    PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
    PATCHCORE_MODEL_SEED,
    PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
    PATCHCORE_SCORE_SPACE,
    _add_panel_headers,
    _configure_patchcore_partitions,
    _load_heatmap_overlays,
    _save_heatmap_overlays,
    _seed_patchcore_run,
    delete_cached_patchcore_model,
    extract_and_save_pr_metrics,
    find_cached_patchcore_model,
    list_trashed_patchcore_models,
    purge_patchcore_trash,
    restore_cached_patchcore_model,
    run_baseline,
)


def test_patchcore_visualization_headers_do_not_cover_panels() -> None:
    """Four-panel labels live in a separate header instead of obscuring image data."""
    grid = Image.new("RGB", (1024, 256), (12, 34, 56))

    labelled = _add_panel_headers(
        grid,
        ["Image", "Ground-Truth Mask", "Anomaly Map Overlay", "Predicted Mask"],
    )

    assert labelled.size == (1024, 290)
    assert np.array_equal(np.asarray(labelled)[34:], np.asarray(grid))


class _SamplesDataset:
    """Minimal Anomalib-like dataset exposing an ordered samples frame."""

    def __init__(self, paths: list[str]) -> None:
        self.samples = pd.DataFrame({"image_path": paths})

    def __len__(self) -> int:
        """Return the number of sample rows."""
        return len(self.samples)


def _small_protocol_manifest(tmp_path: Path) -> pd.DataFrame:
    """Create a small manifest suitable for split and partition unit tests."""
    return pd.DataFrame(
        [
            {
                "path": str(tmp_path / "bottle" / "train" / "good" / f"{index:03}.png"),
                "product": "bottle",
                "split": "train",
                "is_anomaly": False,
            }
            for index in range(20)
        ]
        + [
            {
                "path": str(tmp_path / "bottle" / "test" / "good" / "000.png"),
                "product": "bottle",
                "split": "test",
                "is_anomaly": False,
            },
            {
                "path": str(tmp_path / "bottle" / "test" / "broken" / "001.png"),
                "product": "bottle",
                "split": "test",
                "is_anomaly": True,
            },
        ]
    )


def test_patchcore_partitions_use_only_shared_protocol_paths(tmp_path: Path) -> None:
    """PatchCore fitting excludes every shared validation and official test path."""
    manifest = _small_protocol_manifest(tmp_path)
    split = build_fair_evaluation_split(manifest, "bottle")
    datamodule = SimpleNamespace(
        setup=lambda: None,
        train_data=_SamplesDataset(split.fitting_paths + split.validation_paths),
        test_data=_SamplesDataset(split.test_paths),
    )

    _configure_patchcore_partitions(datamodule, split)

    train_paths = datamodule.train_data.samples["image_path"].tolist()
    validation_paths = datamodule.val_data.samples["image_path"].tolist()
    test_paths = datamodule.test_data.samples["image_path"].tolist()
    assert train_paths == split.fitting_paths
    assert validation_paths == split.validation_paths
    assert test_paths == split.test_paths
    assert not set(train_paths) & (set(validation_paths) | set(test_paths))


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
    captured: dict[str, Any] = {}

    def fake_compute_aupimo(maps: list[np.ndarray], masks: list[np.ndarray | None], fpr_bounds: Any) -> float:
        captured["maps"] = maps
        captured["masks"] = masks
        captured["bounds"] = fpr_bounds
        return 0.42

    monkeypatch.setattr("app.pipelines.evaluation.cae_metrics.compute_aupimo", fake_compute_aupimo)

    result = extract_and_save_pr_metrics(engine, object(), object(), object(), tmp_path)

    assert [item.shape for item in captured["maps"]] == [(256, 256), (256, 256)]
    assert [item.shape for item in captured["masks"]] == [(256, 256), (256, 256)]
    assert captured["bounds"] == (1e-5, 1e-4)
    assert np.allclose(result[6:10], (0.42, 0.0, 0.7, 0.7))
    with np.load(tmp_path / "pixel_metrics.npz") as pixel_metrics:
        assert float(pixel_metrics["aupimo"]) == 0.42
        assert np.array_equal(pixel_metrics["aupimo_fpr_bounds"], np.array([1e-5, 1e-4]))


def test_patchcore_threshold_depends_only_on_validation_scores(tmp_path: Path, monkeypatch: Any) -> None:
    """Changing official test scores cannot alter the frozen validation threshold."""
    validation_loader = object()
    test_loader = object()
    validation_batch = SimpleNamespace(
        anomaly_map=torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]], dtype=torch.float32),
        pred_score=torch.tensor([0.25]),
    )

    def test_batch(scores: list[float]) -> SimpleNamespace:
        return SimpleNamespace(
            anomaly_map=torch.tensor(
                [
                    [[[0.0, 0.1], [0.1, 0.0]]],
                    [[[0.2, 0.4], [0.8, 1.0]]],
                ],
                dtype=torch.float32,
            ),
            gt_mask=torch.tensor(
                [
                    [[[0, 0], [0, 0]]],
                    [[[0, 0], [1, 1]]],
                ],
                dtype=torch.uint8,
            ),
            pred_score=torch.tensor(scores),
            gt_label=torch.tensor([0, 1]),
        )

    class FakeEngine:
        def __init__(self, scores: list[float]) -> None:
            self.scores = scores

        def predict(self, **kwargs: Any) -> list[SimpleNamespace]:
            return [validation_batch] if kwargs["dataloaders"] is validation_loader else [test_batch(self.scores)]

    monkeypatch.setattr("app.pipelines.evaluation.cae_metrics.compute_aupimo", lambda *_args, **_kwargs: 0.5)
    first = extract_and_save_pr_metrics(
        FakeEngine([0.1, 0.9]), object(), validation_loader, test_loader, tmp_path / "first"
    )
    second = extract_and_save_pr_metrics(
        FakeEngine([100.0, -100.0]), object(), validation_loader, test_loader, tmp_path / "second"
    )

    assert first[4] == second[4] == 0.25
    assert first[-1] == 1.0
    assert second[-1] == 0.0


def test_patchcore_seed_reproduces_random_state() -> None:
    """The PatchCore model seed resets NumPy and Torch coreset randomness."""
    _seed_patchcore_run(PATCHCORE_MODEL_SEED)
    first_numpy = np.random.rand(4)
    first_torch = torch.rand(4)

    _seed_patchcore_run(PATCHCORE_MODEL_SEED)

    assert np.array_equal(np.random.rand(4), first_numpy)
    assert torch.equal(torch.rand(4), first_torch)


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


def test_run_baseline_cached_loading(tmp_path: Path, monkeypatch: Any) -> None:
    """Verify run_baseline directly loads cached results when matching model exists.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest fixture used to supply the protocol manifest.
    """
    model_dir = tmp_path / "cached_run"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez(model_dir / "image_metrics.npz", auroc=0.99)
    np.savez(model_dir / "pixel_metrics.npz", aupimo=0.92, t_aupimo_min=0.35)
    overlays = {3: {"heatmap": np.zeros((2, 2, 3), dtype=np.uint8).tolist()}}
    _save_heatmap_overlays(overlays, model_dir / "heatmap_overlays.npz")

    manifest = pd.DataFrame(
        [
            {
                "path": str(tmp_path / "bottle" / "train" / "good" / f"{index:03}.png"),
                "product": "bottle",
                "split": "train",
                "is_anomaly": False,
            }
            for index in range(20)
        ]
        + [
            {
                "path": str(tmp_path / "bottle" / "test" / "good" / "000.png"),
                "product": "bottle",
                "split": "test",
                "is_anomaly": False,
            },
            {
                "path": str(tmp_path / "bottle" / "test" / "broken" / "001.png"),
                "product": "bottle",
                "split": "test",
                "is_anomaly": True,
            },
        ]
    )
    evidence = {
        **build_fair_evaluation_split(manifest, "bottle").evidence(),
        **fair_metric_evidence(),
        "model_seed": PATCHCORE_MODEL_SEED,
        "score_space": PATCHCORE_SCORE_SPACE,
        "image_threshold_quantile": PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
        "pixel_threshold_quantile": PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
    }
    monkeypatch.setattr("app.pipelines.modelling.baseline.build_mvtec_manifest", lambda _root: manifest)
    meta = {
        "hash": "cached_run",
        "category": "bottle",
        "backbone": "resnet18",
        "coreset_sampling_ratio": 0.1,
        "fpr_limit": 1e-4,
        "preprocessing_steps": [],
        "dataset_split": evidence,
        "protocol": "fair-eval-v1",
        "model_seed": PATCHCORE_MODEL_SEED,
        "score_space": PATCHCORE_SCORE_SPACE,
        "image_threshold_quantile": PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
        "pixel_threshold_quantile": PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
        "image_auroc": 0.99,
        "pixel_auroc": 0.91,
        "manual_image_f1": 0.95,
        "manual_pixel_f1": 0.88,
        "manual_image_prec": 1.0,
        "manual_image_rec": 1.0,
        "img_threshold": 0.42,
        "true_positives": 1,
        "false_positives": 0,
        "false_negatives": 0,
        "true_negatives": 1,
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
    assert result["dataset_split"]["train_normal"] == 17
    assert result["heatmap_overlays"] == overlays


def test_fair_evaluation_rejects_legacy_patchcore_cache(tmp_path: Path) -> None:
    """A legacy cache without protocol evidence cannot satisfy a fair-evaluation lookup."""
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

    result = find_cached_patchcore_model(
        category="bottle",
        target_hash="legacy_run",
        registry_base=tmp_path,
        expected_split_evidence={"protocol": "fair-eval-v1"},
    )

    assert result is None
