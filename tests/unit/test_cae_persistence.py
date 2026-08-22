"""Unit tests verifying Keras CAE model persistence, caching, and inference consistency."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from app.pipelines.modelling.keras_cae.cae_keras import build_cae
from app.pipelines.modelling.keras_cae.cae_pipeline import (
    delete_cached_model,
    find_cached_model,
    list_trashed_models,
    purge_trash,
    restore_cached_model,
    run_keras_cae_pipeline,
)


def test_keras_cae_save_and_load_numerical_consistency(tmp_path: Path) -> None:
    """Ensure that saving a CAE and loading via tf.keras reproduces identical predictions.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    model = build_cae(crop_size=32, latent_channels=16)
    dummy_input = np.random.RandomState(42).rand(4, 32, 32, 3).astype(np.float32)
    pred_orig = model.predict(dummy_input, verbose=0)

    save_path = tmp_path / "model.keras"
    model.save(save_path)

    loaded_model = tf.keras.models.load_model(save_path, compile=False)
    pred_loaded = loaded_model.predict(dummy_input, verbose=0)

    np.testing.assert_allclose(
        pred_orig,
        pred_loaded,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Loaded model predictions deviate from in-memory model!",
    )


def test_find_cached_model_resolution(tmp_path: Path) -> None:
    """Verify that find_cached_model accurately identifies matching models in registry.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    # Create fake cached model metadata
    reg_dir = tmp_path / "model_abc"
    reg_dir.mkdir(parents=True, exist_ok=True)
    (reg_dir / "model.keras").touch()
    meta = {
        "hash": "model_abc",
        "category": "bottle",
        "img_size": 128,
        "crop_size": 32,
        "crop_stride": 16,
        "latent_channels": 16,
        "epochs": 1,
        "batch_size": 4,
        "mask_ratio": 0.25,
        "mask_patch_size": 8,
        "preprocessing_steps": [{"name": "foreground_mask", "params": {}}],
        "timestamp": "2026-08-21T10:00:00",
    }
    with open(reg_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # Search with matching parameters
    hit = find_cached_model(
        category="bottle",
        img_size=128,
        crop_size=32,
        crop_stride=16,
        latent_channels=16,
        epochs=1,
        batch_size=4,
        mask_ratio=0.25,
        mask_patch_size=8,
        preprocessing_steps=[{"name": "foreground_mask", "params": {}}],
        registry_base=tmp_path,
    )
    assert hit is not None
    found_dir, found_meta = hit
    assert found_dir == reg_dir
    assert found_meta["hash"] == "model_abc"


def test_keras_cae_pipeline_cached_evaluation(mock_mvtec_dataset: str) -> None:
    """Run pipeline to train and save a model on mock data, then reload and verify exact match.

    Args:
        mock_mvtec_dataset: Path to the temporary mock MVTec dataset root.
    """
    # 1. Fresh training
    res1 = run_keras_cae_pipeline(
        data_root=mock_mvtec_dataset,
        category="bottle",
        img_size=32,
        crop_size=16,
        crop_stride=8,
        latent_channels=8,
        epochs=1,
        batch_size=2,
        mask_ratio=0.0,
        preprocessing_steps=[],
        run_heatmap=False,
        force_retrain=True,
    )

    model_hash = res1["model_hash"]

    # 2. Reload via cache
    res2 = run_keras_cae_pipeline(
        data_root=mock_mvtec_dataset,
        category="bottle",
        img_size=32,
        crop_size=16,
        crop_stride=8,
        latent_channels=8,
        epochs=1,
        batch_size=2,
        mask_ratio=0.0,
        preprocessing_steps=[],
        run_heatmap=False,
        force_retrain=False,
        model_hash=model_hash,
    )

    # 3. Assert predictions and metrics are identical
    assert np.isclose(res1["threshold"], res2["threshold"], atol=1e-5)
    assert np.isclose(res1["image_level"]["auroc"], res2["image_level"]["auroc"], atol=1e-5)
    assert np.allclose(res1["scores"], res2["scores"], atol=1e-5)


def test_delete_and_restore_cached_model(tmp_path: Path) -> None:
    """Verify soft-deletion moves model to .trash and restore restores it back.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    model_hash = "model_to_test"
    model_dir = tmp_path / model_hash
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.keras").touch()
    (model_dir / "metadata.json").write_text('{"hash": "model_to_test", "category": "bottle"}', encoding="utf-8")

    assert model_dir.exists()

    # 1. Soft Delete
    deleted = delete_cached_model(model_hash, registry_base=tmp_path, soft_delete=True)
    assert deleted is True
    assert not model_dir.exists()
    assert (tmp_path / ".trash" / model_hash).exists()

    # 2. List Trashed Models
    trashed = list_trashed_models(registry_base=tmp_path)
    assert len(trashed) == 1
    assert trashed[0]["hash"] == model_hash

    # 3. Restore Model
    restored = restore_cached_model(model_hash, registry_base=tmp_path)
    assert restored is True
    assert model_dir.exists()
    assert not (tmp_path / ".trash" / model_hash).exists()


def test_delete_cached_model_hard(tmp_path: Path) -> None:
    """Verify hard deletion permanently removes the directory.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    model_hash = "model_to_hard_delete"
    model_dir = tmp_path / model_hash
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.keras").touch()
    (model_dir / "metadata.json").write_text("{}", encoding="utf-8")

    deleted = delete_cached_model(model_hash, registry_base=tmp_path, soft_delete=False)
    assert deleted is True
    assert not model_dir.exists()
    assert not (tmp_path / ".trash" / model_hash).exists()


def test_purge_trash(tmp_path: Path) -> None:
    """Verify purge_trash removes specific or all models from .trash directory.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    for h in ["model_1", "model_2"]:
        m_dir = tmp_path / h
        m_dir.mkdir(parents=True, exist_ok=True)
        (m_dir / "model.keras").touch()
        (m_dir / "metadata.json").write_text("{}", encoding="utf-8")
        delete_cached_model(h, registry_base=tmp_path, soft_delete=True)

    assert len(list_trashed_models(registry_base=tmp_path)) == 2

    # Purge single model
    purged_1 = purge_trash(registry_base=tmp_path, model_hash="model_1")
    assert purged_1 == 1
    assert len(list_trashed_models(registry_base=tmp_path)) == 1

    # Purge all remaining
    purged_all = purge_trash(registry_base=tmp_path)
    assert purged_all == 1
    assert len(list_trashed_models(registry_base=tmp_path)) == 0


def test_delete_cached_model_safety_guards(tmp_path: Path) -> None:
    """Verify safety checks prevent path traversal or invalid hash deletion.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    assert delete_cached_model("", registry_base=tmp_path) is False
    assert delete_cached_model("ab", registry_base=tmp_path) is False
    assert delete_cached_model("..", registry_base=tmp_path) is False
    assert delete_cached_model("../..", registry_base=tmp_path) is False
    assert delete_cached_model(".trash", registry_base=tmp_path) is False
    assert delete_cached_model("model_123", registry_base=tmp_path / "non_existent") is False
    assert restore_cached_model("non_existent", registry_base=tmp_path) is False
