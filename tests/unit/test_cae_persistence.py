"""Unit tests verifying Keras CAE model persistence, caching, and inference consistency."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from app.pipelines.multi_stage_ae.cae_keras import build_cae
from app.pipelines.multi_stage_ae.cae_pipeline import find_cached_model, run_keras_cae_pipeline


def test_keras_cae_save_and_load_numerical_consistency(tmp_path: Path) -> None:
    """Ensure that saving a CAE and loading via tf.keras reproduces identical predictions.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    model = build_cae(crop_size=32, latent_channels=16)
    dummy_input = np.random.RandomState(42).rand(4, 32, 32, 3).astype(np.float32)

    pred_before = model.predict(dummy_input, verbose=0)

    model_path = tmp_path / "model.keras"
    model.save(model_path)

    loaded_model = tf.keras.models.load_model(model_path, compile=False)
    pred_after = loaded_model.predict(dummy_input, verbose=0)

    assert np.allclose(pred_before, pred_after, atol=1e-6)


def test_find_cached_model_matching_preprocessing(tmp_path: Path) -> None:
    """Verify that find_cached_model strictly matches preprocessing_steps.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    reg_dir = tmp_path / "model_abc"
    reg_dir.mkdir(parents=True, exist_ok=True)
    (reg_dir / "model.keras").touch()

    meta = {
        "hash": "model_abc",
        "category": "bottle",
        "img_size": 64,
        "crop_size": 32,
        "crop_stride": 16,
        "latent_channels": 16,
        "epochs": 1,
        "batch_size": 4,
        "mask_ratio": 0.25,
        "mask_patch_size": 8,
        "preprocessing_steps": [{"name": "foreground_mask", "params": {}}],
        "timestamp": "2026-08-22T00:00:00+00:00",
    }
    with open(reg_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # 1. Search without matching preprocessing -> should return None
    miss = find_cached_model(
        category="bottle",
        img_size=64,
        crop_size=32,
        crop_stride=16,
        latent_channels=16,
        epochs=1,
        batch_size=4,
        mask_ratio=0.25,
        mask_patch_size=8,
        preprocessing_steps=[],
        registry_base=tmp_path,
    )
    assert miss is None

    # 2. Search with matching preprocessing -> should return the model
    hit = find_cached_model(
        category="bottle",
        img_size=64,
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
