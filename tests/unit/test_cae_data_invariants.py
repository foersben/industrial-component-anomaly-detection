"""Tests for data partitioning, isolation invariants, and augmentation boundaries in the CAE pipeline.

This module validates that train/validation splitting occurs strictly on normal samples without
cross-split contamination, and that photometric data augmentations are exclusively applied to
the training partition.
"""

from typing import Any
from unittest.mock import patch

import sklearn.model_selection as ms

import app.pipelines.multi_stage_ae.cae_pipeline as p_mod
from app.pipelines.multi_stage_ae.cae_pipeline import run_keras_cae_pipeline


def test_partition_label_isolation(mock_large_mvtec_dataset: str) -> None:
    """Verify that dataset partitioning isolates normal train/validation samples with zero leakage.

    Args:
        mock_large_mvtec_dataset: Path to the large temporary mock MVTec dataset root.
    """
    original_split = ms.train_test_split
    split_records: list[dict[str, Any]] = []

    def mock_train_test_split(*arrays: Any, **options: Any) -> Any:
        result = original_split(*arrays, **options)
        split_records.append({"arrays": arrays, "result": result, "options": options})
        return result

    class AbortPipelineError(Exception):
        """Sentinel exception to abort pipeline immediately after data partitioning."""

    with patch("sklearn.model_selection.train_test_split", side_effect=mock_train_test_split):
        with patch("app.pipelines.multi_stage_ae.cae_pipeline.build_cae", side_effect=AbortPipelineError):
            try:
                run_keras_cae_pipeline(
                    data_root=mock_large_mvtec_dataset,
                    category="bottle",
                    img_size=32,
                    crop_size=16,
                    crop_stride=16,
                    latent_channels=8,
                    epochs=1,
                    batch_size=2,
                    force_retrain=True,
                )
            except AbortPipelineError:
                pass

    assert len(split_records) == 1
    rec = split_records[0]
    assert rec["options"].get("test_size") == 0.15
    assert len(rec["arrays"][0]) == 20
    assert len(rec["result"][0]) == 17
    assert len(rec["result"][1]) == 3

    train_paths = set(rec["result"][0]["path"].tolist())
    val_paths = set(rec["result"][1]["path"].tolist())
    assert len(train_paths.intersection(val_paths)) == 0


def test_augmentation_leakage_prevention(mock_mvtec_dataset: str) -> None:
    """Verify that data augmentations are applied strictly to training batches and never to validation/test sets.

    Args:
        mock_mvtec_dataset: Path to the temporary mock MVTec dataset root.
    """
    original_augment = p_mod.augment_batch
    augment_calls: list[int] = []

    def mock_augment_batch(batch: Any, augmenter: Any) -> Any:
        augment_calls.append(len(batch))
        return original_augment(batch, augmenter)

    class AbortPipelineError(Exception):
        """Sentinel exception to abort pipeline before neural network initialization."""

    with patch("app.pipelines.multi_stage_ae.cae_pipeline.augment_batch", side_effect=mock_augment_batch):
        with patch("app.pipelines.multi_stage_ae.cae_pipeline.build_cae", side_effect=AbortPipelineError):
            try:
                run_keras_cae_pipeline(
                    data_root=mock_mvtec_dataset,
                    category="bottle",
                    img_size=32,
                    crop_size=16,
                    crop_stride=16,
                    latent_channels=8,
                    epochs=1,
                    batch_size=2,
                    force_retrain=True,
                )
            except AbortPipelineError:
                pass

    # With 10 train normal images and test_size=0.15 (ceil/round to 2 val), 8 train images are augmented
    assert sum(augment_calls) == 8
