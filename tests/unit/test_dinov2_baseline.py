"""Tests for the fair frozen-DINOv2 nearest-neighbour baseline."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from app.domain.categories import ANOMALY_DINO_MASKED_CATEGORIES, MVTEC_CATEGORIES
from app.pipelines.modelling.dino import (
    EnhancedAnomalyDINOModel,
    resolve_masking,
    run_dinov2_baseline,
)


def test_enhanced_knn_restricts_matches_to_patch_position() -> None:
    """A visually identical token at the wrong position is not a valid exact match."""
    model = EnhancedAnomalyDINOModel.__new__(EnhancedAnomalyDINOModel)
    torch.nn.Module.__init__(model)
    model.num_neighbours = 1
    model.position_radius = 0
    model.spatial_weight = 0.05
    model.memory_bank = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    model.memory_valid = torch.ones((1, 2), dtype=torch.bool)
    model.memory_density = torch.ones((1, 2))
    model.density_reference = torch.tensor(1.0)
    queries = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])

    patch_scores, _ = model._score_features(queries, grid_size=(1, 2))

    assert patch_scores[0, 0] == pytest.approx(1.0)
    assert patch_scores[0, 1] == pytest.approx(0.0)


def test_enhanced_image_score_reweights_the_worst_patch() -> None:
    """Image aggregation applies neighborhood confidence to the worst patch."""
    neighbours = torch.tensor([[[0.1, 0.2, 0.3], [1.0, 1.2, 1.4]]])

    score = EnhancedAnomalyDINOModel.patchcore_image_score(neighbours)

    assert score.shape == (1, 1)
    assert 0 < score.item() < neighbours[0, 1].mean().item()


def test_enhanced_knn_falls_back_when_masking_empties_local_window() -> None:
    """Masked local windows use spatially penalized global normal candidates."""
    model = EnhancedAnomalyDINOModel.__new__(EnhancedAnomalyDINOModel)
    torch.nn.Module.__init__(model)
    model.num_neighbours = 1
    model.position_radius = 0
    model.spatial_weight = 0.05
    model.memory_bank = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    model.memory_valid = torch.tensor([[False, True]])
    model.memory_density = torch.ones((1, 2))
    model.density_reference = torch.tensor(1.0)
    queries = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])

    patch_scores, _ = model._score_features(queries, grid_size=(1, 2))

    assert patch_scores[0, 0] == pytest.approx(0.05)
    assert patch_scores[0, 1] == pytest.approx(0.0)


def test_published_masking_policy_matches_full_shot_categories() -> None:
    """The published policy is fixed before test evaluation, never inferred from scores."""
    for category in ANOMALY_DINO_MASKED_CATEGORIES:
        assert resolve_masking("published", category)
    assert not resolve_masking("published", "bottle")
    assert resolve_masking("on", "bottle")
    assert not resolve_masking("off", "capsule")


def test_invalid_masking_policy_is_rejected() -> None:
    """Unknown experiment policies fail rather than silently changing the model."""
    with pytest.raises(ValueError, match="masking must be one of"):
        resolve_masking("automatic", "bottle")  # type: ignore[arg-type]


def test_all_category_dispatch_reports_macro_averages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The all dispatcher runs the fixed category list without changing configuration."""
    calls: list[dict[str, Any]] = []

    def fake_category(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        value = float(MVTEC_CATEGORIES.index(kwargs["category"]) + 1) / len(MVTEC_CATEGORIES)
        return {
            "image_level": {
                "f1_score": value,
                "recall": value,
                "precision": value,
                "auroc": value,
                "average_precision": value,
            },
            "pixel_level": {"f1_score": value, "auroc": value, "aupimo": value},
            "heatmap_overlays": {0: {"heatmap": [[[1]]]}},
        }

    monkeypatch.setattr("app.pipelines.modelling.dino.engine.run_dino_category", fake_category)
    fake_manifest = object()
    monkeypatch.setattr("app.pipelines.modelling.dino.engine.build_mvtec_manifest", lambda _root: fake_manifest)

    result = run_dinov2_baseline(category="all", masking="published", registry_base=tmp_path)

    assert result["category"] == "all"
    assert list(result["categories"]) == list(MVTEC_CATEGORIES)
    assert [call["category"] for call in calls] == list(MVTEC_CATEGORIES)
    assert all(call["masking"] == "published" for call in calls)
    assert all(call["reuse_complete"] is True for call in calls)
    assert all(call["manifest"] is fake_manifest for call in calls)
    assert all(not category_result["heatmap_overlays"] for category_result in result["categories"].values())
    assert result["macro_average"]["image_f1"] == pytest.approx(8 / 15)


def test_dinov2_runner_uses_shared_partitions_and_raw_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training, calibration, and test loaders remain distinct throughout the run."""
    fitting_loader = object()
    validation_loader = object()
    test_loader = object()
    fake_split = SimpleNamespace(
        evidence=lambda: {
            "protocol": "fair-eval-v1",
            "split_seed": 42,
            "validation_fraction": 0.15,
            "train_normal": 17,
            "val_normal": 3,
            "test_total": 2,
            "fitting_path_digest": "fit",
            "validation_path_digest": "val",
            "test_path_digest": "test",
        },
        test=pd.DataFrame({"is_anomaly": [False, True]}),
    )

    class FakeDataModule:
        def train_dataloader(self) -> object:
            return fitting_loader

        def val_dataloader(self) -> object:
            return validation_loader

        def test_dataloader(self) -> object:
            return test_loader

    captured: dict[str, Any] = {}

    class FakeFeatureEncoder:
        def requires_grad_(self, value: bool) -> None:
            captured["encoder_requires_grad"] = value

    class FakeEngine:
        def __init__(self, **kwargs: Any) -> None:
            captured["engine"] = kwargs

        def fit(self, model: object, train_dataloaders: object) -> None:
            captured["fit_model"] = model
            captured["fit_loader"] = train_dataloaders

    def fake_model(**kwargs: Any) -> SimpleNamespace:
        captured["model_kwargs"] = kwargs
        return SimpleNamespace(
            visualizer=kwargs["visualizer"],
            model=SimpleNamespace(feature_encoder=FakeFeatureEncoder()),
        )

    def fake_evaluate(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        captured["evaluation_args"] = args
        captured["evaluation_kwargs"] = kwargs
        base_dir = Path(args[4])
        np.savez(
            base_dir / "image_metrics.npz",
            precision=np.array([0.5, 1.0, 1.0]),
            recall=np.array([1.0, 0.5, 0.0]),
            thresholds=np.array([0.2, 0.8]),
        )
        np.savez(base_dir / "pixel_metrics.npz", aupimo=0.4)
        return (
            0.7,
            0.3,
            0.8,
            0.625,
            0.25,
            0.9,
            0.4,
            0.0,
            1.0,
            1.0,
            {},
            [1],
            1,
            0,
            1,
            1,
            0.5,
            0.85,
        )

    monkeypatch.setattr("app.pipelines.modelling.dino.engine.build_mvtec_manifest", lambda _root: object())
    monkeypatch.setattr(
        "app.pipelines.modelling.dino.engine.build_fair_evaluation_split",
        lambda _manifest, _category: fake_split,
    )
    monkeypatch.setattr("app.pipelines.modelling.dino.engine.build_pipeline_from_configs", lambda _steps: [])
    monkeypatch.setattr("app.pipelines.modelling.dino.engine.MVTecAD", lambda **_kwargs: FakeDataModule())
    monkeypatch.setattr("app.pipelines.modelling.dino.engine.configure_anomalib_partitions", lambda *_args: None)
    monkeypatch.setattr("app.pipelines.modelling.dino.engine.AnomalyDINO", fake_model)
    monkeypatch.setattr("app.pipelines.modelling.dino.engine.Engine", FakeEngine)
    monkeypatch.setattr("app.pipelines.modelling.dino.engine.extract_and_save_pr_metrics", fake_evaluate)

    result = run_dinov2_baseline(
        data_root=tmp_path,
        category="capsule",
        masking="published",
        registry_base=tmp_path / "models",
    )

    assert captured["fit_loader"] is fitting_loader
    assert captured["evaluation_args"][2] is validation_loader
    assert captured["evaluation_args"][3] is test_loader
    assert captured["evaluation_kwargs"] == {"model_name": "DINOv2"}
    assert captured["model_kwargs"]["masking"] is True
    assert captured["model_kwargs"]["coreset_subsampling"] is False
    assert captured["model_kwargs"]["post_processor"] is False
    assert captured["model_kwargs"]["evaluator"] is False
    assert captured["encoder_requires_grad"] is False
    assert result["image_level"]["average_precision"] == pytest.approx(0.75)
    assert result["dataset_split"]["threshold_source"] == "normal_validation"
    assert result["metadata"]["masking_mode"] == "published"
