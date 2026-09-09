"""Tests for the fair frozen-DINOv3 nearest-neighbour baseline."""

import json
from pathlib import Path
from typing import Any

import pytest

from app.pipelines.modelling.dinov2_baseline import MVTEC_CATEGORIES
from app.pipelines.modelling.dinov3_baseline import (
    DINO_V3_ENCODER,
    DINO_V3_INPUT_SIZE,
    DINO_V3_PATCH_SIZE,
    _run_dinov3_category,
    run_dinov3_baseline,
)


def test_dinov3_category_uses_shared_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    """DINOv3 changes only the encoder generation and patch-aligned input."""
    captured: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"image_level": {}, "pixel_level": {}}

    monkeypatch.setattr("app.pipelines.modelling.dinov3_baseline._run_dinov2_category", fake_run)

    _run_dinov3_category(category="capsule")

    assert captured["encoder_name"] == DINO_V3_ENCODER
    assert captured["model_generation"] == "dinov3"
    assert captured["model_name"] == "DINOv3"
    assert captured["input_size"] == DINO_V3_INPUT_SIZE
    assert captured["patch_size"] == DINO_V3_PATCH_SIZE
    assert captured["masking"] == "off"


def test_dinov3_rejects_dinov2_specific_masking() -> None:
    """DINOv2's absolute PCA threshold must not create an empty DINOv3 bank."""
    with pytest.raises(ValueError, match="empty DINOv3 memory bank"):
        _run_dinov3_category(masking="published")


def test_dinov3_all_category_dispatch_reports_macro_averages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All-category execution uses the same compact macro summary as DINOv2."""
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

    monkeypatch.setattr("app.pipelines.modelling.dinov3_baseline._run_dinov3_category", fake_category)
    fake_manifest = object()
    monkeypatch.setattr("app.pipelines.modelling.dinov3_baseline.build_mvtec_manifest", lambda _root: fake_manifest)

    result = run_dinov3_baseline(category="all", registry_base=tmp_path)

    assert list(result["categories"]) == list(MVTEC_CATEGORIES)
    assert all(call["reuse_complete"] is True for call in calls)
    assert all(call["manifest"] is fake_manifest for call in calls)
    assert all(not category_result["heatmap_overlays"] for category_result in result["categories"].values())
    assert result["macro_average"]["image_f1"] == pytest.approx(8 / 15)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["macro_average"]["image_f1"] == pytest.approx(8 / 15)
    assert (tmp_path / "category_metrics.csv").is_file()


def test_dinov3_api_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """The API forwards the DINOv3 evaluation configuration."""
    from app.api.main import DINOv3EvaluationRequest, run_dinov3_pipeline

    captured: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"image_level": {"f1_score": 0.8}}

    monkeypatch.setattr("app.api.main.run_dinov3_baseline", fake_run)
    response = run_dinov3_pipeline(DINOv3EvaluationRequest(category="capsule", num_neighbors=3))

    assert response["status"] == "success"
    assert captured["encoder_name"] == DINO_V3_ENCODER
    assert captured["num_neighbors"] == 3
    assert captured["masking"] == "off"
