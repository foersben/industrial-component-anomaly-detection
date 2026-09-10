"""DINO vision foundation model API router."""

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app.pipelines.modelling.dinov2_baseline import run_dinov2_baseline
from app.pipelines.modelling.dinov3_baseline import DINO_V3_ENCODER, run_dinov3_baseline

router = APIRouter(tags=["dino"])


class DINOv2EvaluationRequest(BaseModel):
    """Request schema for the frozen DINOv2 nearest-neighbour baseline."""

    data_root: str = "data/raw/mvtec_ad"
    category: str = "bottle"
    preprocessing_steps: list[dict[str, Any]] | None = None
    fpr_limit: float = 1e-4
    encoder_name: str = "vit_small_patch14_dinov2"
    num_neighbors: int = 1
    masking: Literal["off", "on", "published"] = "published"
    run_heatmap: bool = False
    variant: Literal["baseline", "enhanced"] = "baseline"


class DINOv3EvaluationRequest(BaseModel):
    """Request schema for the frozen DINOv3 nearest-neighbour baseline."""

    data_root: str = "data/raw/mvtec_ad"
    category: str = "bottle"
    preprocessing_steps: list[dict[str, Any]] | None = None
    fpr_limit: float = 1e-4
    encoder_name: str = DINO_V3_ENCODER
    num_neighbors: int = 1
    masking: Literal["off"] = "off"
    run_heatmap: bool = False


@router.post("/dinov2")
@router.post("/pipelines/dinov2")
def run_dinov2_pipeline(req: DINOv2EvaluationRequest) -> dict[str, Any]:
    """Run the frozen DINOv2 patch nearest-neighbour baseline."""
    results = run_dinov2_baseline(
        data_root=Path(req.data_root),
        category=req.category,
        pipeline=req.preprocessing_steps,
        fpr_limit=req.fpr_limit,
        encoder_name=req.encoder_name,
        num_neighbors=req.num_neighbors,
        masking=req.masking,
        run_heatmap=req.run_heatmap,
        variant=req.variant,
    )
    return {
        "status": "success",
        "category": req.category,
        "message": f"DINOv2 baseline execution finished for category '{req.category}'.",
        "results": results,
    }


@router.post("/dinov3")
@router.post("/pipelines/dinov3")
def run_dinov3_pipeline(req: DINOv3EvaluationRequest) -> dict[str, Any]:
    """Run the frozen DINOv3 patch nearest-neighbour baseline."""
    results = run_dinov3_baseline(
        data_root=Path(req.data_root),
        category=req.category,
        pipeline=req.preprocessing_steps,
        fpr_limit=req.fpr_limit,
        encoder_name=req.encoder_name,
        num_neighbors=req.num_neighbors,
        masking=req.masking,
        run_heatmap=req.run_heatmap,
    )
    return {
        "status": "success",
        "category": req.category,
        "message": f"DINOv3 baseline execution finished for category '{req.category}'.",
        "results": results,
    }
