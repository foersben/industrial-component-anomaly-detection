"""PatchCore pipeline API router."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.pipelines.modelling.patchcore import run_patchcore_pipeline

router = APIRouter(tags=["patchcore"])


class PatchcoreEvaluationRequest(BaseModel):
    """Request schema for PatchCore pipeline evaluation.

    Attributes:
        data_root: Path to the MVTec AD dataset root directory.
        category: Component category to evaluate on (e.g., 'bottle', 'wood').
        preprocessing_steps: Optional ordered list of preprocessing steps to apply.
        fpr_limit: Maximum allowable False Positive Rate for AUPIMO bounds.
        backbone: Feature extractor backbone for Patchcore (e.g., 'resnet18', 'wide_resnet50_2').
        coreset_sampling_ratio: Fraction of the feature pool to keep in the memory bank.
        num_neighbors: Number of nearest neighbors for scoring.
        run_heatmap: Whether to compute Anomaly Heatmap overlays for anomalous images.
        force_retrain: If True, bypasses model cache and forces refitting.
        model_hash: Optional target model hash to load from cache.
    """

    data_root: str = "data/raw/mvtec_ad"
    category: str = "bottle"
    preprocessing_steps: list[dict[str, Any]] | None = None
    fpr_limit: float = 1e-4
    backbone: str = "resnet18"
    coreset_sampling_ratio: float = 0.1
    num_neighbors: int = 9
    run_heatmap: bool = False
    force_retrain: bool = False
    model_hash: str | None = None


# Backward-compatible alias
BaselineEvaluationRequest = PatchcoreEvaluationRequest


@router.post("/patchcore")
@router.post("/pipelines/patchcore")
@router.post("/baseline")
@router.post("/pipelines/baseline")
def run_patchcore_endpoint(req: PatchcoreEvaluationRequest) -> dict[str, Any]:
    """Run PatchCore pipeline evaluation endpoint.

    Args:
        req: Request schema for PatchCore evaluation.

    Returns:
        Dictionary containing evaluation metrics and execution summary.
    """
    results = run_patchcore_pipeline(
        data_root=Path(req.data_root),
        category=req.category,
        pipeline=req.preprocessing_steps,
        fpr_limit=req.fpr_limit,
        backbone=req.backbone,
        coreset_sampling_ratio=req.coreset_sampling_ratio,
        num_neighbors=req.num_neighbors,
        run_heatmap=req.run_heatmap,
        force_retrain=req.force_retrain,
        model_hash=req.model_hash,
    )
    return {
        "status": "success",
        "category": req.category,
        "message": f"Patchcore execution finished for category '{req.category}'.",
        "results": results,
    }


run_baseline_pipeline = run_patchcore_endpoint
