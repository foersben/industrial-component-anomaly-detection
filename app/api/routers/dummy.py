"""Dummy classifier API router."""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.pipelines.modelling.dummy_classifier import run_dummy_evaluation, run_real_data_dummy

router = APIRouter(tags=["dummy"])


class DummyEvaluationRequest(BaseModel):
    """Request schema for dummy classifier pipeline evaluation.

    Attributes:
        mode: Mode of the dummy classifier (theoretical or real).
        pixels: Number of pixels to use for theoretical evaluation.
        anomaly_ratio: Ratio of anomalies to use for theoretical evaluation.
        data_root: Path to the root directory of the MVTec AD dataset.
        category: The specific category to evaluate (e.g., 'bottle').
    """

    mode: str = "theoretical"
    pixels: int = 1000000
    anomaly_ratio: float = 0.015
    data_root: str = "data/raw/mvtec_ad"
    category: str = "bottle"


@router.post("/dummy")
@router.post("/pipelines/dummy")
def run_dummy_pipeline(req: DummyEvaluationRequest) -> dict[str, Any]:
    """Run dummy classifier evaluation endpoint.

    Args:
        req: Request schema for dummy classifier pipeline evaluation.

    Returns:
        Dictionary containing evaluation metrics and summary lines.
    """
    if req.mode == "theoretical":
        acc = run_dummy_evaluation(total_pixels=req.pixels, anomaly_ratio=req.anomaly_ratio)
        return {
            "mode": "theoretical",
            "accuracy": acc,
            "message": f"Theoretical dummy evaluation completed with accuracy {acc * 100:.2f}%",
        }
    else:
        results = run_real_data_dummy(data_root=req.data_root, category=req.category)
        return {
            "mode": "real",
            "category": req.category,
            "results": results,
            "message": f"Real dataset dummy evaluation executed for category '{req.category}'.",
        }
