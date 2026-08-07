"""FastAPI backend server module."""

import warnings
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from app.pipelines.modelling.baseline import run_baseline
from app.pipelines.modelling.dummy_classifier import run_dummy_evaluation, run_real_data_dummy

# Suppress timm deprecation warnings emitted by downstream libraries
warnings.filterwarnings("ignore", category=FutureWarning, message=".*timm.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=".*timm.*")

app = FastAPI(title="Industrial Component Anomaly Detection API")


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


class BaselineEvaluationRequest(BaseModel):
    """Request schema for Patchcore baseline evaluation.

    Attributes:
        data_root: Path to the root directory of the MVTec AD dataset.
        category: The specific category to evaluate (e.g., 'bottle').
        fpr_limit: Maximum allowable False Positive Rate for AUPIMO threshold.
    """

    data_root: str = "data/raw/mvtec_ad"
    category: str = "bottle"
    fpr_limit: float = 1e-4
    preprocessing_steps: list[dict[str, Any]] | None = None


@app.get("/")
def read_root() -> dict[str, str]:
    """Root endpoint returning greeting message.

    Returns:
        Dictionary containing greeting message.
    """
    return {"message": "Hello from the FastAPI backend!"}


@app.get("/api/greet/{name}")
def greet_user(name: str) -> dict[str, str]:
    """Greeting endpoint taking user's name.

    Args:
        name: Name of the user.

    Returns:
        Dictionary containing greeting message.
    """
    return {"greeting": f"Hello, {name}! This was processed by FastAPI."}


@app.post("/api/pipelines/dummy")
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


@app.post("/api/pipelines/baseline")
def run_baseline_pipeline(req: BaselineEvaluationRequest) -> dict[str, Any]:
    """Run Patchcore baseline evaluation endpoint.

    Args:
        req: Request schema for Patchcore baseline evaluation.

    Returns:
        Dictionary containing evaluation metrics and summary lines.
    """
    results = run_baseline(
        data_root=req.data_root,
        category=req.category,
        fpr_limit=req.fpr_limit,
        preprocessing_steps=req.preprocessing_steps,
    )
    return {
        "status": "success",
        "category": req.category,
        "message": f"Baseline Patchcore execution finished for category '{req.category}'.",
        "results": results,
    }
