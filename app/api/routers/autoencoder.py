"""Convolutional Autoencoder baseline API router."""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.pipelines.modelling.autoencoder import run_autoencoder_pipeline

router = APIRouter(tags=["autoencoder"])


class AutoencoderEvaluationRequest(BaseModel):
    """Request schema for Convolutional Autoencoder baseline evaluation.

    Attributes:
        data_root: Path to the root directory of the MVTec AD dataset.
        category: The specific category to evaluate (e.g., 'bottle').
        epochs: Number of training epochs.
        batch_size: Batch size for training and evaluation.
        latent_dim: Latent space bottleneck dimension.
        img_size: Image size for resizing.
        lr: Learning rate.
    """

    data_root: str = "data/raw/mvtec_ad"
    category: str = "bottle"
    epochs: int = 5
    batch_size: int = 16
    latent_dim: int = 64
    img_size: int = 64
    lr: float = 1e-3


@router.post("/autoencoder")
@router.post("/pipelines/autoencoder")
def run_autoencoder_endpoint(req: AutoencoderEvaluationRequest) -> dict[str, Any]:
    """Run Convolutional Autoencoder evaluation endpoint.

    Args:
        req: Request schema for Convolutional Autoencoder evaluation.

    Returns:
        Dictionary containing evaluation metrics and classification report.
    """
    results = run_autoencoder_pipeline(
        data_root=req.data_root,
        category=req.category,
        epochs=req.epochs,
        batch_size=req.batch_size,
        lr=req.lr,
        latent_dim=req.latent_dim,
        img_size=req.img_size,
    )
    return {
        "status": "success",
        "category": req.category,
        "message": f"Convolutional Autoencoder evaluation finished for category '{req.category}'.",
        "results": results,
    }
