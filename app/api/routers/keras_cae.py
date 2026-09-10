"""Keras Convolutional Autoencoder (CAE) API router."""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.pipelines.modelling.keras_cae.cae_pipeline import run_keras_cae_pipeline

router = APIRouter(tags=["keras_cae"])


class KerasCAERequest(BaseModel):
    """Request schema for the state-of-the-art Keras Convolutional Autoencoder pipeline.

    Attributes:
        data_root: Path to the MVTec AD dataset root directory.
        category: Component category to train/evaluate on (e.g., 'bottle', 'wood').
        img_size: Spatial size (H=W) for image resizing.
        crop_size: Size of sliding window crops extracted from the base image.
        crop_stride: Stride of the sliding window.
        latent_channels: Number of channels in the convolutional bottleneck.
        epochs: Number of training epochs.
        batch_size: Number of images per gradient step.
        mask_ratio: Fraction of image patches to mask for Masked Image Modeling.
        threshold_method: Adaptive threshold method: 'quantile' or 'mahalanobis'.
        k_fraction: Top-K fraction for image-level anomaly scoring.
        preprocessing_steps: Optional preprocessing step configurations.
        run_heatmap: Whether to compute Reconstruction Error heatmap overlays for anomalous images.
        force_retrain: If True, bypasses model cache and forces training of a new model.
        model_hash: Target model hash to load directly from registry.
    """

    data_root: str = "data/raw/mvtec_ad"
    category: str = "bottle"
    img_size: int = 256
    crop_size: int = 64
    crop_stride: int = 32
    latent_channels: int = 32
    epochs: int = 20
    batch_size: int = 16
    mask_ratio: float = 0.25
    threshold_method: str = "quantile"
    k_fraction: float = 0.002
    preprocessing_steps: list[dict[str, Any]] | None = None
    run_heatmap: bool = False
    force_retrain: bool = False
    model_hash: str | None = None


@router.post("/keras_cae")
@router.post("/pipelines/keras_cae")
@router.post("/cae")
@router.post("/pipelines/cae")
def run_keras_cae_endpoint(req: KerasCAERequest) -> dict[str, Any]:
    """Run the state-of-the-art Keras CAE anomaly detection pipeline.

    This pipeline incorporates: ELU activations, Masked Image Modeling (MIM),
    combined SSIM+MSE loss, AdamW optimiser, Top-K pooling, adaptive thresholds,
    AUPIMO pixel-level evaluation, and optional Reconstruction Error explainability.

    Args:
        req: Request schema with all pipeline hyperparameters.

    Returns:
        Dictionary with AUROC, AUPIMO, accuracy, precision, recall, and optional Heatmap overlays.
    """
    results = run_keras_cae_pipeline(
        data_root=req.data_root,
        category=req.category,
        img_size=req.img_size,
        crop_size=req.crop_size,
        crop_stride=req.crop_stride,
        latent_channels=req.latent_channels,
        epochs=req.epochs,
        batch_size=req.batch_size,
        mask_ratio=req.mask_ratio,
        threshold_method=req.threshold_method,
        k_fraction=req.k_fraction,
        preprocessing_steps=req.preprocessing_steps,
        run_heatmap=req.run_heatmap,
        force_retrain=req.force_retrain,
        model_hash=req.model_hash,
    )
    return {
        "status": "success",
        "category": req.category,
        "results": results,
    }
