"""FastAPI backend server module."""

import warnings
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from app.pipelines.modelling.autoencoder import run_autoencoder_pipeline
from app.pipelines.modelling.baseline import run_baseline
from app.pipelines.modelling.dummy_classifier import run_dummy_evaluation, run_real_data_dummy
from app.pipelines.modelling.keras_cae.cae_pipeline import run_keras_cae_pipeline

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
    """Request schema for the Patchcore baseline pipeline.

    Attributes:
        data_root: Path to the MVTec AD dataset root directory.
        category: Component category to evaluate on (e.g., 'bottle', 'wood').
        preprocessing_steps: Optional ordered list of preprocessing steps to apply.
        fpr_limit: Maximum allowable False Positive Rate for AUPIMO bounds.
        backbone: Feature extractor backbone for Patchcore (e.g., 'resnet18', 'wide_resnet50_2').
        coreset_sampling_ratio: Fraction of the feature pool to keep in the memory bank.
        run_heatmap: Whether to compute Anomaly Heatmap overlays for anomalous images.
    """

    data_root: str = "data/raw/mvtec_ad"
    category: str = "bottle"
    preprocessing_steps: list[dict[str, Any]] | None = None
    fpr_limit: float = 1e-4
    backbone: str = "resnet18"
    coreset_sampling_ratio: float = 0.1
    run_heatmap: bool = False


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
        data_root=Path(req.data_root),
        category=req.category,
        pipeline=req.preprocessing_steps,
        fpr_limit=req.fpr_limit,
        backbone=req.backbone,
        coreset_sampling_ratio=req.coreset_sampling_ratio,
        run_heatmap=req.run_heatmap,
    )
    return {
        "status": "success",
        "category": req.category,
        "message": f"Baseline Patchcore execution finished for category '{req.category}'.",
        "results": results,
    }


@app.post("/api/pipelines/autoencoder")
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
        use_segmentation: Whether to apply Otsu+Canny foreground extraction.
        run_heatmap: Whether to compute Reconstruction Error heatmap overlays for anomalous images.
        force_retrain: If True, bypasses model cache and forces training of a new model.
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


@app.post("/api/pipelines/keras_cae")
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
