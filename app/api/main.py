"""FastAPI application for Industrial Component Anomaly Detection."""

import warnings
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers.autoencoder import (
    AutoencoderEvaluationRequest,
    run_autoencoder_endpoint,
)
from app.api.routers.autoencoder import (
    router as autoencoder_router,
)
from app.api.routers.dino import (
    DINOv2EvaluationRequest,
    DINOv3EvaluationRequest,
)
from app.api.routers.dino import (
    router as dino_router,
)
from app.api.routers.dummy import (
    DummyEvaluationRequest,
    run_dummy_pipeline,
)
from app.api.routers.dummy import (
    router as dummy_router,
)
from app.api.routers.keras_cae import (
    KerasCAERequest,
    run_keras_cae_endpoint,
)
from app.api.routers.keras_cae import (
    router as keras_cae_router,
)
from app.api.routers.patchcore import (
    PatchcoreEvaluationRequest,
    run_patchcore_endpoint,
)
from app.api.routers.patchcore import (
    router as patchcore_router,
)
from app.pipelines.modelling.autoencoder import run_autoencoder_pipeline
from app.pipelines.modelling.dinov2_baseline import run_dinov2_baseline
from app.pipelines.modelling.dinov3_baseline import DINO_V3_ENCODER, run_dinov3_baseline
from app.pipelines.modelling.dummy_classifier import run_dummy_evaluation, run_real_data_dummy
from app.pipelines.modelling.keras_cae.cae_pipeline import run_keras_cae_pipeline
from app.pipelines.modelling.patchcore import run_patchcore_pipeline

# Suppress timm deprecation warnings emitted by downstream libraries
warnings.filterwarnings("ignore", category=FutureWarning, message=".*timm.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=".*timm.*")


def run_dinov2_pipeline(req: DINOv2EvaluationRequest) -> dict[str, Any]:
    """Compatibility wrapper dispatching to current run_dinov2_baseline symbol."""
    results = globals()["run_dinov2_baseline"](
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


def run_dinov3_pipeline(req: DINOv3EvaluationRequest) -> dict[str, Any]:
    """Compatibility wrapper dispatching to current run_dinov3_baseline symbol."""
    results = globals()["run_dinov3_baseline"](
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


def create_app() -> FastAPI:
    """Application factory for FastAPI backend."""
    app = FastAPI(
        title="Industrial Component Anomaly Detection API",
        description="REST API for benchmarking and running industrial anomaly detection pipelines.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount domain routers
    app.include_router(dummy_router, prefix="/api")
    app.include_router(patchcore_router, prefix="/api")
    app.include_router(keras_cae_router, prefix="/api")
    app.include_router(autoencoder_router, prefix="/api")
    app.include_router(dino_router, prefix="/api")

    @app.get("/")
    def read_root() -> dict[str, str]:
        """Root endpoint returning greeting message."""
        return {"message": "Hello from the FastAPI backend!"}

    @app.get("/health")
    def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

    return app


app = create_app()

__all__ = [
    "DINO_V3_ENCODER",
    "AutoencoderEvaluationRequest",
    "DINOv2EvaluationRequest",
    "DINOv3EvaluationRequest",
    "DummyEvaluationRequest",
    "KerasCAERequest",
    "PatchcoreEvaluationRequest",
    "app",
    "create_app",
    "run_autoencoder_endpoint",
    "run_autoencoder_pipeline",
    "run_dinov2_baseline",
    "run_dinov2_pipeline",
    "run_dinov3_baseline",
    "run_dinov3_pipeline",
    "run_dummy_evaluation",
    "run_dummy_pipeline",
    "run_keras_cae_endpoint",
    "run_keras_cae_pipeline",
    "run_patchcore_endpoint",
    "run_patchcore_pipeline",
    "run_real_data_dummy",
]
