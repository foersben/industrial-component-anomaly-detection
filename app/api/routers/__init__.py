"""API routers package."""

from app.api.routers.autoencoder import router as autoencoder_router
from app.api.routers.dino import router as dino_router
from app.api.routers.dummy import router as dummy_router
from app.api.routers.keras_cae import router as keras_cae_router
from app.api.routers.patchcore import router as patchcore_router

__all__ = [
    "autoencoder_router",
    "dino_router",
    "dummy_router",
    "keras_cae_router",
    "patchcore_router",
]
