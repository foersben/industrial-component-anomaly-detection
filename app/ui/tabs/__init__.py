"""UI tabs package."""

from app.ui.tabs.autoencoder_tab import render_autoencoder_tab
from app.ui.tabs.dino_tab import render_dino_tab
from app.ui.tabs.dummy_tab import render_dummy_evaluation_tab
from app.ui.tabs.guide_tab import render_evaluation_guide_tab
from app.ui.tabs.keras_cae_tab import render_keras_cae_tab
from app.ui.tabs.patchcore_tab import render_baseline_patchcore_tab

__all__ = [
    "render_autoencoder_tab",
    "render_baseline_patchcore_tab",
    "render_dino_tab",
    "render_dummy_evaluation_tab",
    "render_evaluation_guide_tab",
    "render_keras_cae_tab",
]
