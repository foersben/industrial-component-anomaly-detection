"""DINO foundation vision transformer baseline pipelines for industrial anomaly detection.

Provides standard and enhanced multi-layer patch nearest-neighbor anomaly detection
using self-supervised vision transformer representations (DINOv2 and DINOv3).
"""

from app.pipelines.modelling.dino.engine import (
    release_accelerator_memory,
    resolve_masking,
    run_dino_all_categories,
    run_dino_category,
)
from app.pipelines.modelling.dino.enhanced import EnhancedAnomalyDINOModel
from app.pipelines.modelling.dino.registry import (
    delete_cached_dino_model,
    list_trashed_dino_models,
    purge_dino_trash,
    restore_cached_dino_model,
)
from app.pipelines.modelling.dino.types import (
    AllCategoriesResult,
    BaselineResult,
    DINOv2Variant,
    DINOVariant,
    MaskingMode,
)
from app.pipelines.modelling.dino.v2 import (
    DINO_V2_BATCH_SIZE,
    DINO_V2_ENCODER,
    DINO_V2_INPUT_SIZE,
    DINO_V2_PATCH_SIZE,
    ENHANCED_DINO_LAYERS,
    run_dinov2_baseline,
)
from app.pipelines.modelling.dino.v3 import (
    DINO_V3_ENCODER,
    DINO_V3_INPUT_SIZE,
    DINO_V3_PATCH_SIZE,
    run_dinov3_baseline,
)

__all__ = [
    "DINO_V2_BATCH_SIZE",
    "DINO_V2_ENCODER",
    "DINO_V2_INPUT_SIZE",
    "DINO_V2_PATCH_SIZE",
    "DINO_V3_ENCODER",
    "DINO_V3_INPUT_SIZE",
    "DINO_V3_PATCH_SIZE",
    "ENHANCED_DINO_LAYERS",
    "AllCategoriesResult",
    "BaselineResult",
    "DINOVariant",
    "DINOv2Variant",
    "EnhancedAnomalyDINOModel",
    "MaskingMode",
    "delete_cached_dino_model",
    "list_trashed_dino_models",
    "purge_dino_trash",
    "release_accelerator_memory",
    "resolve_masking",
    "restore_cached_dino_model",
    "run_dino_all_categories",
    "run_dino_category",
    "run_dinov2_baseline",
    "run_dinov3_baseline",
]
