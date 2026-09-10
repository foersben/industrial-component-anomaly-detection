"""Type definitions for DINO baseline pipelines."""

from typing import Literal, TypedDict

from app.pipelines.modelling.patchcore import BaselineResult

MaskingMode = Literal["off", "on", "published"]
DINOVariant = Literal["baseline", "enhanced"]
DINOv2Variant = DINOVariant


class AllCategoriesResult(TypedDict):
    """Results and unweighted macro averages for all MVTec categories."""

    category: Literal["all"]
    masking_mode: MaskingMode
    categories: dict[str, BaselineResult]
    macro_average: dict[str, float]


__all__ = [
    "AllCategoriesResult",
    "BaselineResult",
    "DINOVariant",
    "DINOv2Variant",
    "MaskingMode",
]
