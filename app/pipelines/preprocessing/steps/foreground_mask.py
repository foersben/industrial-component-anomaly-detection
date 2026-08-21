"""Foreground masking step using Otsu + Canny edge detection."""

from typing import Any

import numpy as np

from app.pipelines.preprocessing.base import BasePreprocessingStep


class ForegroundMaskStep(BasePreprocessingStep):
    """Applies Otsu + Canny foreground masking, replacing background with black."""

    name = "foreground_mask"

    def __init__(self, morph_kernel_size: int = 5, canny_sigma: float = 0.33) -> None:
        """Initialise the foreground masking step.

        Args:
            morph_kernel_size: Side length of morphological kernel.
            canny_sigma: Scaling factor for Canny bounds.
        """
        self.morph_kernel_size = morph_kernel_size
        self.canny_sigma = canny_sigma

        from app.pipelines.multi_stage_ae.segmentation import OtsuCannySegmentor

        self.segmentor = OtsuCannySegmentor(morph_kernel_size=self.morph_kernel_size, canny_sigma=self.canny_sigma)

    def __call__(self, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply foreground mask to a single image.

        Args:
            image: Image array, assumed to be RGB uint8.

        Returns:
            Image with background replaced by black (0, 0, 0).
        """
        masked_image, _ = self.segmentor.apply(image)
        return masked_image
