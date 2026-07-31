"""CLAHE preprocessing step."""

from typing import Any

import cv2
import numpy as np

from app.pipelines.preprocessing.base import BasePreprocessingStep


class CLAHEStep(BasePreprocessingStep):
    """Contrast Limited Adaptive Histogram Equalization step.

    Attributes:
        name: Name of the preprocessing step.
    """

    name = "clahe"

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> None:
        """Initialize the CLAHE step.

        Args:
            clip_limit: Threshold for contrast enhancement.
            tile_grid_size: Size of the tile grid for histogram equalization.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply CLAHE to the image.

        Args:
            image: Input image array (Grayscale or RGB).

        Returns:
            Image with CLAHE applied as a NumPy array.
        """
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

        if image.ndim == 2:  # Grayscale
            return clahe.apply(image)

        # RGB Image: Apply to L-channel in LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_enhanced = clahe.apply(l_chan)
        lab_enhanced = cv2.merge((l_enhanced, a_chan, b_chan))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
