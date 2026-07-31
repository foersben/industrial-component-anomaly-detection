"""Image preprocessing module containing CLAHE transformation."""

from typing import Any

import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray[Any, Any],
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray[Any, Any]:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    Args:
        image: Input image array (Grayscale or RGB).
        clip_limit: Threshold for contrast clipping.
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        Enhanced image as a NumPy array.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if image.ndim == 2:  # Grayscale
        return clahe.apply(image)

    # Convert RGB to LAB, apply CLAHE to L-channel, and convert back
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))

    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
