"""Gaussian blur preprocessing step."""

from typing import Any

import cv2
import numpy as np

from app.pipelines.preprocessing.base import BasePreprocessingStep


class GaussianBlurStep(BasePreprocessingStep):
    """Gaussian Blur noise reduction step.

    Attributes:
        name: Name of the preprocessing step.
    """

    name = "gaussian_blur"

    def __init__(self, kernel_size: int = 5, sigma: float = 0.0, ksize: int | None = None) -> None:
        """Initialize the Gaussian Blur step.

        Args:
            kernel_size: Size of the Gaussian kernel.
            sigma: Standard deviation of the Gaussian kernel.
            ksize: Alias for kernel_size.
        """
        chosen_ksize = ksize if ksize is not None else kernel_size
        self.kernel_size = chosen_ksize if chosen_ksize % 2 != 0 else chosen_ksize + 1
        self.sigma = sigma

    def __call__(self, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply Gaussian Blur to the image.

        Args:
            image: Input image array (Grayscale or RGB).

        Returns:
            Image with Gaussian Blur applied as a NumPy array.
        """
        return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)
