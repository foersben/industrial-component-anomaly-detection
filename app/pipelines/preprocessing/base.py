"""Base module for image preprocessing pipelines."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BasePreprocessingStep(ABC):
    """Abstract Strategy for a single image preprocessing step."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier for the step."""
        pass

    @abstractmethod
    def __call__(self, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply the transformation to a NumPy image array (RGB or Grayscale)."""
        pass


class PreprocessingPipeline:
    """Composite pattern to chain and execute multiple preprocessing steps in order.

    Attributes:
        steps: List of preprocessing steps to apply in order.
    """

    def __init__(self, steps: list[BasePreprocessingStep] | None = None) -> None:
        """Initialize the preprocessing pipeline.

        Args:
            steps: List of preprocessing steps to apply in order.
        """
        self.steps: list[BasePreprocessingStep] = steps or []

    def add_step(self, step: BasePreprocessingStep) -> "PreprocessingPipeline":
        """Chain a new step onto the pipeline.

        Args:
            step: Preprocessing step to add.

        Returns:
            Updated preprocessing pipeline.
        """
        self.steps.append(step)
        return self

    def __call__(self, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Sequential execution of all configured preprocessing steps.

        Args:
            image: Input image array (Grayscale or RGB).

        Returns:
            Preprocessed image as a NumPy array.
        """
        for step in self.steps:
            image = step(image)
        return image

    def __len__(self) -> int:
        """Get the number of preprocessing steps in the pipeline.

        Returns:
            Number of preprocessing steps.
        """
        return len(self.steps)
