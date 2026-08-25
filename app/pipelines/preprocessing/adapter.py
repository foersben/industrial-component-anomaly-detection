"""Adapter to bridge custom PreprocessingPipeline with PyTorch/Anomalib transforms."""

from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from app.pipelines.preprocessing.base import PreprocessingPipeline


class PreprocessingTransformAdapter:
    """Adapts a PreprocessingPipeline callable for PyTorch/Anomalib image loaders."""

    def __init__(self, pipeline: PreprocessingPipeline) -> None:
        """Initialize the adapter.

        Args:
            pipeline: The preprocessing pipeline to adapt.
        """
        self.pipeline = pipeline

    def __call__(
        self,
        image: Image.Image | np.ndarray[Any, Any] | torch.Tensor,
    ) -> Image.Image | np.ndarray[Any, Any] | torch.Tensor:
        """Applies the pipeline to PIL Images, NumPy arrays, or PyTorch Tensors.

        Args:
            image: Image in any supported format (PIL, NumPy, or PyTorch Tensor).

        Returns:
            Processed image in the same format as input.
        """
        if len(self.pipeline) == 0:
            return image

        if isinstance(image, Image.Image):
            img_np = np.array(image)
            processed_np = self.pipeline(img_np)
            return Image.fromarray(processed_np)

        if isinstance(image, np.ndarray):
            return self.pipeline(image)

        if isinstance(image, torch.Tensor):
            # Convert C x H x W Tensor -> H x W x C NumPy array
            img_np = image.detach().cpu().numpy().transpose(1, 2, 0)
            
            # Safely handle float32 in [0, 1]
            is_float = img_np.dtype in (np.float32, np.float64)
            if is_float:
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255.0).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)

            processed_np = self.pipeline(img_np)
            
            # Restore to float if original was float
            if is_float:
                processed_np = processed_np.astype(np.float32) / 255.0
                
            # Convert back to C x H x W Tensor
            return torch.from_numpy(processed_np).permute(2, 0, 1).to(image.dtype)

        return image


class PreprocessedAnomalibDataset(Dataset[Any]):
    """PyTorch Dataset wrapper that applies custom preprocessing to Anomalib samples.

    Wraps an existing AnomalibDataset to apply PreprocessingTransformAdapter
    directly during dataset indexing (__getitem__), ensuring 100% compatibility
    across Anomalib versions.
    """

    def __init__(self, dataset: Any, transform_adapter: PreprocessingTransformAdapter) -> None:
        """Initialize the adapter.

        Args:
            dataset: The dataset to wrap.
            transform_adapter: The transformation adapter to apply to the dataset.
        """
        self.dataset = dataset
        self.transform_adapter = transform_adapter

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            The number of samples in the dataset.
        """
        return len(self.dataset)

    def __getattr__(self, name: str) -> Any:
        """Proxy any unknown attributes to the underlying dataset."""
        return getattr(self.dataset, name)

    def __getitem__(self, idx: int) -> Any:
        """Get a sample from the dataset and apply the transformation.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            The sample from the dataset with the transformation applied.
        """
        item = self.dataset[idx]

        # Anomalib items can be dicts or objects with an 'image' attribute
        if isinstance(item, dict) and "image" in item:
            item["image"] = self.transform_adapter(item["image"])
        elif hasattr(item, "image"):
            item.image = self.transform_adapter(item.image)

        return item
