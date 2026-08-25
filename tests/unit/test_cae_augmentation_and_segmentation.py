"""Unit tests for category-specific data augmentations and foreground segmentation.

This module validates that dataset categories route to the appropriate domain augmenters
(e.g., TextureAugmenter for textures vs ObjectAugmenter for structured objects), that batch
augmentation preserves tensor datatypes and bounds, and that Otsu+Canny morphology correctly
isolates prominent foreground objects.
"""

import numpy as np

from app.pipelines.preprocessing.augmentation import ObjectAugmenter, TextureAugmenter, augment_batch, get_augmenter
from app.pipelines.preprocessing.segmentation import OtsuCannySegmentor, extract_largest_component


def test_get_augmenter_routing() -> None:
    """Verify that get_augmenter correctly routes categories to TextureAugmenter or ObjectAugmenter."""
    # Texture categories should return TextureAugmenter
    assert isinstance(get_augmenter("carpet"), TextureAugmenter)
    assert isinstance(get_augmenter("grid"), TextureAugmenter)

    # Object categories should return ObjectAugmenter
    assert isinstance(get_augmenter("bottle"), ObjectAugmenter)
    assert isinstance(get_augmenter("cable"), ObjectAugmenter)

    # Unknown category should default to ObjectAugmenter
    assert isinstance(get_augmenter("unknown"), ObjectAugmenter)


def test_augment_batch() -> None:
    """Verify that batch augmentation preserves shapes, uint8 datatypes, and valid [0, 255] ranges."""
    n_val, h, w, c_val = 2, 32, 32, 3
    batch = np.random.randint(0, 256, size=(n_val, h, w, c_val), dtype=np.uint8)

    augmenter = get_augmenter("carpet")
    augmented_batch = augment_batch(batch, augmenter)

    assert augmented_batch.shape == (n_val, h, w, c_val)
    assert augmented_batch.dtype == np.uint8
    assert np.all(augmented_batch >= 0)
    assert np.all(augmented_batch <= 255)


def test_otsucanny_segmentor() -> None:
    """Verify foreground/background binary mask segmentation using Otsu thresholding + Canny edges."""
    segmentor = OtsuCannySegmentor(morph_kernel_size=3, canny_sigma=0.33)

    # Create a synthetic image with a bright centered object on a dark background
    h, w = 64, 64
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:, :] = 10
    image[20:44, 20:44, :] = 250

    masked_img, mask = segmentor.apply(image)

    assert mask.shape == (h, w)
    assert masked_img.shape == (h, w, 3)
    assert np.all(np.isin(mask, [0, 255]))

    # Center is foreground (255) and edges are background (0)
    assert mask[32, 32] == 255
    assert mask[5, 5] == 0
    assert np.all(masked_img[5, 5] == 0)


def test_extract_largest_component() -> None:
    """Verify that extract_largest_component retains the largest connected component and removes noise."""
    h, w = 64, 64
    mask = np.zeros((h, w), dtype=np.uint8)

    # Large component (area: 400)
    mask[10:30, 10:30] = 255

    # Small noise component (area: 100)
    mask[40:50, 40:50] = 255

    cleaned_mask = extract_largest_component(mask)

    assert cleaned_mask.shape == (h, w)
    assert cleaned_mask.dtype == np.uint8
    assert cleaned_mask[20, 20] == 255
    assert cleaned_mask[45, 45] == 0
