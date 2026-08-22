import numpy as np

from app.pipelines.multi_stage_ae.augmentation import ObjectAugmenter, TextureAugmenter, augment_batch, get_augmenter
from app.pipelines.multi_stage_ae.segmentation import OtsuCannySegmentor, extract_largest_component


def test_get_augmenter_routing() -> None:
    """Test routing logic for getting augmenters based on category."""
    # Texture categories should return TextureAugmenter
    assert isinstance(get_augmenter("carpet"), TextureAugmenter)
    assert isinstance(get_augmenter("grid"), TextureAugmenter)

    # Object categories should return ObjectAugmenter
    assert isinstance(get_augmenter("bottle"), ObjectAugmenter)
    assert isinstance(get_augmenter("cable"), ObjectAugmenter)

    # Unknown category should default to ObjectAugmenter
    assert isinstance(get_augmenter("unknown"), ObjectAugmenter)


def test_augment_batch() -> None:
    """Test batch augmentation preserves shapes and value ranges."""
    n_val, h, w, c_val = 2, 32, 32, 3
    # Create uint8 batch in range [0, 255]
    batch = np.random.randint(0, 256, size=(n_val, h, w, c_val), dtype=np.uint8)

    augmenter = get_augmenter("carpet")
    augmented_batch = augment_batch(batch, augmenter)

    # Check shape
    assert augmented_batch.shape == (n_val, h, w, c_val)

    # Check dtype
    assert augmented_batch.dtype == np.uint8

    # Check value bounds
    assert np.all(augmented_batch >= 0)
    assert np.all(augmented_batch <= 255)


def test_otsucanny_segmentor() -> None:
    """Test foreground/background segmentation using Otsu + Canny."""
    segmentor = OtsuCannySegmentor(morph_kernel_size=3, canny_sigma=0.33)

    # Create a synthetic image with a clear bright object on a dark background
    h, w = 64, 64
    image = np.zeros((h, w, 3), dtype=np.uint8)

    # Needs to be a very strong contrast without noise for deterministic Otsu Thresholding
    # Very dark background, pure white square in the middle
    image[:, :] = 10
    image[20:44, 20:44, :] = 250

    masked_img, mask = segmentor.apply(image)

    # Check shape
    assert mask.shape == (h, w)
    assert masked_img.shape == (h, w, 3)

    # Check mask values are binary (0 and 255)
    assert np.all(np.isin(mask, [0, 255]))

    # The center should be foreground (255) and edges should be background (0)
    assert mask[32, 32] == 255
    assert mask[5, 5] == 0

    # The masked image should be zero where mask is zero
    assert np.all(masked_img[5, 5] == 0)


def test_extract_largest_component() -> None:
    """Test extracting the largest connected component from a binary mask."""
    h, w = 64, 64
    mask = np.zeros((h, w), dtype=np.uint8)

    # Create a large component
    mask[10:30, 10:30] = 255  # Area: 20x20 = 400

    # Create a smaller noise component
    mask[40:50, 40:50] = 255  # Area: 10x10 = 100

    cleaned_mask = extract_largest_component(mask)

    # Shape and type
    assert cleaned_mask.shape == (h, w)
    assert cleaned_mask.dtype == np.uint8

    # The large component should remain
    assert cleaned_mask[20, 20] == 255

    # The smaller component should be removed (set to 0)
    assert cleaned_mask[45, 45] == 0
