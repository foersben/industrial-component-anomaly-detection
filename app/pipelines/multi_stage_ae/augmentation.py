"""Category-aware data augmentation for the Keras CAE anomaly detection pipeline.

Why Does Augmentation Strategy Matter for Anomaly Detection?
============================================================
In industrial anomaly detection, the autoencoder is trained **exclusively on defect-free
normal images**. The goal of augmentation is NOT to help the model generalise to new
defect classes (that would be wrong), but to:

1. **Prevent overfitting** to the exact photographic conditions of the training set
   (lighting angles, minor camera vibration, batch-to-batch variation).
2. **Make the model robust** to permissible natural variance (e.g., slightly different
   grain orientation in wood), while still flagging genuine defects as anomalous.

The Critical Distinction: Textures vs. Rigid Objects
-----------------------------------------------------
This is the most important design decision in augmentation:

**Texture categories** (wood, carpet, leather, tile, grid):
    These materials have **spatial invariance** - the statistical pattern of wood grain
    looks essentially the same whether you rotate it 90 degrees or not. Therefore, heavy
    geometric augmentations (random rotations, flips, scale jitter) are very effective.
    They teach the model "what does normal wood look like from any direction?".

**Rigid object categories** (transistor, pill, screw, capsule, metal_nut, bolt):
    These objects are **directionally aligned** on the conveyor belt or inspection jig.
    A transistor always arrives with its legs pointing down. If you rotate it 90 degrees
    during training, the model learns that an upside-down transistor is normal - which
    destroys the anomaly detection capability for orientation defects entirely.
    For these, only light colour/intensity augmentations are safe.

Module Contents
---------------
- ``TEXTURE_CATEGORIES``: Set of MVTec categories that are textures.
- ``OBJECT_CATEGORIES``: Set of MVTec categories that are rigid objects.
- ``TextureAugmenter``: Heavy augmentation pipeline for textures.
- ``ObjectAugmenter``: Light augmentation pipeline for rigid objects.
- ``get_augmenter``: Factory that returns the correct augmenter by category name.
"""

import logging
import random
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)

# MVTec AD has 15 categories split into textures and objects.
# Source: https://www.mvtec.com/company/research/datasets/mvtec-ad
TEXTURE_CATEGORIES: frozenset[str] = frozenset(
    {
        "carpet",
        "grid",
        "leather",
        "tile",
        "wood",
    }
)
"""Set of MVTec AD texture category names.

For these categories, spatial augmentations (rotations, flips) are safe and beneficial
because the texture patterns are statistically invariant under spatial transformations.
"""

OBJECT_CATEGORIES: frozenset[str] = frozenset(
    {
        "bottle",
        "cable",
        "capsule",
        "hazelnut",
        "metal_nut",
        "pill",
        "screw",
        "toothbrush",
        "transistor",
        "zipper",
    }
)
"""Set of MVTec AD rigid object category names.

For these categories, spatial transformations would destroy the learned object orientation,
so only light photometric (colour/brightness/noise) augmentations are applied.
"""


class TextureAugmenter:
    """Heavy augmentation pipeline for spatially invariant texture categories.

    Spatial invariance means the visual statistics of the material do not fundamentally
    change under rotation or reflection. Wood grain rotated 90 degrees still looks like
    normal wood - so we exploit this to generate more training variety.

    Augmentations applied in random order:
    1. Random 90°/180°/270° rotation (or no rotation).
    2. Random horizontal flip.
    3. Random vertical flip.
    4. Random scale crop (zooms into 80-100% of the image, then resizes back).
    5. Slight brightness jitter (±20% brightness variation).
    6. Slight contrast jitter (±20% contrast variation).

    Attributes:
        brightness_range: Tuple (min, max) multiplier for brightness jitter.
        contrast_range: Tuple (min, max) multiplier for contrast jitter.
        scale_range: Tuple (min_crop_fraction, max_crop_fraction) for scale jitter.
    """

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.8, 1.2),
        contrast_range: tuple[float, float] = (0.8, 1.2),
        scale_range: tuple[float, float] = (0.8, 1.0),
    ) -> None:
        """Initialize the texture augmenter with configurable jitter ranges.

        Args:
            brightness_range: (min, max) brightness multiplier. 1.0 = original.
            contrast_range: (min, max) contrast multiplier. 1.0 = original.
            scale_range: (min_fraction, max_fraction) of image area to crop before resize.
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.scale_range = scale_range

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply the full texture augmentation pipeline to a single PIL image.

        Args:
            image: Input PIL Image in RGB mode.

        Returns:
            Augmented PIL Image in RGB mode, same size as input.
        """
        original_size = image.size  # (width, height) in PIL convention

        # 1. Random 90-degree rotation (0, 90, 180, or 270 degrees)
        rotation_angle = random.choice([0, 90, 180, 270])
        if rotation_angle != 0:
            image = image.rotate(rotation_angle, expand=False)

        # 2. Random horizontal flip (50% probability)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 3. Random vertical flip (50% probability)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

        # 4. Random scale jitter (crop a random sub-region and resize back)
        crop_fraction = random.uniform(*self.scale_range)
        if crop_fraction < 1.0:
            w, h = image.size
            crop_w = int(w * crop_fraction)
            crop_h = int(h * crop_fraction)
            left = random.randint(0, w - crop_w)
            top = random.randint(0, h - crop_h)
            image = image.crop((left, top, left + crop_w, top + crop_h))
            image = image.resize(original_size, Image.BILINEAR)

        # 5. Random brightness jitter
        brightness_factor = random.uniform(*self.brightness_range)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)

        # 6. Random contrast jitter
        contrast_factor = random.uniform(*self.contrast_range)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)

        return image


class ObjectAugmenter:
    """Light augmentation pipeline for directionally aligned rigid object categories.

    These objects (e.g., transistors, pills, screws) are always positioned in a
    consistent orientation in the MVTec dataset. Applying rotations would teach the
    model that an upside-down transistor is "normal" - completely defeating anomaly
    detection for orientation-related defects.

    Therefore, only **photometric** (colour/intensity) augmentations are applied:
    1. Slight brightness jitter (±10%).
    2. Slight contrast jitter (±10%).
    3. Slight saturation jitter (±10%).
    4. Light additive Gaussian noise (very small standard deviation, ±2% intensity).

    Attributes:
        brightness_range: Tuple (min, max) brightness multiplier.
        contrast_range: Tuple (min, max) contrast multiplier.
        saturation_range: Tuple (min, max) colour saturation multiplier.
        noise_std: Standard deviation of additive Gaussian noise (0.0 to 1.0 scale).
    """

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.9, 1.1),
        contrast_range: tuple[float, float] = (0.9, 1.1),
        saturation_range: tuple[float, float] = (0.9, 1.1),
        noise_std: float = 0.02,
    ) -> None:
        """Initialize the object augmenter with configurable photometric jitter.

        Args:
            brightness_range: (min, max) brightness multiplier. 1.0 = no change.
            contrast_range: (min, max) contrast multiplier. 1.0 = no change.
            saturation_range: (min, max) saturation multiplier. 1.0 = no change.
            noise_std: Standard deviation of Gaussian noise added to normalised [0,1] pixels.
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.noise_std = noise_std

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply the light object augmentation pipeline to a single PIL image.

        Args:
            image: Input PIL Image in RGB mode.

        Returns:
            Augmented PIL Image in RGB mode, same size as input.
        """
        # 1. Slight brightness jitter
        brightness_factor = random.uniform(*self.brightness_range)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)

        # 2. Slight contrast jitter
        contrast_factor = random.uniform(*self.contrast_range)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)

        # 3. Slight colour saturation jitter
        saturation_factor = random.uniform(*self.saturation_range)
        image = ImageEnhance.Color(image).enhance(saturation_factor)

        # 4. Additive Gaussian noise (very light synthetic sensor noise)
        if self.noise_std > 0.0:
            image_array = np.array(image, dtype=np.float32) / 255.0
            noise = np.random.normal(loc=0.0, scale=self.noise_std, size=image_array.shape)
            image_array = np.clip(image_array + noise, 0.0, 1.0)
            image = Image.fromarray((image_array * 255).astype(np.uint8))

        return image


def get_augmenter(category: str) -> TextureAugmenter | ObjectAugmenter:
    """Factory that returns the correct augmenter for a given MVTec category.

    This function automatically selects the appropriate augmentation strategy:
    - Heavy spatial augmentation (``TextureAugmenter``) for texture categories.
    - Light photometric augmentation (``ObjectAugmenter``) for object categories.
    - Falls back to ``ObjectAugmenter`` (conservative) for unknown categories.

    Args:
        category: MVTec AD category name (e.g., 'wood', 'bottle', 'screw').

    Returns:
        TextureAugmenter if the category is a texture, ObjectAugmenter otherwise.
    """
    category_lower = category.lower().strip()
    if category_lower in TEXTURE_CATEGORIES:
        logger.info("Category '%s' is a texture → using TextureAugmenter (heavy spatial augmentation).", category)
        return TextureAugmenter()
    logger.info("Category '%s' is an object → using ObjectAugmenter (light photometric augmentation).", category)
    return ObjectAugmenter()


def augment_batch(
    images: np.ndarray,
    augmenter: TextureAugmenter | ObjectAugmenter,
) -> np.ndarray:
    """Apply augmentation to a batch of numpy images.

    Args:
        images: Batch of images as a numpy array of shape (N, H, W, 3), values in [0, 255].
        augmenter: An instantiated augmenter (TextureAugmenter or ObjectAugmenter).

    Returns:
        Augmented batch as numpy array of shape (N, H, W, 3), values in [0, 255].
    """
    augmented: list[Any] = []
    for img_array in images:
        pil_img = Image.fromarray(img_array.astype(np.uint8), mode="RGB")
        aug_img = augmenter(pil_img)
        augmented.append(np.array(aug_img, dtype=np.uint8))
    return np.stack(augmented, axis=0)
