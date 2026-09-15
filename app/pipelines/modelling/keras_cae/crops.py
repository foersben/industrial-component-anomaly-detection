"""Image cropping and crop stitching utilities for convolutional autoencoders."""

import numpy as np


def extract_crops(images: np.ndarray, crop_size: int, crop_stride: int) -> np.ndarray:
    """Extract overlapping regular square crops from a batch of images.

    Args:
        images: Array of input images of shape (N, H, W, C).
        crop_size: Height and width of the square crops.
        crop_stride: Horizontal and vertical stride between consecutive crops.

    Returns:
        Flattened array of extracted crops with shape (N * num_crops, crop_size, crop_size, C).
    """
    _, h, w, c = images.shape
    crops = []
    for i in range(0, h - crop_size + 1, crop_stride):
        for j in range(0, w - crop_size + 1, crop_stride):
            crops.append(images[:, i : i + crop_size, j : j + crop_size, :])

    crops_stack = np.stack(crops, axis=1)
    return crops_stack.reshape(-1, crop_size, crop_size, c)


def stitch_crops(
    crops: np.ndarray,
    n_images: int,
    img_h: int,
    img_w: int,
    crop_size: int,
    crop_stride: int,
) -> np.ndarray:
    """Stitch overlapping predicted crops back into full-resolution images via averaging.

    Overlapping regions are accumulated and normalized by the number of overlapping patches.

    Args:
        crops: Batch of cropped image patches of shape (n_images * num_crops, crop_size, crop_size, C).
        n_images: Number of full images represented in the crops batch.
        img_h: Target full image height.
        img_w: Target full image width.
        crop_size: Size of the square crops.
        crop_stride: Stride between consecutive crops used during extraction.

    Returns:
        Reconstructed full image array of shape (n_images, img_h, img_w, C).
    """
    c = crops.shape[-1]
    reconstructed = np.zeros((n_images, img_h, img_w, c), dtype=np.float32)
    counts = np.zeros((n_images, img_h, img_w, 1), dtype=np.float32)

    num_crops = crops.shape[0] // n_images
    crops_reshaped = crops.reshape(n_images, num_crops, crop_size, crop_size, c)

    idx = 0
    for i in range(0, img_h - crop_size + 1, crop_stride):
        for j in range(0, img_w - crop_size + 1, crop_stride):
            reconstructed[:, i : i + crop_size, j : j + crop_size, :] += crops_reshaped[:, idx]
            counts[:, i : i + crop_size, j : j + crop_size, :] += 1.0
            idx += 1

    counts = np.maximum(counts, 1.0)
    return np.asarray(reconstructed / counts, dtype=np.float32)
