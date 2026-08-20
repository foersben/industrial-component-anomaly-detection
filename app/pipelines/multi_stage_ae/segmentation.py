"""Foreground extraction and background replacement for the Keras CAE pipeline.

Why Do We Need Foreground Extraction?
======================================
Industrial components photographed against a background introduce a fundamental problem:
the autoencoder wastes representational capacity learning the background (conveyor belt,
mounting jig, inspection stage). This background is **not** the object under inspection.

Even worse, subtle background variations (dust, lighting reflections, shadow changes)
can drive up the reconstruction error and produce **false positive** anomaly detections.

The Two-Step Solution: Segmentation + Background Replacement (BGRP-G)
----------------------------------------------------------------------
This module implements the **BGRP-G (Background Replacement to Grey/Black)** strategy:

1. **Segment the foreground**: Use classical computer vision to find the component pixels.
2. **Zero-fill the background**: Replace all background pixels with solid black (0, 0, 0).

Why Black (zero) as the Replacement Colour?
    Black = (0, 0, 0) is the most "out-of-distribution" value for typical industrial
    inspection images, which tend to be brighter and coloured. The autoencoder, trained
    only on images with black backgrounds, will learn to perfectly reconstruct black
    background regions with near-zero error. This means the background contributes
    nothing to the anomaly score - which is exactly what we want.

    Important: We must keep colour information in the foreground intact, since colour
    defects (e.g., surface discolouration) are valid anomaly types in MVTec.

Classical CV Approach: Otsu + Adaptive Canny
--------------------------------------------
Rather than using SAM (Segment Anything Model, which requires ~2.5 GB model weights),
we use a fast, dependency-free classical pipeline:

1. **Otsu's Thresholding**: A global binarization method that automatically finds the
   optimal greyscale threshold to separate foreground from background. It maximises the
   inter-class variance between foreground and background pixel distributions.

2. **Adaptive Canny Edge Detection**: Canny finds sharp pixel intensity transitions
   (edges). The "adaptive" variant sets high/low thresholds automatically from the
   image's median pixel intensity, making it robust across varying illumination.

3. **Morphological Closing**: Fills small holes in the combined binary mask (gaps between
   edges and the Otsu region) by dilating then eroding with a kernel.

4. **Largest Connected Component**: Selects only the single largest foreground blob,
   discarding small spurious fragments from dust or image noise.

Module Contents
---------------
- ``OtsuCannySegmentor``: Full foreground extraction pipeline.
- ``extract_largest_component``: Helper to isolate the largest blob in a binary mask.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    """Extract only the largest connected foreground region from a binary mask.

    After Otsu + Canny segmentation, the mask may contain multiple disconnected blobs
    (e.g., the main component plus dust particles or image artefacts). This function
    keeps only the largest blob, which is almost always the actual component.

    How it works:
        1. Label all connected components in the binary mask.
        2. Count pixels in each component.
        3. Return a mask with only the largest component filled.

    Args:
        binary_mask: 2D binary numpy array (dtype uint8), 255 = foreground, 0 = background.

    Returns:
        Cleaned 2D binary mask with only the largest connected component kept (uint8, 0/255).
    """
    # cv2.connectedComponentsWithStats returns:
    # - num_labels: total number of components found (including background = label 0)
    # - labels: 2D array where each pixel has its component label
    # - stats: per-component statistics (bounding box, area)
    # - centroids: per-component centroid (x, y)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    if num_labels <= 1:
        # Only background found (empty mask), return as-is
        return binary_mask

    # stats[i, cv2.CC_STAT_AREA] gives pixel count of component i
    # Label 0 is the background - skip it by starting from label 1
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(component_areas)) + 1  # +1 to re-align with labels array

    # Build new mask with only the largest component
    largest_mask = np.zeros_like(binary_mask)
    largest_mask[labels == largest_label] = 255

    return largest_mask


class OtsuCannySegmentor:
    """Foreground segmentation combining Otsu thresholding with Adaptive Canny edge detection.

    This class provides a fast, reliable foreground extraction pipeline that works well
    on the standard MVTec AD inspection setup (component on a uniform background).

    Pipeline Steps:
        1. Convert RGB to greyscale for efficient threshold computation.
        2. Apply Otsu's global threshold to create a coarse binary foreground mask.
        3. Compute adaptive Canny edge map using the image's median intensity as the
           threshold anchor.
        4. Combine (OR) the Otsu mask and Canny edges into one binary map.
        5. Apply morphological closing to fill gaps between adjacent edges.
        6. Extract the single largest connected component to remove noise artefacts.
        7. Replace all background pixels (mask = 0) in the original RGB image with black.

    Attributes:
        morph_kernel_size: Side length (pixels) of the square structuring element used
            for morphological closing. Larger = fills bigger gaps.
        canny_sigma: Scaling factor applied to the median pixel intensity to derive
            the Canny low and high thresholds. Higher = fewer, stronger edges detected.
    """

    def __init__(self, morph_kernel_size: int = 5, canny_sigma: float = 0.33) -> None:
        """Initialise the segmentor with morphological and edge detection parameters.

        Args:
            morph_kernel_size: Side length of the square kernel for morphological closing.
                5 pixels works well for most MVTec categories.
            canny_sigma: Controls the spread of Canny threshold bounds around the
                image median. Larger values = more conservative edge detection.
        """
        self.morph_kernel_size = morph_kernel_size
        self.canny_sigma = canny_sigma
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (morph_kernel_size, morph_kernel_size),
        )

    def compute_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """Compute a binary foreground mask for the input RGB image.

        Args:
            image_rgb: Input image as a numpy array of shape (H, W, 3), dtype uint8, RGB order.

        Returns:
            Binary mask as a 2D numpy array (H, W), dtype uint8, values 0 or 255.
            255 = foreground (component), 0 = background.
        """
        # Step 1: Convert RGB to greyscale
        # Greyscale = 0.299R + 0.587G + 0.114B (standard luminance formula)
        grey = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Step 2: Otsu global thresholding
        # Otsu automatically finds the threshold that maximises inter-class variance.
        # THRESH_BINARY_INV inverts so the foreground (usually darker component on
        # lighter background) becomes 255.
        _, otsu_mask = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 3: Adaptive Canny edge detection
        # Derive thresholds from the image's median pixel intensity:
        # - low  threshold = (1 - sigma) * median
        # - high threshold = (1 + sigma) * median
        median_val = float(np.median(grey))
        low_thresh = max(0.0, (1.0 - self.canny_sigma) * median_val)
        high_thresh = min(255.0, (1.0 + self.canny_sigma) * median_val)
        canny_edges = cv2.Canny(grey, low_thresh, high_thresh)

        # Step 4: Combine Otsu mask and Canny edges (logical OR)
        combined = cv2.bitwise_or(otsu_mask, canny_edges)

        # Step 5: Morphological closing to fill small internal holes
        # Closing = dilate then erode → expands foreground then shrinks back,
        # but keeps filled any gaps smaller than the kernel.
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self._kernel)

        # Step 6: Keep only the largest connected foreground blob
        largest = extract_largest_component(closed)

        return largest

    def apply(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract foreground and replace background with black (BGRP-G strategy).

        The BGRP-G strategy (Background Replacement with a Guaranteed Out-of-Distribution
        colour) zeroes out all background pixels. This forces the autoencoder to learn only
        from the component surface, eliminating background as a source of false anomalies.

        Args:
            image_rgb: Input RGB image as numpy array (H, W, 3), dtype uint8.

        Returns:
            Tuple of:
                - masked_image: RGB image with background pixels zeroed, shape (H, W, 3).
                - foreground_mask: Binary foreground mask, shape (H, W), values 0 or 255.
        """
        foreground_mask = self.compute_mask(image_rgb)

        # Expand mask to 3 channels so it can multiply with RGB image
        mask_3ch = np.stack([foreground_mask] * 3, axis=-1)  # (H, W, 3)

        # Apply mask: foreground stays, background becomes 0 (black)
        masked_image = (image_rgb * (mask_3ch > 0)).astype(np.uint8)

        logger.debug(
            "Segmentation complete. Foreground pixels: %d / %d (%.1f%%)",
            int(np.sum(foreground_mask > 0)),
            foreground_mask.size,
            100.0 * np.sum(foreground_mask > 0) / foreground_mask.size,
        )

        return masked_image, foreground_mask
