"""Recompose selected PatchCore visualizations for presentation slide 6.

The source four-panel PNGs contain aligned 256 px panels: source image,
ground-truth mask, anomaly-map overlay, and the predicted-mask contour.
This script overlays the two existing boundaries on the anomaly-map panel;
it does not estimate a new threshold or alter the model scores.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "Patchcore"
OUTPUT = ROOT / "docs" / "latex" / "latex_beamer_presentation" / "figures" / "report_figures"
PANEL_SIZE = 256
HEADER_HEIGHT = 34
HEATMAP_BLEND = 0.58

SAMPLES = {
    "wood_liquid": RESULTS / "v30" / "images_regenerated" / "liquid" / "003.png",
    "zipper_combined": RESULTS / "v31" / "images_regenerated" / "combined" / "004.png",
    "transistor_misplaced": RESULTS / "v29" / "images_regenerated" / "misplaced" / "001.png",
    "toothbrush_defective": RESULTS / "v28" / "images_regenerated" / "defective" / "015.png",
}


def _panels(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(path).convert("RGB"))
    expected_shape = (HEADER_HEIGHT + PANEL_SIZE, 4 * PANEL_SIZE, 3)
    if image.shape != expected_shape:
        raise ValueError(f"Unexpected panel layout in {path}: {image.shape}")
    body = image[HEADER_HEIGHT:]
    return tuple(body[:, i * PANEL_SIZE : (i + 1) * PANEL_SIZE].copy() for i in range(4))  # type: ignore[return-value]


def _predicted_outline(source: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Isolate the red outline Anomalib drew over the source image."""
    red = prediction[:, :, 0].astype(np.int16)
    green = prediction[:, :, 1].astype(np.int16)
    blue = prediction[:, :, 2].astype(np.int16)
    changed = np.max(np.abs(prediction.astype(np.int16) - source.astype(np.int16)), axis=2) > 45
    return (red > 160) & (red - green > 75) & (red - blue > 75) & changed


def _draw_gt_contour(canvas: np.ndarray, gt_mask: np.ndarray) -> None:
    contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, contours, -1, (20, 27, 30), 2, cv2.LINE_AA)
    cv2.drawContours(canvas, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)


def build_overlay(source_path: Path, output_path: Path) -> tuple[int, int]:
    """Save one heatmap with GT and calibrated-prediction boundaries."""
    source, gt_panel, heatmap, prediction = _panels(source_path)
    gt_mask = np.min(gt_panel, axis=2) > 127
    predicted_outline = _predicted_outline(source, prediction)
    gt_pixels = int(gt_mask.sum())
    predicted_pixels = int(predicted_outline.sum())
    if gt_pixels < 20 or predicted_pixels < 20:
        raise ValueError(f"Missing mask evidence in {source_path}: GT={gt_pixels}, prediction={predicted_pixels}")

    # Blend the already-rendered heatmap with its aligned source image. This
    # changes presentation opacity only, not the underlying model prediction.
    canvas = cv2.addWeighted(heatmap, HEATMAP_BLEND, source, 1 - HEATMAP_BLEND, 0)
    # Preserve the existing predicted boundary with a narrow contrast halo.
    outline = predicted_outline.astype(np.uint8)
    halo = cv2.dilate(outline, np.ones((3, 3), np.uint8)) > 0
    stroke = cv2.dilate(outline, np.ones((2, 2), np.uint8)) > 0
    canvas[halo] = (20, 27, 30)
    canvas[stroke] = (96, 255, 56)
    _draw_gt_contour(canvas, gt_mask)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(output_path)
    return gt_pixels, predicted_pixels


def main() -> None:
    """Build the four slide-ready localization examples."""
    for name, source_path in SAMPLES.items():
        destination = OUTPUT / f"slide_06_{name}.png"
        gt_pixels, predicted_pixels = build_overlay(source_path, destination)
        print(f"{destination.relative_to(ROOT)}: GT={gt_pixels}, predicted-outline={predicted_pixels}")


if __name__ == "__main__":
    main()
