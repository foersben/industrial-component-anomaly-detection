"""Visualization and visual evaluation utilities for the PatchCore pipeline."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from anomalib.utils.path import generate_output_filename
from anomalib.visualization import ImageVisualizer
from anomalib.visualization.image.item_visualizer import visualize_image_item
from PIL import Image, ImageDraw, ImageFont


def _add_panel_headers(
    grid: Image.Image,
    labels: list[str],
    panel_width: int = 256,
    header_height: int = 34,
) -> Image.Image:
    """Place unobstructed labels in a header above a horizontal image grid.

    Args:
        grid: Horizontal grid image containing concatenated visualization panels.
        labels: Text labels for each panel column.
        panel_width: Width in pixels of each panel column.
        header_height: Height in pixels of the newly created top header.

    Returns:
        New image with a clean top header strip containing centered panel labels.
    """
    labelled = Image.new("RGB", (grid.width, grid.height + header_height), "white")
    labelled.paste(grid.convert("RGB"), (0, header_height))
    draw = ImageDraw.Draw(labelled)
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for index, label in enumerate(labels):
        left = index * panel_width
        if left >= grid.width:
            break
        right = min(left + panel_width, grid.width)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        x = left + max(0, (right - left - text_width) // 2)
        y = max(0, (header_height - text_height) // 2 - text_box[1])
        draw.text((x, y), label, fill="black", font=font)
    return labelled


class _RawScoreImageVisualizer(ImageVisualizer):
    """Render raw continuous anomaly maps with optional calibrated masks and unobstructed headers."""

    def __init__(
        self,
        pixel_threshold: float | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """Configure four-panel rendering without drawing labels over image data.

        Args:
            pixel_threshold: Optional calibrated threshold for binarizing predicted masks.
            output_dir: Directory where generated four-panel visualizer images are saved.
        """
        super().__init__(text_config={"enable": False}, output_dir=output_dir)
        self.pixel_threshold = pixel_threshold
        self.render_enabled = True

    def on_test_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Render raw anomaly maps without mutating the batch used for evaluation."""
        del pl_module, outputs, batch_idx, dataloader_idx
        if not self.render_enabled:
            return

        anomaly_map = getattr(batch, "anomaly_map", None)
        if not isinstance(anomaly_map, torch.Tensor) or anomaly_map.ndim < 2:
            return

        flattened = anomaly_map.reshape(anomaly_map.shape[0], -1)
        minimum = flattened.min(dim=1).values
        maximum = flattened.max(dim=1).values
        view_shape = (anomaly_map.shape[0],) + (1,) * (anomaly_map.ndim - 1)
        minimum = minimum.reshape(view_shape)
        score_range = (maximum - flattened.min(dim=1).values).reshape(view_shape)
        normalized_map = torch.where(
            score_range > torch.finfo(anomaly_map.dtype).eps,
            (anomaly_map - minimum) / score_range,
            torch.zeros_like(anomaly_map),
        )
        updates: dict[str, Any] = {"anomaly_map": normalized_map}
        if self.pixel_threshold is not None:
            updates["pred_mask"] = anomaly_map > self.pixel_threshold
        visualization_batch = batch.update(in_place=False, **updates)

        if self.output_dir is None:
            self.output_dir = Path(trainer.default_root_dir) / "images"

        labels = ["Image", "Ground-Truth Mask", "Anomaly Map Overlay", "Predicted Mask"]
        for item in visualization_batch:
            image = visualize_image_item(
                item,
                fields=self.fields,
                overlay_fields=self.overlay_fields,
                field_size=self.field_size,
                fields_config=self.fields_config,
                overlay_fields_config=self.overlay_fields_config,
                text_config={"enable": False},
            )
            if image is None:
                continue
            image = _add_panel_headers(image, labels, panel_width=self.field_size[0])
            datamodule = getattr(trainer, "datamodule", None)
            dataset_name = getattr(datamodule, "name", None) if datamodule else None
            category = getattr(datamodule, "category", None) if datamodule else None
            filename = generate_output_filename(
                input_path=item.image_path or "",
                output_path=self.output_dir,
                dataset_name=dataset_name,
                category=category,
            )
            image.save(filename)


def _print_patchcore_results_table(results: Mapping[str, float]) -> None:
    """Print corrected PatchCore metrics in Anomalib's table layout."""
    from rich.console import Console
    from rich.table import Table

    table = Table(show_header=True, header_style="bold")
    table.add_column("Test metric")
    table.add_column("DataLoader 0", justify="right")
    for name, value in results.items():
        table.add_row(name, f"{value:.6f}")
    Console().print(table)


__all__ = [
    "_RawScoreImageVisualizer",
    "_add_panel_headers",
    "_print_patchcore_results_table",
]
