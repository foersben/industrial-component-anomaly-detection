"""Exploratory data analysis slide."""

# ruff: noqa: E501

from app.ui.presentation.theme import DATASET, img, slide_html


def _finding(category: str, defect: str, index: str, headline: str, copy: str) -> str:
    image = DATASET / category / "test" / defect / f"{index}.png"
    mask = DATASET / category / "ground_truth" / defect / f"{index}_mask.png"
    return f"""
    <div style="display:grid;grid-template-rows:29cqh auto;min-width:0">
      <div style="display:grid;grid-template-columns:1fr .36fr;gap:.65cqw;min-height:0">
        <div class="image-frame">{img(image, f"{category} defect", "photo")}</div>
        <div class="image-frame" style="background:#050505">{img(mask, f"{category} defect mask", "photo contain")}</div>
      </div>
      <div style="padding-top:1.6cqh">
        <div style="font-size:1.3cqw;font-weight:780">{headline}</div>
        <div style="font-size:1.02cqw;line-height:1.35;color:var(--muted);margin-top:.6cqh">{copy}</div>
      </div>
    </div>
    """


def render() -> None:
    """Render three visual findings from original data and masks."""
    cards = [
        _finding("cable", "missing_cable", "010", "Structure matters", "Missing parts alter relationships, not just colour."),
        _finding("grid", "glue", "010", "Texture is unforgiving", "Normal repetition can produce residual noise."),
        _finding("screw", "thread_top", "019", "Defects stay local", "Tiny regions demand patch-level evidence."),
    ]
    body = f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:2.2cqw;height:100%">
      {''.join(cards)}
    </div>
    <div class="source">Original MVTec AD images and ground-truth masks</div>
    """
    slide_html("What EDA taught us", body, eyebrow="02 · Visual evidence")
