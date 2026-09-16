"""Exploratory data analysis slide."""

from app.ui.presentation.theme import DEFECT_EXAMPLES, img, slide_html


def _finding(asset_name: str, headline: str, copy: str) -> str:
    image = DEFECT_EXAMPLES / f"slide_03_{asset_name}_contour.png"
    visual = img(image, f"{headline} defect with ground-truth contour", "photo contain")
    return f"""
    <div style="display:grid;grid-template-rows:29cqh auto;width:29cqh;margin:0 auto;min-width:0">
      <div class="image-frame" style="background:transparent">{visual}</div>
      <div style="padding-top:1.6cqh">
        <div style="font-size:1.3cqw;font-weight:780">{headline}</div>
        <div style="font-size:1.02cqw;line-height:1.35;color:var(--muted);margin-top:.6cqh">{copy}</div>
      </div>
    </div>
    """


def render() -> None:
    """Render three visual findings from original data and masks."""
    cards = [
        _finding("cable_missing", "Structure matters", "Missing parts alter relationships, not just colour."),
        _finding("grid_glue", "Texture is unforgiving", "Normal repetition can produce residual noise."),
        _finding("screw_thread", "Defects stay local", "Tiny regions demand patch-level evidence."),
    ]
    body = f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:2.2cqw;height:100%">
      {''.join(cards)}
    </div>
    <div class="source">Original MVTec AD images with ground-truth mask contours</div>
    """
    slide_html("What EDA taught us", body, eyebrow="02 · Visual evidence")
