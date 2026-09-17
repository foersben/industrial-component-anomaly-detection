"""Exploratory data analysis slide."""

# ruff: noqa: E501

from app.ui.presentation.theme import DEFECT_EXAMPLES, img, slide_html


def _card(asset_name: str, title: str, subtitle: str, category_type: str) -> str:
    image = DEFECT_EXAMPLES / f"slide_03_{asset_name}_contour.png"
    visual = img(image, f"{title} defect with ground-truth contour", "photo contain")
    return f"""
    <div class="card-box" style="display:grid;grid-template-rows:22cqh auto;gap:1.2cqh;padding:1.4cqh 1.2cqw">
      <div class="image-frame" style="background:#071117">{visual}</div>
      <div>
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span style="font-size:1.15cqw;font-weight:780;color:var(--ink)">{title}</span>
          <span class="badge" style="font-size:.72cqw">{category_type}</span>
        </div>
        <div style="font-size:.95cqw;line-height:1.35;color:var(--muted);margin-top:.6cqh">{subtitle}</div>
      </div>
    </div>
    """


def render() -> None:
    """Render EDA findings: taxonomy, statistical proof, and visual evidence."""
    cards = [
        _card(
            "cable_missing",
            "Cable: Missing Wire",
            "Structural anomaly altering part geometry and electrical topology.",
            "Object",
        ),
        _card(
            "grid_glue",
            "Grid: Glue Contamination",
            "High-frequency texture where normal repetition easily triggers pseudo-alarms.",
            "Texture",
        ),
        _card(
            "screw_thread",
            "Screw: Thread Scratch",
            "Minute defect (<0.4% area) demanding localized feature resolution.",
            "Object",
        ),
    ]
    body = f"""
    <div style="display:grid;grid-template-rows:auto 1fr;gap:2cqh;height:100%">
      <div style="display:grid;grid-template-columns:1.15fr .85fr;gap:1.8cqw;align-items:stretch">
        <div class="card-box" style="background:#dceee9;border-color:rgba(0,127,122,.25);padding:1.4cqh 1.4cqw">
          <div style="font-size:1.05cqw;font-weight:760;color:var(--teal)">Objects (10) vs. Textures (5) Taxonomy</div>
          <div style="font-size:.92cqw;line-height:1.35;color:var(--ink);margin-top:.4cqh">
            Discrete objects require pose/orientation care and foreground extraction; continuous textures require spatial invariance and photometric stability.
          </div>
        </div>
        <div class="card-box" style="background:#f8e5d8;border-color:rgba(240,108,84,.3);padding:1.4cqh 1.4cqw">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:1.05cqw;font-weight:760;color:var(--coral)">Kruskal-Wallis: H = 626.5</span>
            <span class="badge badge-coral" style="font-size:.7cqw">p &lt; 10⁻¹²⁴</span>
          </div>
          <div style="font-size:.92cqw;line-height:1.35;color:var(--ink);margin-top:.4cqh">
            Proves a 43-fold spread in anomalous pixel area. A universal threshold across products is <b>mathematically invalid</b>.
          </div>
        </div>
      </div>
      <div class="slide-grid-3">
        {"".join(cards)}
      </div>
    </div>
    <div class="source">Source: MVTec AD EDA report · Kruskal-Wallis H-test on 1,258 defective masks</div>
    """
    slide_html("EDA: Taxonomy & Statistical Heterogeneity", body, eyebrow="02 · Data Exploration")
