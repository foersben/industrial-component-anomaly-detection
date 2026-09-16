"""PatchCore architecture slide."""

# ruff: noqa: E501

from app.ui.presentation.theme import slide_html


def render() -> None:
    """Render a clean PatchCore data-flow diagram."""
    node = "border-radius:1.05cqw;padding:1.55cqh 1.15cqw;text-align:center;font-size:1.15cqw;font-weight:760;line-height:1.18"
    body = f"""
    <div style="display:grid;grid-template-columns:1.08fr .22fr 1fr .22fr 1fr .35fr 1.15fr;height:100%;align-items:center">
      <div style="display:grid;gap:1.7cqh">
        <div style="{node};background:#dceee9">Normal training images</div>
        <div style="text-align:center;color:var(--teal);font-size:2cqw">↓</div>
        <div style="{node};background:#dceee9">ResNet-18<br><span style='font-weight:500'>frozen features</span></div>
        <div style="text-align:center;color:var(--teal);font-size:2cqw">↓</div>
        <div style="{node};background:#dceee9">Patch embeddings</div>
      </div>
      <div style="text-align:center;color:var(--teal);font-size:2.3cqw">→</div>
      <div style="{node};background:#0f3038;color:#eef7f6;padding:4.2cqh 1.3cqw">Normal<br>memory bank</div>
      <div style="text-align:center;color:var(--coral);font-size:2.3cqw">↔</div>
      <div style="display:grid;gap:1.7cqh">
        <div style="{node};background:#f7e2d5">Test image</div>
        <div style="text-align:center;color:var(--coral);font-size:2cqw">↓</div>
        <div style="{node};background:#f7e2d5">Patch features</div>
      </div>
      <div style="text-align:center;color:var(--coral);font-size:2.3cqw">→</div>
      <div>
        <div class="tiny-label" style="margin-bottom:1.2cqh">nearest-neighbour distance</div>
        <div style="{node};background:#f06c54;color:white;padding:3.5cqh 1.4cqw">Anomaly map</div>
        <div style="text-align:center;color:var(--coral);font-size:2cqw;margin:1cqh 0">↓</div>
        <div style="{node};border:2px solid #f06c54">Image score</div>
      </div>
    </div>
    """
    slide_html("How PatchCore works", body, eyebrow="05 · Feature matching")
