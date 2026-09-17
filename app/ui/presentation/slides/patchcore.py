"""PatchCore architecture slide."""

# ruff: noqa: E501

from app.ui.presentation.theme import slide_html


def render() -> None:
    """Render the PatchCore deep feature transfer and coreset memory architecture."""
    node = "border-radius:1cqw;padding:.9cqh .8cqw;text-align:center;font-size:1.02cqw;font-weight:750;line-height:1.15"
    body = f"""
    <div style="display:grid;grid-template-rows:1fr auto;gap:1.5cqh;height:calc(100% - 2.5cqh);align-items:center">
      <div style="display:grid;grid-template-columns:1.05fr .18fr 1.15fr .18fr 1.05fr .22fr 1.15fr;align-items:center">
        <div style="display:grid;gap:1cqh">
          <div style="{node};background:#dceee9;color:var(--ink)">Normal Train Images</div>
          <div style="text-align:center;color:var(--teal);font-size:1.4cqw">↓</div>
          <div style="{node};background:#dceee9;color:var(--ink)">Frozen ResNet-18<br><span style='font-size:.82cqw;font-weight:500'>Layers 2 & 3 Transfer</span></div>
          <div style="text-align:center;color:var(--teal);font-size:1.4cqw">↓</div>
          <div style="{node};background:#dceee9;color:var(--ink)">Patch Embeddings</div>
        </div>
        <div style="text-align:center;color:var(--teal);font-size:1.8cqw">→</div>
        <div style="{node};background:#0f3038;color:#eef7f6;padding:2.5cqh 1cqw">
          <span class="badge" style="background:rgba(141,216,201,.2);color:#8dd8c9;border-color:#8dd8c9;margin-bottom:.8cqh">Coreset Subsampling</span><br>
          <b>Memory Bank</b><br>
          <span style="font-size:.88cqw;color:#b8ccce;font-weight:400">90% compression via minimax greedy selection</span>
        </div>
        <div style="text-align:center;color:var(--coral);font-size:1.8cqw">↔</div>
        <div style="display:grid;gap:1cqh">
          <div style="{node};background:#f7e2d5;color:var(--ink)">Test Image</div>
          <div style="text-align:center;color:var(--coral);font-size:1.4cqw">↓</div>
          <div style="{node};background:#f7e2d5;color:var(--ink)">Query Patch Features</div>
        </div>
        <div style="text-align:center;color:var(--coral);font-size:1.8cqw">→</div>
        <div>
          <div class="tiny-label" style="margin-bottom:.6cqh">1-NN distance in ℝᵈ</div>
          <div style="{node};background:#f06c54;color:white;padding:1.8cqh .9cqw">Anomaly Map</div>
          <div style="text-align:center;color:var(--coral);font-size:1.4cqw;margin:.4cqh 0">↓</div>
          <div style="{node};border:2px solid #f06c54;color:var(--ink)">Image-Level Score</div>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1.4cqw">
        <div class="card-box" style="padding:1cqh 1.1cqw">
          <b style="font-size:1cqw">Zero Client Backprop</b><br>
          <span style="font-size:.88cqw;color:var(--muted)">Weights remain frozen. Onboarding a new part requires only a single forward pass.</span>
        </div>
        <div class="card-box" style="padding:1cqh 1.1cqw">
          <b style="font-size:1cqw">Data Sovereignty & IP</b><br>
          <span style="font-size:.88cqw;color:var(--muted)">Part geometry lives in the local memory bank, never serialized or shared in network weights.</span>
        </div>
        <div class="card-box" style="padding:1cqh 1.1cqw">
          <b style="font-size:1cqw">Fast Edge Retrieval</b><br>
          <span style="font-size:.88cqw;color:var(--muted)">Coreset reduction preserves coverage while bounding nearest-neighbor latency for conveyor speeds.</span>
        </div>
      </div>
    </div>
    <div class="source">Source: Roth et al. (2022) · PatchCore ResNet-18 Production Implementation</div>
    """
    slide_html("The Production Engine: PatchCore", body, eyebrow="07 · Technical Architecture")
