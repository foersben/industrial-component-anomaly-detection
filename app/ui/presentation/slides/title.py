"""Title slide."""

# ruff: noqa: E501

from app.ui.presentation.theme import DEFECT_EXAMPLES, img, slide_html

CATEGORIES = [
    {"name": "Bottle", "desc": "Broken rim", "good": "title_good_bottle.png", "bad": "title_anomaly_bottle.png"},
    {"name": "Capsule", "desc": "Scratch", "good": "title_good_capsule.png", "bad": "title_anomaly_capsule.png"},
    {"name": "Leather", "desc": "Surface cut", "good": "title_good_leather.png", "bad": "title_anomaly_leather.png"},
    {"name": "Screw", "desc": "Thread defect", "good": "title_good_screw.png", "bad": "title_anomaly_screw.png"},
]


def render() -> None:
    """Render the defense title, project objective, and 4+4 category showcase."""
    category_rows = "".join(
        f"""<tr style="background:rgba(247,246,243,0.6);">
          <td style="padding:0.35cqh 0.6cqw; border-radius:6px 0 0 6px; vertical-align:middle; width:15cqw;">
            <div style="font-size:0.85cqw;font-weight:700;color:var(--ink)">{cat["name"]}</div>
            <div style="font-size:0.7cqw;color:var(--muted);margin-top:0.1cqh">{cat["desc"]}</div>
          </td>
          <td></td>
          <td style="padding:0.35cqh 0.6cqw; vertical-align:middle; width:12cqh;">
            <div style="position:relative;border-radius:6px;overflow:hidden;border:1px solid #dcdad5;width:10cqh;height:10cqh;aspect-ratio:1/1;margin:0 auto;">
              {img(DEFECT_EXAMPLES / cat["good"], f"Conforming {cat['name'].lower()}", "photo")}
              <span style="position:absolute;bottom:3px;left:3px;background:rgba(0,127,122,0.92);color:#ffffff;font-size:0.58cqw;font-weight:700;padding:1px 4px;border-radius:3px">GOOD</span>
            </div>
          </td>
          <td style="padding:0.35cqh 0.6cqw; border-radius:0 6px 6px 0; vertical-align:middle; width:12cqh;">
            <div style="position:relative;border-radius:6px;overflow:hidden;border:1px solid #f2c5be;width:10cqh;height:10cqh;aspect-ratio:1/1;margin:0 auto;">
              {img(DEFECT_EXAMPLES / cat["bad"], f"Defective {cat['name'].lower()}", "photo")}
              <span style="position:absolute;bottom:3px;left:3px;background:rgba(240,108,84,0.92);color:#ffffff;font-size:0.58cqw;font-weight:700;padding:1px 4px;border-radius:3px">DEFECT</span>
            </div>
          </td>
        </tr>"""
        for cat in CATEGORIES
    )

    body = f"""<div class="slide-grid-2" style="grid-template-columns:0.88fr 1.12fr;gap:2cqw;align-items:center;height:100%">
      <div>
        <div style="font-size:1.65cqw;line-height:1.25;color:var(--ink);font-weight:600">
          Unsupervised defect classification and pixel-level anomaly localization from normal-only manufacturing parts.
        </div>
        <div class="def-rule"></div>
        <div style="display:flex;gap:.7cqw;margin-bottom:2cqh;flex-wrap:wrap">
          <span class="badge">Unsupervised AD</span>
          <span class="badge">Deep Transfer Learning</span>
          <span class="badge">Zero-Leakage Protocol</span>
        </div>
        <div class="tiny-label">Team</div>
        <div style="margin-top:.8cqh;font-size:1.05cqw;line-height:1.45;color:var(--muted)">
          Benjamin Förster &middot; Ali Abul Hawa &middot; Mahboubeh Nadaf &middot; Karim Khalifa
        </div>
      </div>
      <div class="card-box" style="padding:1.2cqh 1.2cqw;background:#ffffff;box-shadow:0 4px 8px rgba(0,0,0,0.04)">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8cqh">
          <div class="tiny-label" style="margin:0;font-weight:700">MVTec AD Benchmark Landscape</div>
          <span class="badge" style="font-size:0.72cqw;padding:0.2cqh 0.6cqw">4 Pairs &middot; Objects &amp; Texture</span>
        </div>
        <table style="width:100%; border-collapse:separate; border-spacing:0 0.7cqh; margin-bottom:0.4cqh;">
          <thead>
            <tr style="font-size:0.68cqw;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.06em">
              <th style="text-align:left; padding:0 0.6cqw; width:15cqw; font-weight:700;">Category</th>
              <th></th>
              <th style="text-align:center; padding:0 0.6cqw; width:12cqh; font-weight:700;">Conforming</th>
              <th style="text-align:center; padding:0 0.6cqw; width:12cqh; font-weight:700;">Defective</th>
            </tr>
          </thead>
          <tbody>
            {category_rows}
          </tbody>
        </table>
        <div style="display:flex;justify-content:space-between;margin-top:0.8cqh;font-size:0.75cqw;color:var(--muted);padding-top:0.6cqh;border-top:1px solid #eae8e3">
          <span><b>Middle Column:</b> 100% normal training baseline</span>
          <span><b>Right Column:</b> Unseen defect challenge</span>
        </div>
      </div>
    </div>"""
    slide_html(
        "Industrial Component<br><span class='accent'>Anomaly Detection</span>", body, eyebrow="Project presentation"
    )
