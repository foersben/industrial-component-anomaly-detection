"""Title slide."""

# ruff: noqa: E501

from app.ui.presentation.theme import DATASET, img, slide_html


def render() -> None:
    """Render the defense title and project objective."""
    visual = DATASET / "capsule" / "test" / "scratch" / "022.png"
    body = f"""
    <div class="slide-grid-2" style="grid-template-columns:1.06fr .94fr;align-items:center">
      <div>
        <div style="font-size:2.05cqw;line-height:1.25;max-width:39cqw">
          Detect unseen manufacturing defects from normal examples --?? and show operators where the evidence lies.
        </div>
        <div class="def-rule"></div>
        <div class="tiny-label">Project defense</div>
        <div style="margin-top:1.1cqh;font-size:1.15cqw;line-height:1.5;color:#aec2c7">
          Benjamin Förster · Ali Abul Hawa<br>Mahboubeh Nadaf · Karim Khalifa
        </div>
      </div>
      <div class="image-frame" style="height:54cqh;transform:rotate(1.4deg)">
        {img(visual, "Capsule with a subtle scratch defect", "photo")}
      </div>
    </div>
    """
    slide_html("Industrial Component<br><span class='accent'>Anomaly Detection</span>", body, eyebrow="Project defense", extra_class="dark")
