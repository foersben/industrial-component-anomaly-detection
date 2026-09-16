# Presentation assets

This directory contains the self-contained asset subset required by the defense presentation.

- `defect_examples/` contains presentation-sized copies of the MVTec AD evidence used on slides 1–3. Slide 3 uses copies with the matching dataset-mask boundary drawn directly over each uncropped image.
- `generated_charts/` contains the existing PatchCore threshold chart used on slide 8.
- `demo_results/patchcore/` contains the 15 category metric summaries and two existing four-panel PatchCore results per category for slide 9. Each four-panel result preserves the source image, dataset mask, anomaly overlay, and predicted mask in one aligned file.

The live demo is deliberately curated instead of copying the complete evaluation output. Its image files use fixed slide-specific names under each category's `four_panel/` directory. Single-image evidence is stored at 512×512 pixels. The wide chart and four-panel inspection results retain larger dimensions because they render substantially wider and contain small labels. The three slide 3 contour overlays were generated from matching MVTec AD image-mask pairs without cropping.

MVTec AD images remain subject to the dataset terms documented in the repository's `LICENSE-DATA.md`.
