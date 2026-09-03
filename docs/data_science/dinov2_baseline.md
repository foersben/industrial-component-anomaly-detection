---
type: Data Science
title: "DINOv2 Patch Nearest-Neighbour Baseline"
description: "A fair AnomalyDINO baseline using frozen DINOv2 patch tokens and a normal-only memory bank."
tags: [dinov2, anomaly-detection, localization, transfer-learning, aupimo]
---

# DINOv2 Patch Nearest-Neighbour Baseline

This baseline uses AnomalyDINO's pretrained, frozen DINOv2-small Vision Transformer. It stores patch-token features
from the shared normal fitting partition and scores every validation or test patch by cosine distance to its nearest
normal training patch. The mean of the highest-scoring 1% of patches is the image anomaly score; the spatial patch
scores are interpolated into a dense anomaly map.

No gradient training, anomalous samples, test labels, or test masks influence the feature bank or thresholds.

## Fair-comparison protocol

The implementation calls the same shared helpers as PatchCore and the Keras CAE:

- fixed 85/15 split of official normal training images, seed 42;
- fitting patch bank constructed from the 85% fitting paths only;
- image and pixel thresholds calibrated from the 15% normal validation paths only;
- unchanged official MVTec AD test paths used once for final reporting;
- image F1, recall, and precision as deployment metrics;
- image AUROC and average precision as threshold-independent supporting metrics;
- pixel maps and masks canonicalized to 256×256;
- full-map AUPIMO with 50,000 thresholds and FPR bounds from 1e-5 to 1e-4;
- no fallback localization metric when AUPIMO cannot be computed.

## Pre-registered configurations

The feature bank always uses the full fitting set. Coreset subsampling is disabled. The only AnomalyDINO ablation is
foreground masking:

| Mode | Meaning |
| --- | --- |
| `off` | Plain frozen DINOv2 patch-token nearest-neighbour baseline. |
| `on` | Enable AnomalyDINO's PCA-derived foreground masking for an explicit ablation. |
| `published` | Enable masking only for `capsule`, `hazelnut`, `pill`, `screw`, and `toothbrush`. |

The default is `published`, a policy fixed before this repository's test results are observed. Because validation is
normal-only, it can measure false-alarm behaviour and calibrate thresholds but cannot estimate anomalous recall or F1.
Do not choose among masking modes using official test F1, recall, AUPIMO, or any other test metric. For a clean
ablation, pre-register `off` and `on`, report both once, and do not promote the better test result as a tuned model.

## Run

Use the published full-shot masking policy:

```bash
pixi run python -m app.cli dinov2 --category capsule --masking published --heatmap
```

Run the plain baseline:

```bash
pixi run python -m app.cli dinov2 --category bottle --masking off
```

Run all 15 categories sequentially with the same pre-registered policy:

```bash
pixi run python -m app.cli dinov2 --category all --masking published
```

The all-category runner resumes safely: it reuses a category only when its metadata and both metric archives are
complete (and its heatmap archive is present when `--heatmap` was requested). Between categories it releases trainer
and accelerator state. Saved heatmaps remain available on disk but are omitted from the in-memory all-category
response to prevent their nested pixel lists from accumulating into several gigabytes. The response contains compact
per-category results and unweighted macro averages. These aggregates are final reporting outputs only; they are not
used to change masking, neighbours, thresholds, or any other setting.

The API exposes the same runner at `POST /api/pipelines/dinov2`. Artifacts are written under
`data/models/dinov2/<model-hash>/` as `metadata.json`, `image_metrics.npz`, `pixel_metrics.npz`, and optional compressed
heatmap overlays. Metadata includes the exact path digests, split parameters, masking policy, thresholds, metric
settings, and raw summary values needed to audit comparisons.

## Interpretation

The anomaly map is a patch-feature distance map, not an attention map. It localizes regions whose DINOv2 tokens are
unlike all stored normal tokens. Transformer attention can be visualized separately for learning and diagnosis, but it
must not replace the anomaly-distance map in quantitative localization evaluation.
