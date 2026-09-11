---
type: Data Science
title: "DINOv3 Patch Nearest-Neighbour Baseline and Results"
description: "A fair frozen-DINOv3 baseline and its reproducible all-category result summary."
tags: [dinov3, anomaly-detection, localization, transfer-learning, aupimo]
---

# DINOv3 Patch Nearest-Neighbour Baseline and Results

This experiment applies the original DINOv2 baseline's evaluation design to a DINOv3 ViT-S/16 encoder. The normal-only memory bank, 1-NN patch scoring, fixed split, seed, validation-derived thresholds, and evaluation metrics remain unchanged. DINOv3 uses its native 256×256 input and a 16×16 patch size; masking differs as documented below.

Foreground masking is disabled for DINOv3. Anomalib's published masking implementation uses an absolute PCA threshold calibrated for DINOv2 features. With DINOv3 it can select no patches and produce an empty memory bank, so transferring that setting would not be a valid encoder-only comparison. No replacement threshold is tuned on the reused test set.

The pretrained weights use the DINOv3 license. Review and accept those terms before running or distributing model materials. The `timm` weight repository may also require Hugging Face authentication depending on its access policy.

## Run all categories

```bash
pixi run -e dev python -m app.cli dinov3 --category all
```

The encoder is frozen: the fitting phase constructs a normal patch memory bank but does not update model weights. Completed category artifacts are reused, so an interrupted all-category run can be restarted with the same command. Artifacts are stored under `data/models/dinov3/<model-hash>/`.

After all 15 categories finish, the runner writes:

- `data/models/dinov3/summary.json`, containing configuration, macro means, and per-category metrics;
- `data/models/dinov3/category_metrics.csv`, containing one compact row per category.

## Results

The pre-registered all-category run completed for all 15 categories. DINOv3 slightly weakens the primary image decision metrics but produces the strongest AUPIMO result in the repository comparison.

| Model | Image F1 ↑ | Image PR-AUC ↑ | Pixel F1 ↑ | Pixel AUROC ↑ | AUPIMO ↑ |
|---|---:|---:|---:|---:|---:|
| DINOv3, single-layer 1-NN | 0.932 | 0.986 | 0.196 | 0.958 | **0.680** |
| DINOv2, single-layer 1-NN | 0.941 | 0.989 | 0.290 | 0.969 | 0.602 |

DINOv2 remains preferred for deployment because Image F1 is the primary selection metric. DINOv3 improves AUPIMO by `0.078`, but its pixel F1 is lower in every category, so the gain is specific to stringent low-FPR region overlap rather than general segmentation quality. See the [combined DINO experiment analysis](dinov2_experiment_results.md) for category deltas, metric interpretation, and the masking comparability limitation.

Because the DINOv2 test results were already known when DINOv3 was added, this is a follow-up comparison on a reused benchmark test set rather than fresh final-test evidence. It is still useful as a structured follow-up comparison, but the DINOv3 result must not be used to revise the configuration and then re-report the same test set as unbiased.

The API exposes the same runner at `POST /api/pipelines/dinov3`. A single category can be run with, for example:

```bash
pixi run -e dev python -m app.cli dinov3 --category bottle --heatmap
```
