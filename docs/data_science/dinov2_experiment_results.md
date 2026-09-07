# DINOv2 experiment results

This page records repository-only experimental results for the DINOv2 pipelines. It does not alter the shared project report. All numbers are macro means across the 15 MVTec AD categories under `fair-eval-v1` (seed 42, normal-only validation thresholds).

## Deployment-oriented comparison

Image F1 is the primary selection metric because the deployed system must make a binary decision for each inspected image. PR-AUC is threshold-independent context; AUPIMO measures low-false-positive pixel localization and should be treated as a secondary diagnostic metric.

| Model | Image F1 ↑ | Image PR-AUC ↑ | AUPIMO ↑ | Deployment reading |
|---|---:|---:|---:|---|
| DINOv2, single-layer 1-NN | **0.941** | **0.989** | 0.602 | Best image decision performance |
| PatchCore | 0.929 | 0.988 | **0.603** | Nearly tied localization; lower Image F1 |
| DINOv2, enhanced | 0.920 | 0.983 | 0.545 | Better dense maps, but worse deployed decision metric |
| Keras CAE | 0.495 | 0.867 | 0.058 | Substantially behind feature-matching methods |

The original DINOv2 baseline is therefore the preferred DINO configuration for deployment: it improves Image F1 by **1.2 percentage points** over PatchCore while effectively tying its ranking and localization metrics. The enhanced variant should not replace it as the default because its Image F1 falls by **2.1 points** versus the original DINOv2 pipeline.

The apparent contradiction—better patch maps but worse image decisions—is caused by aggregation and calibration. Multi-layer features, position-aware 5-NN, local-density normalization, and PatchCore-style aggregation raise mean pixel F1 from **0.290 to 0.385** and pixel AUROC from **0.969 to 0.979**, but they also change the score distribution. With the same normal-validation thresholding policy, that distribution produces more image-level classification errors in difficult categories.

## Enhanced variant by category

| Category | Image F1 | AUPIMO | Change from original DINOv2 AUPIMO |
|---|---:|---:|---:|
| bottle | 0.992 | 0.815 | -0.085 |
| cable | 0.900 | 0.292 | -0.204 |
| capsule | 0.925 | 0.372 | +0.249 |
| carpet | 0.983 | 0.973 | +0.013 |
| grid | 0.974 | 0.954 | -0.028 |
| hazelnut | 0.965 | 0.280 | -0.325 |
| leather | 0.995 | 0.996 | -0.004 |
| metal nut | 0.995 | 0.636 | +0.096 |
| pill | 0.907 | 0.491 | -0.070 |
| screw | 0.678 | 0.109 | -0.157 |
| tile | 0.982 | 0.828 | -0.136 |
| toothbrush | 0.652 | 0.022 | -0.117 |
| transistor | 0.892 | 0.326 | +0.213 |
| wood | 0.967 | 0.488 | -0.198 |
| zipper | 0.987 | 0.587 | -0.109 |

Only four categories improve on AUPIMO, while the largest Image F1 regressions occur for `screw` and `toothbrush`. This suggests that future work should tune aggregation and threshold calibration per category before adding more feature complexity.

## Configuration and provenance

The enhanced experiment uses DINOv2 ViT-S/14 layers 8, 10, and 11; five neighbours; a one-patch positional radius; local-density normalization; and PatchCore-style image aggregation. Both DINO variants use the same data split, seed, score space, and validation-derived threshold policy as the comparison baselines.

The machine-readable summary is in [`dinov2_comparison.csv`](dinov2_comparison.csv). Per-category metrics and configuration are retained in each artifact's `metadata.json` under `data/models/`. The implementation commits are `f996223`, `57fd337`, and `5eb448d`.

PR-AUC values use trapezoidal integration of each stored precision-recall curve, matching the shared report's Table 1 convention. Rounded display values can differ slightly from average precision stored in DINOv2 metadata.
