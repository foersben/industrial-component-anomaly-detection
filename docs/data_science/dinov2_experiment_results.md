# DINO experiment results

This page records repository-only experimental results for the DINOv2 and DINOv3 pipelines. It does not alter the
shared project report. All numbers are unweighted macro means across the 15 MVTec AD categories under `fair-eval-v1`
(seed 42, normal-only validation thresholds).

## Deployment-oriented comparison

Image F1 is the primary selection metric because the deployed system must make a binary decision for each inspected image. PR-AUC is threshold-independent context; AUPIMO measures low-false-positive pixel localization and should be treated as a secondary diagnostic metric.

| Model | Image F1 ↑ | Image PR-AUC ↑ | AUPIMO ↑ | Deployment reading |
|---|---:|---:|---:|---|
| DINOv2, single-layer 1-NN | **0.941** | **0.989** | 0.602 | Best image decision performance |
| DINOv3, single-layer 1-NN | 0.932 | 0.986 | **0.680** | Best low-FPR localization; second-best Image F1 |
| PatchCore | 0.929 | 0.988 | 0.603 | Matches DINOv2 localization; lower Image F1 |
| DINOv2, enhanced | 0.920 | 0.983 | 0.545 | Better dense maps, but worse deployed decision metric |
| Keras CAE | 0.495 | 0.867 | 0.058 | Substantially behind feature-matching methods |

The original DINOv2 baseline remains the preferred deployment configuration. DINOv3 lowers Image F1 by **0.8
percentage points** and image PR-AUC by **0.3 points** relative to DINOv2. Its mean precision falls from `0.949` to
`0.933`, while recall rises slightly from `0.938` to `0.941`; the resulting operating-point trade-off is worse for the
primary image decision metric. DINOv3 still narrowly exceeds PatchCore Image F1 by `0.3` points, but its PR-AUC is
`0.2` points lower.

DINOv3's strength is specifically low-false-positive localization: AUPIMO improves by **7.8 points** over DINOv2 and
**7.7 points** over PatchCore. This is not a general pixel-level improvement. Compared with DINOv2, DINOv3 lowers
pixel AUROC by `1.2` points and pixel F1 by `9.4` points; its pixel F1 is lower in all 15 categories. AUPIMO measures
region overlap across a stringent low-FPR range, whereas pixel AUROC measures global ranking and pixel F1 uses one
normal-validation-derived threshold. DINOv3 can therefore improve the first while weakening the latter two.

The enhanced DINOv2 variant should also not replace the original default because its Image F1 falls by **2.1 points**
versus the original DINOv2 pipeline.

The apparent contradiction—better patch maps but worse image decisions—is caused by aggregation and calibration. Multi-layer features, position-aware 5-NN, local-density normalization, and PatchCore-style aggregation raise mean pixel F1 from **0.290 to 0.385** and pixel AUROC from **0.969 to 0.979**, but they also change the score distribution. With the same normal-validation thresholding policy, that distribution produces more image-level classification errors in difficult categories.

## DINOv3 by category

| Category | Image F1 | Change from DINOv2 Image F1 | AUPIMO | Change from DINOv2 AUPIMO |
|---|---:|---:|---:|---:|
| bottle | 0.992 | +0.000 | 0.889 | -0.011 |
| cable | 0.957 | +0.051 | 0.719 | +0.224 |
| capsule | 0.888 | +0.034 | 0.521 | +0.398 |
| carpet | 0.937 | -0.052 | 0.969 | +0.009 |
| grid | 0.926 | -0.032 | 0.839 | -0.143 |
| hazelnut | 0.979 | +0.000 | 0.823 | +0.218 |
| leather | 0.968 | +0.015 | 1.000 | +0.000 |
| metal nut | 0.984 | -0.011 | 0.891 | +0.352 |
| pill | 0.964 | +0.027 | 0.449 | -0.112 |
| screw | 0.732 | -0.117 | 0.012 | -0.253 |
| tile | 0.982 | -0.006 | 0.941 | -0.024 |
| toothbrush | 0.938 | -0.030 | 0.701 | +0.562 |
| transistor | 0.854 | +0.043 | 0.270 | +0.157 |
| wood | 0.916 | -0.044 | 0.580 | -0.107 |
| zipper | 0.970 | -0.004 | 0.588 | -0.107 |

DINOv3 improves Image F1 in five categories, ties in two, and regresses in eight. Its largest gains are on `cable`
and `transistor`; its largest regression is on `screw`. AUPIMO improves in eight categories, led by `toothbrush`,
`capsule`, and `metal_nut`, but the `screw` and `grid` regressions are substantial.

The masking configurations require care when attributing these changes. DINOv2 uses its published category masking
policy, while DINOv3 disables masking because Anomalib's absolute DINOv2 PCA threshold produces an empty DINOv3
memory bank. On the ten categories where DINOv2 masking also resolves to off, DINOv3 changes mean Image F1 by
`-0.004`, pixel F1 by `-0.084`, pixel AUROC by `-0.012`, and AUPIMO by `+0.035`. The direction of the overall finding
therefore remains—slightly weaker image decisions and markedly weaker thresholded pixel masks, with better AUPIMO—but
the full 15-category AUPIMO gain cannot be attributed to the encoder alone.

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

The DINOv3 experiment uses frozen `vit_small_patch16_dinov3.lvd1689m` tokens at 256×256, one neighbour, no coreset,
and masking off. The enhanced experiment uses DINOv2 ViT-S/14 layers 8, 10, and 11; five neighbours; a one-patch
positional radius; local-density normalization; and PatchCore-style image aggregation. All DINO variants use the same
data split, seed, score space, and validation-derived threshold policy as the comparison baselines.

The machine-readable comparison is in [`dinov2_comparison.csv`](dinov2_comparison.csv). The generated DINOv3 macro
and category summaries are in `data/models/dinov3/summary.json` and `data/models/dinov3/category_metrics.csv`.
Per-category metrics and configuration are retained in each artifact's `metadata.json` under `data/models/`.

PR-AUC values use trapezoidal integration of each stored precision-recall curve, matching the shared report's Table 1
convention. Rounded display values can differ slightly from average precision stored in DINO metadata.
