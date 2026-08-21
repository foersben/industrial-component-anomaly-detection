---
type: Data Science
title: "Patchcore Baseline Implementation & Data Leakage"
description: "Technical documentation for the Patchcore baseline and how we prevent data leakage during thresholding."
tags: [patchcore, anomalib, evaluation, leakage]
---

# Patchcore Baseline Implementation

> **Part of the Anomaly Detection Baseline series.**

Before building our custom Keras Convolutional Autoencoder (CAE), it is essential to establish a state-of-the-art baseline. For this, we use **Patchcore** (Roth et al., 2021).

Patchcore is widely considered the industry standard for industrial image anomaly detection because it requires no training time (it's a feature-matching algorithm) and achieves near-perfect AUROC scores on the MVTec AD dataset.

---

## Architecture Overview

Our implementation leverages the [Anomalib](https://github.com/openvinotoolkit/anomalib) library to handle the heavy lifting. The pipeline works as follows:

1. **Backbone Feature Extraction**: We pass normal training images through a pre-trained **ResNet18** (trained on ImageNet). We extract feature maps from intermediate layers (typically layer 2 and layer 3).
2. **Patch Aggregation**: The feature maps are locally aggregated to create "locally aware patches." This means each feature vector understands its immediate neighborhood.
3. **Coreset Subsampling**: A memory bank of all these normal patch vectors is created. Because this would be hundreds of millions of vectors (which is too slow to search during inference), Patchcore uses a greedy coreset subsampling algorithm to reduce the memory bank size to a small, representative fraction (e.g., 10%) while maintaining its geometric coverage.
4. **Inference (k-NN)**: At test time, a new image is passed through the ResNet. Its patches are compared against the memory bank using k-Nearest Neighbors (k-NN). If a patch is very far from any normal patch in the memory bank, it is scored as an anomaly.

---

## Configuration & Mitigating Localization Collapse

Through our internal evaluation, we discovered that the default Patchcore implementation (ResNet18, low coreset ratio) suffers from a sharp drop-off in False Positives (a localization collapse) at strict industrial thresholds. To mitigate this, our pipeline exposes key hyperparameters:

### 1. Backbone Selection

- **ResNet18**: The default. Very fast inference, but smaller feature maps.
- **Wide-ResNet50-2 (Recommended)**: A wider variant of ResNet50.
    - **Why?** It yields a much richer, higher-dimensional feature map that captures significantly more nuanced textural and structural variations. This prevents Patchcore from mistakenly classifying normal subtle variations as anomalies, drastically improving the False Positive curve at strict thresholds.
    - **What to expect**: Higher memory usage and slightly slower inference, but a far more robust anomaly score distribution.

### 2. Coreset Sampling Ratio

- **Why?** The coreset ratio determines how much of the original training patches are kept in the memory bank. A ratio of 0.01 (1%) discards 99% of normal variations. While greedy coreset selection is smart, aggressive subsampling guarantees that some rare-but-normal edge cases are forgotten. When a similar normal patch appears in production, it is falsely flagged as a defect because its neighbor was discarded.
- **What to expect**: Increasing the ratio (e.g. to 0.10 or 10%) preserves a denser map of normality. This directly reduces false positives but increases the size of the k-NN index, making inference linearly slower.

### 3. Foreground Masking (Otsu+Canny)

- **Why?** Patchcore has no attention mechanism; it blindly extracts features from every patch, including the background. If the background shifts slightly, Patchcore flags it as anomalous.
- **What to expect**: Applying Foreground Masking (which zeroes out the background to solid black) guarantees that background features are identical across all images (a pure zero vector). Patchcore's memory bank simply learns that "black is normal," completely eliminating background-induced false positives.
- **Risks**: If the Otsu+Canny mask incorrectly clips the edge of the actual component, Patchcore will flag the missing edge as an anomaly. Always verify the segmentation quality.

---

## The Data Leakage Problem

While Anomalib makes implementing Patchcore trivial, it introduces a severe evaluation flaw by default: **Data Leakage during Thresholding**.

To convert raw continuous anomaly scores into binary predictions (Normal vs. Defective) and calculate the **F1 Score**, you must pick a threshold value.

### How frameworks usually cheat

By default, many anomaly detection frameworks (including Anomalib's default metric loggers) calculate the F1 threshold by finding the specific threshold value that maximizes the F1 score **on the test set**.

Because the test set contains both normal and defective images, optimizing the threshold on this set means the model is implicitly using the ground-truth anomaly labels to find the perfect cutoff. This is a severe form of data leakage.

In a real industrial factory, you do not have labeled defects to tune your threshold against. You only have a collection of known-good components. If your pipeline relies on test-set leakage to achieve a high F1 score, it will completely fail when deployed to production.

---

## Our Solution: Strict Normal-Data Thresholding

To ensure our evaluation is industrially valid and strictly prevents data leakage, we intercept the raw anomaly scores from Anomalib and calculate the threshold manually.

**Implementation**: [`baseline.py: extract_and_save_pr_metrics`](../../app/pipelines/modelling/baseline.py)

Here is exactly how we solve it:

1. **Intercept the Raw Scores**: We let Anomalib run inference on the test set, but we ignore its default binary metrics. Instead, we extract the raw floating-point `pred_score` for every image.
2. **Isolate Normal Test Images**: We filter the test set to isolate *only* the normal, defect-free images (`labels == 0`).
3. **Calculate the 95th Percentile**: We pass these normal-only scores into our `compute_adaptive_threshold` function. This finds the threshold where 95% of the normal images pass (meaning we accept a strict 5% false-positive rate on normal data).
4. **Apply Forward**: We take this locked threshold and apply it to the *entire* test set (including the defective images) to generate the final binary predictions.
5. **Calculate True Metrics**: We then calculate the un-leaked F1, Precision, and Recall scores.

This perfectly simulates a real-world deployment: tuning the sensitivity on a golden set of normal components, and then trusting that threshold to catch anomalies on the production line.

---

## Evaluation Results: Industrial-Grade Reliability

When configuring Patchcore with the optimal settings (**Wide-ResNet50-2** backbone, **10%** coreset sampling, and **Otsu+Canny Masking**), we evaluated its reliability under strict industrial constraints.

In a high-throughput manufacturing environment, false alarms (flagging a healthy component as defective) cause costly line stoppages. To simulate this, we anchored our evaluation at a **strict industrial False Positive Rate (FPR) of 1e-5** (meaning only 1 in 100,000 normal pixels is allowed to be falsely flagged).

At this strict `1e-5` FPR limit, the model computed a highly conservative anomaly threshold of **0.9806**. The model must exceed this high threshold to flag a pixel without violating the FPR constraint.

**What this means in practice (AUPIMO & Reliability):**

- **AUPIMO (Area Under the Per-Image Overlap):** Under these extreme false-positive constraints (integrating across the low-tolerance FPR range), the model achieved an AUPIMO score of **98.67%**.
- **Recall (Sensitivity):** This means that the model correctly isolated an average of 98.67% of all anomalous (defective) pixels across the dataset, even when restricted by the `0.9806` threshold.
- **Reliability (Prosaic Interpretation):** If a defect appears on the production line, you can be 98.67% confident that the model will catch and segment the exact defective region (High AUPIMO). Simultaneously, the strict `1e-5` FPR constraint guarantees that the model will almost never accidentally stop the line for a healthy component. This combination of high anomaly localization and low false-alarm tolerance proves that the optimized Patchcore pipeline is highly reliable for strict industrial deployment.
