---
type: Guide
title: Data Science & Metrics
description: Introduction to the datasets, mathematical evaluation metrics, and benchmarking methodologies for industrial anomaly detection.
tags: [data-science, metrics, auroc, aupro, aupimo, mvtec-ad, index]
---

# Data Science & Modeling Concepts

Welcome to the data science and evaluation reference section for industrial anomaly detection. In visual anomaly detection (AD) systems, the core challenges lie in learning continuous normal manifolds from highly constrained clean datasets and identifying subtle, local variations without producing costly false alarms.

This section compiles the theoretical foundations, benchmark datasets, and advanced statistical evaluation frameworks used to validate model quality and robustness.

## Core Data Science Chapters

In this section, you will find:

1. **[MVTec AD Dataset](mvtec_ad.md):** A detailed review of the standard, high-resolution industrial anomaly detection benchmark, its categories, target complexity, baseline architectures, and experimental findings.
2. **[Anomaly Detection Metrics](anomaly_detection_metrics.md):** A deep mathematical dive into standard pixel-level classification (AUROC), per-region overlap metrics (AUPRO), and the state-of-the-art normal-validated per-image overlap metric (AUPIMO) and operational factory thresholds ($T_{AUPIMO}^{min}$).
3. **[Patchcore Baseline & Leakage](patchcore_baseline.md):** Documentation covering our ResNet18 Anomalib implementation of Patchcore, and how we solved the framework data leakage problem by enforcing strict normal-data thresholding.
4. **[Keras CAE Architecture](keras_cae_architecture.md):** Deep dive into the convolutional autoencoder architecture, AdamW optimizer, and combined SSIM+MSE loss.
5. **[Keras CAE Preprocessing](keras_cae_preprocessing.md):** Data augmentation, image loading, and Otsu+Canny foreground extraction (BGRP-G).
6. **[Keras CAE Inference & Evaluation](keras_cae_inference.md):** Top-K pooling, adaptive thresholds (Quantile/Mahalanobis), Precision-Recall curves, and strict industrial FPR ($10^{-5}$) validation.
7. **[Keras CAE Explainability](keras_cae_explainability.md):** Reconstruction error heatmaps, sliding window overlap stitching, robust quantile clamping, and side-by-side ground truth validation.
