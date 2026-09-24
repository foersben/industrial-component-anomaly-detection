---
type: GUIDE
title: Post-Report PyTorch Autoencoders
description: Details on the later added PyTorch Unified Perceptual Autoencoder pipelines and evaluations.
tags: [pytorch, autoencoder, ms-ssim, perceptual_loss, post_report]
---

!!! warning "Not Part of the Main Project Report"
    The notebooks and models described in this document (`05_unified_perceptual_autoencoder.ipynb`, `05b_unified_perceptual_autoencoder_eval.ipynb`, and `mina_autoencoder_final_improved_and_ssim.ipynb`) were added during post-report exploratory phases. They are **not** part of the official project deliverables or the primary report. The main project model remains the Keras Convolutional Autoencoder (CAE).


## 1. Context & Motivation

Following the completion of the main project report and the Keras CAE implementation, further exploratory research was conducted using **PyTorch** to evaluate the impact of advanced structural loss functions and Masked Image Modeling (MIM) on fine-grained industrial components (such as screws).

*Subtle technical note regarding the earlier Keras CAE:* During the post-report review, it was noticed that the Keras model accidentally utilized a combination of SSIM+L2 loss, rather than a strictly optimized perceptual formulation. This PyTorch pipeline allowed us to explicitly test MS-SSIM combined with L1 and Perceptual Loss (VGG/ResNet features).

## 2. The Unified Perceptual Autoencoder

The notebook `05_unified_perceptual_autoencoder.ipynb` implements a `ConvAutoencoder` utilizing `ELU` activations and a narrow bottleneck (64 channels at 16x16 resolution) to act as a strong regularization mechanism.

### Key Objectives Investigated

- **MS-SSIM + L1 + Perceptual Loss:** Evaluated against standard MSE. MS-SSIM computes structural similarity across multiple scales, ignoring minor misalignments and lighting variations while forcing the model to reconstruct physical component structures accurately.
- **Masked Image Modeling (MIM) Ablation:** Explored dynamic patch-masking (MIM) to force the model to "inpaint" missing patches.

### Findings on Fine-Grained Defects (Screws)

The ablation study revealed that **MIM is detrimental to fine-grained defect detection**. For items like screws, if a thread is scratched or malformed, the MIM pipeline interprets the defect as "missing information" and perfectly hallucinates a normal screw thread. Because the reconstruction perfectly corrects the defect, the model yields a zero anomaly error over the defective region, severely reducing the Hit Error Rate (HER).

Therefore, the recommended baseline for these components avoids MIM and strictly utilizes Perceptual Loss with MS-SSIM for shape-aware robustness.

## 3. Strict Evaluation Enhancements

The evaluation script (`05b_unified_perceptual_autoencoder_eval.ipynb`) was refined to match the rigorous statistical boundaries established in the dataset benchmarks:

- **Image-Level Thresholding:** Thresholds for binary anomaly classification are now correctly calculated using the 99th percentile of *image-level max pixel errors* in the normal validation set, replacing the noisy all-pixel percentile.
- **AUPIMO Integration:** Formal integration of `anomalib.metrics.AUPIMO` with standardized False Positive Rate bounds (`[1e-5, 1e-4]`) and 50,000 threshold bins.
- **Hit Error Rate (HER):** Computation of the Top-1% Defect Capture rate to measure true spatial localization hit rates.

## 4. Mina's Improved Notebook

Also merged into this branch is `mina_autoencoder_final_improved_and_ssim.ipynb`, which contains corresponding exploratory iterations and Exploratory Data Analysis (EDA) implementations evaluating these improved metrics and architectures.
