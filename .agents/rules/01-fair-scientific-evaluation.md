---
type: rules
title: Fair and Scientifically Sound Model Evaluation Protocol
description: Mandatory scientific invariants, leakage-free data partitioning, canonical resolutions, and metric standards for anomaly detection models.
tags: [agentic, rules, evaluation, data-science, reproducibility, aupimo, fair-eval-v1]
---

# Fair & Scientifically Sound Model Evaluation Protocol

This rule governs all machine learning pipelines, anomaly scoring mechanisms, evaluation scripts, and testing procedures across the repository. All agents and developers MUST adhere strictly to these invariants when implementing, refactoring, or running model training and evaluation.

---

## 1. Deterministic Data Partitioning & Zero Test Leakage

1. **Unified Split Source:**
   All model evaluations MUST obtain data splits via `build_fair_evaluation_split()` in `app.domain.data`. Ad-hoc or per-model splitting routines (`train_test_split` calls inside individual pipeline files) are **strictly forbidden**.
2. **Fitting Partition Restriction:**
   No model (including PatchCore coreset memory banks, Keras CAE weights, and DINO feature stores) may ever train on 100% of the training images. Fitting MUST be restricted strictly to the **85% fitting partition** (`train_fit`).
3. **Threshold Calibration on Validation Only:**
   Decision thresholds (e.g. 99th percentile of normal reconstruction/distance error) MUST be derived **exclusively from the 15% normal validation partition** (`val_normal`).
4. **Zero Test Leakage:**
   Under NO circumstances may test set images (`test`) or their scores be used to calibrate, adjust, or influence model parameters or decision thresholds. The test partition MUST remain completely unseen until final evaluation.
5. **Fixed Random Seed:**
   All dataset partitioning and stochastic sampling (including PatchCore coreset subsampling and DataLoader worker initialization) MUST be deterministic with `seed = 42`.

---

## 2. Canonical Resolution for Pixel Localization Metrics

1. **Fixed 256×256 Standard:**
   Prior to computing pixel-level localization metrics (AUROC and AUPIMO), all continuous anomaly heatmaps and binary ground-truth defect masks MUST be resized to canonical **256×256** spatial resolution.
2. **Interpolation Standards:**
   - **Continuous anomaly score maps:** Resized using **bilinear interpolation** (`cv2.INTER_LINEAR` or PIL `BILINEAR`).
   - **Ground-truth binary masks:** Resized using **nearest-neighbor interpolation** (`cv2.INTER_NEAREST` or PIL `NEAREST`), followed by explicit re-binarization (`mask > 0.5` or `mask > 127`).

---

## 3. Metric Standardization & Prohibition of Fallbacks

1. **Full-Map AUPIMO Benchmark:**
   Pixel-level AUPIMO MUST be computed using `anomalib.metrics.AUPIMO` with official MVTec AD benchmark parameters:
   - False Positive Rate bounds: $\text{FPR} \in [10^{-5}, 10^{-4}]$ (`fpr_bounds=(1e-5, 1e-4)`).
   - Threshold integration resolution: `num_thresholds = 50_000`.
2. **Prohibition of Metric Fallbacks:**
   - Never replace AUPIMO with ad-hoc pooled-pixel recall substitutes, relaxed FPR bounds, or invented approximations.
   - Never return silent default values (e.g., `return 0.0`) on missing inputs or mathematical singularities.
   - Incomplete data, missing masks, or failed metric calculations MUST raise explicit exceptions (`ValueError` or `RuntimeError`).
3. **Pixel AUROC Standard:**
   Pixel-level AUROC MUST be computed using `sklearn.metrics.roc_auc_score` over flattened canonical 256×256 continuous maps and binary masks.
4. **Complete Confusion Matrix Reporting:**
   In addition to threshold-independent ranking metrics (AUROC, PR-AUC), every evaluation pipeline MUST compute and report image-level binary classification counts at the calibrated validation threshold:
   - **True Positives (TP)**, **False Positives (FP)**, **False Negatives (FN)**, **True Negatives (TN)**.
   - **Precision**, **Recall**, and **F1 Score** (anomalous class).

---

## 4. Cache Versioning & Invalidation

1. **Protocol Identity in Cache Keys:**
   All persistent model and metric caches (e.g. `data/cache/`, `.npz` files, serialized registry checkpoints) MUST encode the protocol identity (`fair-eval-v1`), split seed (`42`), path digests, canonical resolution, and threshold count in their cache key.
2. **Rejection of Stale/Legacy Caches:**
   Any cached artifact lacking protocol metadata or generated under legacy, un-split (100% fit) or test-leaked configurations MUST be rejected and invalidated.

---

## 5. Execution & Memory Bounding

1. **GPU Allocation:**
   TensorFlow/Keras execution MUST set `TF_GPU_ALLOCATOR=cuda_malloc_async` and enable dynamic GPU memory growth to avoid out-of-memory crashes on CUDA devices.
2. **Batch Bounding:**
   Crop extraction and patch scoring MUST operate in bounded batch chunks to maintain predictable memory ceilings during sweep and evaluation runs.
