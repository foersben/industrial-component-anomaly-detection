---
type: rules
---
# Workspace Rules

## Git Settings

* **Never bypass Git settings:** Do not use `--no-gpg-sign` or other flags to bypass user Git configurations (such as GPG commit signing). If a commit fails due to a missing GPG key/agent in the agent shell, report the issue to the user and ask them to perform the commit themselves rather than bypassing signature requirements.

## Fair & Scientifically Sound Model Evaluation Protocol

All agents and developers implementing or evaluating models MUST strictly adhere to the [Fair Evaluation Protocol](rules/01-fair-scientific-evaluation.md):

* **Deterministic Split & No Test Leakage:** Always use `build_fair_evaluation_split()` (`seed=42`). Never train/fit any model on 100% of normal training data — fitting is strictly bounded to the **85% fit partition**. Decision thresholds MUST be calibrated exclusively on the **15% normal validation partition**. Test data MUST remain unseen until final scoring.
* **Canonical 256×256 Resolution:** All continuous anomaly score maps and ground-truth binary masks MUST be rescaled to $256 \times 256$ before computing pixel-level localization metrics (bilinear for heatmaps, nearest-neighbor + binarization for ground-truth masks).
* **Strict AUPIMO & AUROC:** AUPIMO must use `anomalib.metrics.AUPIMO` with standard bounds $\text{FPR} \in [10^{-5}, 10^{-4}]$ and `num_thresholds=50_000`. Never use ad-hoc pooled-pixel substitutes or silent fallbacks (`0.0`). Missing data or calculation failures must raise explicit exceptions.
* **Full Confusion Matrix:** All pipelines must compute and persist image-level confusion counts (**TP, FP, FN, TN**), precision, recall, and anomalous-class F1 score alongside AUROC/AUPIMO.
* **Protocol Cache Versioning:** Disk caches must encode the `fair-eval-v1` protocol identity, split digests, canonical resolution, and threshold count. Legacy or test-leaked caches must be invalidated.
