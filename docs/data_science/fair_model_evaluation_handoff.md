---
type: Implementation Handoff
title: Fair PatchCore and CAE Evaluation Protocol
description: Current gaps, required code changes, tests, and completion criteria for a fair model comparison.
tags: [data-science, evaluation, patchcore, keras-cae, aupimo, reproducibility]
---

# Fair PatchCore and CAE Evaluation: Implementation Handoff

## Purpose and scope

This document is the implementation handoff for making the PatchCore and Keras CAE baseline comparison scientifically fair. It records what is wrong in the current code, what must change, where to make each change, and how to prove the correction is complete.

The scope is the baseline comparison only. Optuna and hyperparameter-sweep design are explicitly out of scope.

Do not use the historical report table to select a final model. Those values were produced with unequal fitting data, different image-threshold calibration paths, and previously inconsistent AUPIMO paths. Keep the historical artifacts for traceability, but write corrected runs to protocol-versioned caches.

## Continuation instructions for another chat

As of 2026-09-01, work is on branch `fix/consistent-model-evaluation`. Inspect the repository status before editing, preserve any user changes, and do not commit new work without explicit user permission. The shared fair-evaluation implementation is complete in the working tree but has not been committed. A deterministic PatchCore bottle verification has been completed; the Keras CAE rerun remains outstanding.

A continuation chat should:

1. read this document completely;
2. inspect `git status` and the existing diffs before editing;
3. keep Optuna out of scope;
4. preserve the completed implementation and rerun the automated checks if it changes;
5. diagnose the outstanding Keras CAE runtime failure before rerunning it;
6. inspect the corrected bottle metadata against the acceptance checklist before scaling out;
7. avoid changing final report numbers until corrected artifacts pass the acceptance checklist.

## Implementation status

The shared protocol is now implemented for both baseline pipelines:

- one deterministic split utility supplies identical ordered fitting, validation, and test partitions with path digests;
- PatchCore fitting is restricted to the 85% fitting partition and its threshold is calibrated only on the 15% normal validation partition;
- CAE consumes the same shared split;
- both models use shared 256x256 map/mask canonicalisation, scikit-learn pixel AUROC, and full-map AUPIMO;
- both return and persist image confusion counts;
- cache identity and metadata include `fair-eval-v1`, split and metric evidence, and legacy PatchCore caches are rejected;
- PatchCore explicitly seeds coreset sampling and loader workers with seed 42, disables Anomalib post-processing, and evaluates one documented raw-score prediction path;
- `metrics_summary.csv` persists the confusion counts;
- two independent PatchCore bottle fits produced identical results: image AUROC 1.0, threshold 7.370965, TP 63, FP 5, FN 0, TN 15, and anomalous-class F1 0.961832;
- a read-only real-data smoke check verified bottle counts of 177 fitting, 32 validation, and 83 test images in both the shared split and Anomalib datasets/loaders.

The next action is to complete the automated checks after the deterministic PatchCore change, then diagnose and rerun the bottle Keras CAE. Do not scale to other categories or update report values until both corrected bottle artifacts pass the acceptance checklist.

## Required protocol

For every category, both models must use the following fixed protocol:

1. Split the official normal training partition once into 85% fitting and 15% validation, using seed 42 and the same exact ordered image lists for both models.
2. Fit the CAE weights and construct the PatchCore memory bank/coreset using only the 85% fitting images.
3. Score the same 15% normal validation images with each fitted model and derive that model's image-classification threshold from those scores only.
4. Freeze both models and thresholds before opening the official test set.
5. Evaluate both models on the same exact ordered test-image list. Bottle must contain all 83 official test images.
6. Convert both anomaly maps and masks to 256x256 before shared pixel metrics. Use bilinear interpolation for continuous maps and nearest-neighbour interpolation for masks; binarise masks after resizing.
7. Compute pixel AUROC through one shared implementation.
8. Compute AUPIMO through one shared full-map `anomalib.metrics.AUPIMO` implementation with bounds `(1e-5, 1e-4)` and `num_thresholds=50_000`.
9. Do not use flattened pooled-pixel substitutes, alternate FPR bounds, invented metrics, or numeric fallback values. Missing inputs or failed metrics must raise a visible error.
10. Report TP, FP, FN, precision, recall, and F1 at image level, in addition to ranking and localisation metrics.

AUPIMO's shared FPR axis is calculated from normal images in the final test/evaluation set. This is part of the metric definition and is distinct from the deployed image-classification threshold, which must come from validation images.

## Pre-implementation audit (retained for traceability)

The items in this section describe the state before the continuation implementation above. They are retained to explain why the changes were required and must not be treated as the current code status.

### Already correct or substantially corrected

- `app/pipelines/modelling/keras_cae/cae_pipeline.py` splits normal development images with `test_size=0.15` and `random_state=42`.
- The CAE derives its image threshold from `val_good_images`, and its documentation now identifies these as validation images.
- Both model paths now call the full-map `compute_aupimo()` implementation in `app/pipelines/evaluation/cae_metrics.py`.
- `compute_aupimo()` currently constructs `anomalib.metrics.AUPIMO` with `num_thresholds=50_000`, increased from 10,000 to improve integration resolution inside the narrow FPR interval without the memory and runtime cost of a much denser grid.
- The corrected PatchCore bottle run produced AUPIMO `0.9825` at bounds `(1e-5, 1e-4)`.

### Still incorrect or incomplete

- `app/pipelines/modelling/baseline.py::run_baseline()` gives Anomalib's PatchCore datamodule the complete official normal training partition. PatchCore therefore fits on 100%, not the shared 85%.
- `app/pipelines/modelling/baseline.py::extract_and_save_pr_metrics()` predicts only the test dataloader and derives `img_threshold` from normal test scores. This is test leakage.
- The CAE and PatchCore create their data views independently. There is no shared split object proving they used the same paths and ordering.
- Pixel maps are not passed through one explicit shared 256x256 canonicalisation function before both AUROC and AUPIMO.
- PatchCore pixel AUROC comes from Anomalib `Engine.test()`, while CAE pixel AUROC is calculated separately with scikit-learn. The comparison does not yet use one shared code path.
- Image-level TP, FP, and FN are not consistently returned, persisted, and written to `metrics_summary.csv`.
- PatchCore cached metric loading currently catches corrupt/missing NPZ errors and silently uses `0.0` in some paths. A missing metric must not masquerade as a genuine zero.
- The PatchCore cache hash does not include the evaluation protocol version, split seed or split identity, canonical resolution, or AUPIMO threshold count. An old unfair cache can therefore collide with a corrected run.
- Metadata records split counts but not enough information to verify exact cross-model path equality.

## Required implementation changes

### 1. Create one shared split definition

Add a shared utility in the data or evaluation layer, rather than duplicating `train_test_split` inside both pipelines. It should:

- filter one category's official normal training rows;
- sort by a stable field such as the resolved image path before splitting;
- split with validation fraction `0.15` and seed `42`;
- return fitting paths, validation paths, and the unchanged ordered official test rows;
- reject overlap between partitions;
- return or persist a deterministic digest of each ordered path list.

Refactor `app/pipelines/modelling/keras_cae/cae_pipeline.py` to consume this shared result instead of creating its own split. Refactor `app/pipelines/modelling/baseline.py` to consume the same result.

For bottle, the expected counts are:

```text
official normal development images: 209
85% fitting images:                 177
15% validation images:              32
official test images:                83
```

### 2. Restrict PatchCore fitting to the 85% subset

In `app/pipelines/modelling/baseline.py::run_baseline()`, replace the implicit full-training datamodule behavior with explicit datasets/loaders or a datamodule whose `train_data` contains only the shared fitting paths.

Requirements:

- `engine.fit()` must see only the 85% fitting subset;
- the 15% validation images must never enter `embedding_store`, coreset selection, or `memory_bank`;
- preprocessing must be applied identically to fitting, validation, and test datasets without changing their membership;
- metadata must record `train_normal`, `val_normal`, and `test_total`, plus the ordered-list digests.

For the bottle ResNet-18 baseline at 256x256 with layers 2 and 3, 177 fitting images produce 181,248 candidate patch embeddings before a 10% coreset. This is a useful diagnostic, not a replacement for verifying the actual fitting path list.

### 3. Calibrate the PatchCore image threshold on validation

Refactor `extract_and_save_pr_metrics()` so threshold calibration and final evaluation receive separate predictions:

- obtain PatchCore image scores for the shared normal validation loader;
- calculate the configured normal quantile threshold from those validation scores;
- run final predictions on the official test loader;
- apply the already-frozen threshold to test image scores;
- never filter normal test scores to choose the threshold.

Rename arguments and local variables so `validation_scores` and `test_scores` cannot be confused. Remove the current `normal_image_scores = image_scores_np[image_labels_np == 0]` threshold path from test predictions.

The CAE already performs validation-based calibration. Preserve that behavior while moving it to the shared split definition.

### 4. Add shared canonical pixel preparation

Create one shared evaluation helper used by both pipelines. Given full anomaly maps and optional masks, it must:

- preserve one 2D map per image;
- validate equal image counts and valid two-dimensional shapes;
- resize anomaly maps to 256x256 with bilinear interpolation;
- resize masks to 256x256 with nearest-neighbour interpolation;
- convert maps to `float32` and masks to binary `uint8` or boolean arrays;
- generate all-zero masks for normal images only;
- return canonical arrays without flattening them.

Flatten copies only after canonicalisation when pixel AUROC or PR curves require one-dimensional inputs. AUPIMO must receive the complete 2D arrays.

### 5. Add one shared final metric function

Place the shared implementation in the evaluation layer and call it from both PatchCore and CAE. It should return at least:

```text
pixel_auroc
pixel_aupimo
aupimo_fpr_lower = 1e-5
aupimo_fpr_upper = 1e-4
aupimo_num_thresholds = 50000
canonical_height = 256
canonical_width = 256
```

Requirements:

- use the same `sklearn.metrics.roc_auc_score` call for both pixel-AUROC values;
- use the existing full-map Anomalib AUPIMO path for both models;
- treat NaN, missing classes, shape mismatches, missing masks, and metric failures as errors;
- do not return `0.0` merely because loading or computation failed;
- keep compatibility handling for different Anomalib return object shapes only if every branch extracts the genuine AUPIMO result.

### 6. Return and persist the confusion counts

After applying each validation-derived threshold to final test image scores, calculate and return:

```text
true_positives
false_positives
false_negatives
true_negatives
precision
recall
f1_score
```

Add these values to both pipelines' `image_level` result dictionaries, model metadata, and `results/evaluation/.../metrics_summary.csv`. The test-image labels may be used here because this is final evaluation, not calibration.

### 7. Version the cache and evidence

Introduce a protocol identifier such as `fair-eval-v1`. Include at least the following in model/run hashes and metadata:

- protocol identifier;
- split seed (`42`) and validation fraction (`0.15`);
- ordered fitting, validation, and test path digests;
- canonical map size (`256x256`);
- AUPIMO bounds and `num_thresholds`;
- threshold method and explicit source (`normal_validation`);
- shared pixel-metric implementation version.

Do not load an older cache as a fair-evaluation result when these fields are absent or different. Preserve old directories as historical artifacts unless the user explicitly asks to remove them.

## Tests required before rerunning models

Add or update tests to prove all of the following:

1. The shared split is deterministic at seed 42, disjoint, exhaustive over official normal training rows, and identical for both pipelines.
2. Bottle produces exactly 177 fitting, 32 validation, and 83 test paths.
3. PatchCore `engine.fit()` receives only fitting paths; no validation or test path reaches its memory bank.
4. Changing test scores while keeping validation scores fixed cannot change the calibrated image threshold.
5. Changing validation scores changes the threshold but does not change split membership.
6. Both models' canonical maps and masks reach shared pixel metrics with shape `(N, 256, 256)` and the expected dtypes.
7. Both models call the same pixel AUROC and full-map AUPIMO function with bounds `(1e-5, 1e-4)` and `num_thresholds=50_000`.
8. Metric errors propagate; no test accepts a fabricated `0.0` fallback.
9. TP, FP, FN, precision, recall, and F1 match a small hand-calculated example.
10. A protocol mismatch prevents an old cache hit.
11. Saved metadata contains counts, path digests, threshold provenance, canonical resolution, and metric configuration.
12. `metrics_summary.csv` contains the required confusion counts and metrics for both models.

## Correct implementation order

1. Add and test the shared split utility.
2. Refactor CAE to consume the shared split without changing its current behavior.
3. Restrict PatchCore fitting and add its validation loader.
4. Separate PatchCore validation calibration from test prediction.
5. Add shared 256x256 map/mask canonicalisation.
6. Route both pixel AUROC and AUPIMO through the shared metric function.
7. Add confusion counts and artifact metadata.
8. Version cache identity and reject legacy caches for fair evaluation.
9. Run unit and integration tests.
10. Rerun bottle baseline models first and inspect their split evidence before evaluating other categories.
11. Only after verification, rerun the remaining categories and replace provisional report values.

## Acceptance checklist

The fair baseline comparison is complete only when all boxes are satisfied:

- [ ] Both metadata files show protocol `fair-eval-v1`.
- [ ] Both metadata files contain identical split seed, fraction, and ordered path digests.
- [ ] Bottle counts are 177 fitting, 32 validation, and 83 test.
- [ ] PatchCore coreset contains no validation or test embeddings.
- [ ] Both image thresholds identify their source as `normal_validation`.
- [ ] Both test result files refer to the same ordered test-path digest.
- [ ] Both pixel evaluators receive `(N, 256, 256)` maps and masks.
- [ ] Both AUPIMO runs record bounds `(1e-5, 1e-4)` and 50,000 thresholds.
- [ ] No metric path contains a numeric fallback.
- [ ] Both summaries contain TP, FP, FN, precision, recall, and F1.
- [ ] Historical caches and tables are not presented as corrected results.
- [ ] Corrected bottle artifacts and metrics are reviewed before scaling to other categories.

## Files expected to change

The exact design may consolidate helpers differently, but another implementation chat should expect to inspect or modify:

```text
app/domain/data.py
app/pipelines/modelling/baseline.py
app/pipelines/modelling/keras_cae/cae_pipeline.py
app/pipelines/evaluation/cae_metrics.py
app/pipelines/evaluation/metrics.py
scripts/evaluate.py
tests/unit/test_patchcore_persistence.py
tests/unit/test_cae_data_invariants.py
tests/unit/test_cae_evaluation_and_explainability.py
docs/report_latex/report.tex
```

Prefer adding focused shared split and metric modules if doing so avoids circular dependencies or model-specific conditionals in generic evaluation code.

## Rerun and report policy

Start with bottle baselines. Do not update the report's quantitative table merely because a run completes. First inspect the metadata evidence and confirm the acceptance checklist. Then rerun both baselines using the ordinary evaluation entry point:

```bash
pixi run -e dev python scripts/evaluate.py --model patchcore --category bottle
pixi run -e dev python scripts/evaluate.py --model keras --category bottle
```

Corrected artifacts must use new protocol-aware hashes so the historical PatchCore cache `f0f02342d045` is not mistaken for a fair-evaluation run. Do not commit or delete historical artifacts unless explicitly requested.
