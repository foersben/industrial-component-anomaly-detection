---
type: Guide
title: Model Artifact Storage on Hugging Face
description: Archive and restore trained model artifacts with Hugging Face Hub.
tags: [models, huggingface, storage, guide]
---

# Model Artifact Storage on Hugging Face

Trained artifacts under `data/models/` are generated files and are intentionally excluded from Git. Archive them in a dedicated Hugging Face **model** repository instead of deleting them when local disk space is needed.

## Restore the public archive

From a fresh clone of the [application repository](https://github.com/foersben/industrial-component-anomaly-detection/), install the Pixi environment and restore the model registry:

```bash
pixi install -e dev
pixi run --frozen -e dev hf download \
  abulhawa/industrial-component-anomaly-detection-models \
  --repo-type model --local-dir data/models
pixi run --frozen -e dev ui
```

The archive is public, so downloading does not require a Hugging Face login. Its `four_panel_images/` directories and metadata retain paths relative to each model cache. PatchCore and Keras CAE **Load Saved Results** use the archived snapshots without loading model weights or requiring the raw dataset. The DINO archives contain evaluation outputs, but the DINO pipelines are exposed through the CLI and API rather than the Streamlit tabs.

To train or re-evaluate, download [MVTec AD](dataset_setup.md) separately with `pixi run --frozen -e dev just download-data`. The dataset is needed for protocol validation and inference. The archive does not contain pretrained DINO encoders or fitted PatchCore memory banks.

The historical evaluation fingerprints include the original machine's dataset paths. **Load Saved Results** works after a restore, but re-evaluating one of those historical caches on a different filesystem may be rejected. Run a new evaluation when you need fresh metrics on that machine.

An equivalent restore recipe is `pixi run --frozen -e dev just download-models`. Existing local files at matching paths may be replaced, so use an empty `data/models/` directory when comparing archives.

## Upload updates

Uploading requires a Hugging Face account with write access to the model repository. Authenticate first:

```bash
pixi run --frozen -e dev just hf-login
```

Upload the complete active model registry:

```bash
pixi run --frozen -e dev just upload-models abulhawa/industrial-component-anomaly-detection-models
```

The upload preserves paths below `data/models/` and leaves every local file in place. Soft-deleted models in `.trash/` and legacy migration backups are excluded. To upload a different artifact directory, pass it as the second argument:

```bash
pixi run --frozen -e dev just upload-models abulhawa/industrial-component-anomaly-detection-models models
```

Only remove local artifacts after the command succeeds and the files are visible in the Hub repository. DINOv3 and other third-party-derived artifacts may carry upstream license or access restrictions; review those terms before making the repository public.
