---
type: Guide
title: Model Artifact Storage on Hugging Face
description: Archive and restore trained model artifacts with Hugging Face Hub.
tags: [models, huggingface, storage, guide]
---

# Model Artifact Storage on Hugging Face

Trained artifacts under `data/models/` are generated files and are intentionally excluded from Git. Archive them in a dedicated Hugging Face **model** repository instead of deleting them when local disk space is needed.

## One-time setup

1. Create a model repository on Hugging Face, for example `foersben/industrial-component-anomaly-detection-models`.
2. Prefer a **private** repository unless every artifact and its upstream weights may legally be redistributed.
3. Authenticate with a write token:

   ```bash
   just hf-login
   ```

## Upload

Upload the complete active model registry:

```bash
just upload-models foersben/industrial-component-anomaly-detection-models
```

The upload preserves paths below `data/models/` and leaves every local file in place. Soft-deleted models in `.trash/` and legacy migration backups are excluded. To upload a different artifact directory, pass it as the second argument:

```bash
just upload-models foersben/industrial-component-anomaly-detection-models models
```

Only remove local artifacts after the command succeeds and the files are visible in the Hub repository. DINOv3 and other third-party-derived artifacts may carry upstream license or access restrictions; review those terms before making the repository public.

## Restore

Restore the registry into its standard location:

```bash
just download-models foersben/industrial-component-anomaly-detection-models
```

An alternate destination can be passed as the second argument. Existing local files with the same paths may be replaced by the downloaded snapshot, so use an empty destination when comparing archives.
