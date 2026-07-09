---
type: Guide
title: Dataset Setup & Hugging Face Integration
description: Instructions on how to download, host, and retrieve the MVTec AD dataset for the project.
tags: [dataset, huggingface, setup, guide]
---

# Dataset Setup & Hugging Face Integration

This guide documents how to acquire, host, and retrieve the **MVTec AD (Anomaly Detection)** dataset for the project.

## Dataset Selection & Scope

For a production-grade anomaly detection pipeline, we utilize the MVTec AD dataset (~5 GB total). It contains over 5,000 high-resolution images divided into fifteen different object and texture categories. Each category comprises defect-free training images and a test set with various defects and pixel-precise ground truth anomalies.

---

## Uploading the Data to Hugging Face Hub (One-Time Setup)

To allow team members and CI/CD pipelines to download the dataset on demand without cloning large binary files in Git, we host the dataset on the Hugging Face Hub.

### Step 1: Create a Hugging Face Dataset Repository

1. Register or log in at [huggingface.co](https://huggingface.co/).
2. Create a new dataset repository (e.g. `foersben/mvtec-ad`).
3. Set the visibility to **Public** (CC BY-NC-SA 4.0 allows public sharing for non-commercial research).
4. **IMPORTANT - License Compliance:** Set the license metadata of the Hugging Face repository to `cc-by-nc-sa-4.0` in the repository settings.

### Step 2: Download and Extract locally

1. Download the complete dataset archive from the [MVTec website](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).
2. Extract the archive into a single local staging folder (e.g. `~/Downloads/mvtec_ad_raw/`).

### Step 3: Login and Upload via CLI

1. Generate a **Write** access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2. Authenticate the CLI in your terminal:

   ```bash
   pixi run -e dev hf auth login
   ```

3. Upload the extracted folder directly to your dataset repository:

   ```bash
   pixi run -e dev hf upload foersben/mvtec-ad /path/to/extracted/mvtec_ad_raw/ . --repo-type dataset
   ```

---

## Retrieving the Data in the Workspace

Once the dataset is hosted on Hugging Face, anyone working on the project can download it locally by running:

```bash
just download-data
```

This runs the `huggingface_hub` downloader, which downloads the repository snapshot directly into your local `data/raw/mvtec_ad` directory. The folder is pre-ignored in `.gitignore` to prevent any local binaries from leaking back into Git commits.

## Automated Workspace Recipes

To make this workflow seamless, the [Justfile](file:///home/benni/Documents/antigravity_workspace/industrial-component-anomaly-detection/Justfile) includes targets to automate these operations:

### 1. Extract Dataset Packages (Optional)

If manually downloaded from the official website, you can extract the `.tar.xz` packages natively inside your workspace:

```bash
just extract-data
```

### 2. KeePassXC Integrated Login

If you have KeePassXC configured with the Freedesktop.org Secret Service integration, you can log in to Hugging Face non-interactively. Ensure your API token is stored as the password of an entry titled `[HuggingFace Access Token] [write] workspace-upload`.

Then run:

```bash
just hf-login
```

*(If the database is locked or the entry is missing, this will automatically fall back to standard interactive login).*

### 2. Auto-Upload Dataset

To upload your local dataset modifications back to Hugging Face, run:

```bash
just upload-data
```

*(You can optionally specify a custom source directory as an argument, e.g., `just upload-data ~/Downloads/my-data-folder`).*
