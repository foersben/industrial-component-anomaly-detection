---
type: Guide
title: Dataset Setup & Hugging Face Integration
description: Instructions on how to download, host, and retrieve the MVTec ITODD dataset for the project.
tags: [dataset, huggingface, setup, guide]
---

# Dataset Setup & Hugging Face Integration

This guide documents how to acquire, host, and retrieve the **MVTec ITODD (Industrial Three-Dimensional Object Detection)** dataset for the project.

## Dataset Selection & Scope

For a production-grade anomaly detection pipeline, we utilize the following packages from the MVTec ITODD dataset (~7.5 GB total):

1. **Base Package (150 MB):** Contains CAD models, calibration information, and metadata.
2. **3D Data (Short/Long Baseline Sensors - 7.5 GB total):** Contains range images (X, Y, Z coordinates) and the aligned left-camera 2D grayscale image for each scene.

### Why we omit the 100 GB 2D Grayscale Packages

We do not download or host the separate multi-gigabyte 2D camera packages (which total ~100 GB). Doing so is an unnecessary resource drain because:

* **2D Views are Included:** The 3D Data packages already bundle the high-quality left-camera 2D views corresponding to each depth map.
* **Avoid Pattern Contamination:** The extra grayscale packages include projected patterns designed for active stereo reconstruction. These patterns artificially alter component visuals and contaminate anomaly detection features.
* **Pipeline Speed:** Storing and retrieving an 8 GB dataset is fast and lightweight, whereas a 100 GB dataset introduces network and storage bottlenecks during CI/CD execution and local development.

---

## Uploading the Data to Hugging Face Hub (One-Time Setup)

To allow team members and CI/CD pipelines to download the dataset on demand without cloning large binary files in Git, we host the dataset on the Hugging Face Hub.

### Step 1: Create a Hugging Face Dataset Repository

1. Register or log in at [huggingface.co](https://huggingface.co/).
2. Create a new dataset repository (e.g. `your_hf_username/mvtec-itodd`).
3. Set the visibility to **Public** (CC BY-NC-SA 4.0 allows public sharing for non-commercial research, making it easy for automation tools to download without tokens).

### Step 2: Download and Extract locally

1. Download the **Base Package** and **3D Data** packages from the [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-itodd).
2. Extract all of them into a single local staging folder (e.g. `~/Downloads/mvtec_itodd_raw/`).

### Step 3: Login and Upload via CLI

1. Generate a **Write** access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2. Authenticate the CLI in your terminal:

   ```bash
   pixi run -e dev huggingface-cli login
   ```

3. Upload the extracted folder directly to your dataset repository:

   ```bash
   pixi run -e dev huggingface-cli upload <your_hf_username>/mvtec-itodd /path/to/extracted/mvtec_itodd_raw/ . --repo-type=dataset
   ```

---

## Retrieving the Data in the Workspace

Once the dataset is hosted on Hugging Face, anyone working on the project can download it locally by running:

```bash
just download-data
```

This runs the `huggingface_hub` downloader, which downloads the repository snapshot directly into your local `data/raw/mvtec_itodd` directory. The folder is pre-ignored in `.gitignore` to prevent any local binaries from leaking back into Git commits.

After downloading, you must extract the compressed packages locally using:

```bash
just extract-data
```

## Automated Workspace Recipes

To make this workflow seamless, the [Justfile](file:///home/benni/Documents/antigravity_workspace/industrial-component-anomaly-detection/Justfile) includes targets to automate these operations:

### 1. Extract Dataset Packages

To extract the downloaded `.tar.xz` packages (Base Package and 3D data) natively inside your workspace:

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
