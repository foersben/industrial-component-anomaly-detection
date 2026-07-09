---
type: Guide
title: Jupyter Notebooks Workflow
description: Guide on how to manage individual and shared Jupyter notebooks within the repository.
tags: [jupyter, notebooks, data-science, workflow]
---

# Jupyter Notebooks Workflow

## The `notebooks/` Directory

The `notebooks/` directory at the root of the repository is the dedicated space for all exploratory data analysis, algorithm prototyping, and interactive documentation via Jupyter Notebooks (`.ipynb`).

To accommodate teams working collaboratively on data science or machine learning tasks, this directory is split into two distinct sub-folders with very different operational rules: `scratch/` and `shared/`.

### 1. `notebooks/scratch/` (Individual Playgrounds)

The `scratch/` directory is designed for **personal, ephemeral work**.

* **Purpose:** Use this area to experiment, test hypotheses, or write messy code that isn't ready for review.
* **Git Behavior:** This directory is usually heavily ignored by `.gitignore` (or developers are expected not to commit these files to the main branch). It is your local sandbox.
* **Collaboration:** Do not rely on files in `scratch/` for team operations. Everyone's `scratch/` folder will look different.

### 2. `notebooks/shared/` (Production & Collaborative Notebooks)

The `shared/` directory is for **clean, reviewed, and collaborative notebooks**.

* **Purpose:** Use this area for finalized analysis, repeatable data pipelines, interactive tutorials, or architecture demonstrations that the whole team needs to see.
* **Git Behavior:** Files here are committed to version control and reviewed via Pull Requests.
* **Standards:** Notebooks in `shared/` should be well-documented (using markdown cells), have their outputs cleared before committing to minimize diff noise, and run successfully top-to-bottom.

## How to run JupyterLab

Because we use Pixi to isolate our environment and dependencies, you should **never** run `jupyter lab` directly from your global system environment.

We have provided a unified command in the `Justfile` to launch JupyterLab correctly configured with the repository's dependencies.

From the root of the repository, run:

```bash
just lab
```

This command will:

1. Use Pixi to ensure all required data science dependencies (like `jupyter`, `pandas`, etc., defined in `pyproject.toml`) are available.
2. Launch a local JupyterLab server bound to `127.0.0.1` on port `8888`.
3. Open the Jupyter interface in your default web browser, pointing directly to the repository root so you can easily navigate into `notebooks/scratch/` or `notebooks/shared/`.

## Maintenance

Jupyter notebooks often generate hidden `.ipynb_checkpoints` folders that clutter the workspace. You can clean these up instantly by running:

```bash
just clean-notebooks
```
