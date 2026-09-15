---
type: Concept
title: Vertical Slice Architecture
description: Detailed explanation of Vertical Slice Architecture, why it is used, and how it differs from horizontal layers.
tags: [architecture, vsa, design-patterns]
---

# Vertical Slice Architecture (VSA)

## What is it?

Vertical Slice Architecture (VSA) is a software design pattern where the codebase is organized around **features** (or business capabilities) rather than technical concerns.

In a traditional "Horizontal" architecture (like MVC, Onion, or N-Tier), you organize code by its technical role:

* All Database logic goes in a `models/` or `repositories/` folder.
* All Business logic goes in a `services/` folder.
* All Web routing goes in a `controllers/` or `api/` folder.

In **Vertical Slice Architecture**, you create a folder for a specific feature (e.g., `create_user`), and inside that folder, you put *everything* needed to make that feature work: the route, the business logic, the database query, and the data models.

## Why do we do it?

### 1. High Cohesion, Low Coupling

When you change a feature in a horizontal architecture, you often have to open 5 different files across 5 different directories. In VSA, changing a feature means you only work inside one specific folder. The feature is highly cohesive (everything it needs is nearby) and loosely coupled (it doesn't depend heavily on other features).

### 2. The AI Advantage

This template is optimized for AI coding assistants (like Roo Code, Cursor, or Copilot). A major challenge for AI agents is **context window management**.
If an AI has to navigate a horizontal architecture, it must load a controller file, a service file, a repository file, and a schema file just to understand one feature. This fragments its context and increases token usage and hallucinations.
With VSA, the agent is directed to a single folder (e.g., `app/pipelines/register_user`). It has all the context it needs in one isolated location, drastically improving the AI's ability to generate correct, contained code without breaking distant parts of the application.

### 3. Easier to Delete and Refactor

If a feature is deprecated, you just delete the folder. There are no lingering routes in a massive `routes.py` file or dead code in a global `UserService`.

## How to use it in this repository

Feature slices are organized domain-first inside `app/pipelines/`:

1. **Modelling Slices (`app/pipelines/modelling/`):**
   * Dedicated subpackages for each model family: `patchcore/`, `keras_cae/`, and `dino/`.
   * Each slice encapsulates its dataset loader, model definition, training/fitting logic, Optuna hyperparameter study, and evaluation pipeline.
2. **Preprocessing Slices (`app/pipelines/preprocessing/`):**
   * Cohesive preprocessing steps in `steps/` (`clahe.py`, `gaussian_blur.py`, `foreground_mask.py`).
   * Step composition factory (`factory.py`) and standard adapter (`adapter.py`).
3. **Evaluation Slices (`app/pipelines/evaluation/`):**
   * Standardized scientific evaluation metrics (`metrics.py`), adaptive threshold calibration (`cae_metrics.py`, `scoring.py`), and visualization/heatmap generators (`heatmaps.py`).
4. **Presentation Slices (`app/ui/tabs/` and `app/cli.py`):**
   * Interactive Streamlit dashboard tabs co-located by feature area.
   * Command-line interface subcommands dispatching directly to modelling pipeline methods.
