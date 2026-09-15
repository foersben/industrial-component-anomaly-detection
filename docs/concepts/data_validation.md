---
type: Concept
title: Data Validation with Pydantic
description: Detailed explanation of Pydantic for data validation and settings management.
tags: [pydantic, validation, settings, schemas]
---

# Data Validation (Pydantic)

## What is it?

[Pydantic](https://docs.pydantic.dev/) is the most widely used data validation library for Python. It uses standard Python type hints to define data schemas and guarantees that the data matching those schemas is exactly what you expect.

## Why do we do it?

When data enters your application (from an API request, a database, or environment variables), you must ensure it is safe, correctly formatted, and properly typed.

### 1. Schema Enforcement

If an API endpoint expects a user's age as an integer, but a client sends `"25"` (a string), Pydantic will automatically coerce the string into an integer. If the client sends `"twenty-five"`, Pydantic will instantly raise a clear, standardized validation error, preventing the bad data from crashing your core business logic.

### 2. Pipeline Configuration Validation

Our anomaly detection pipelines and preprocessing transformations accept rich configuration objects (such as CLAHE clip limits, Gaussian blur kernels, crop strides, and coreset sampling ratios). Pydantic ensures that invalid hyperparameter values (e.g. negative crop sizes or ratios exceeding 1.0) are caught at initialization before costly training runs or evaluation sweeps begin.

### 3. Immutable Environment Settings

We use `pydantic-settings` to manage all environment variables (paths, logging levels, hardware overrides) inside `app/core/`. Pydantic validates these variables at application startup. If a critical environment variable is missing or incorrectly formatted, the application fails fast immediately rather than crashing mid-pipeline.

## How to use it

In our architecture, Pydantic schemas and dataclasses are co-located with their respective modules:

1. **Feature Schemas:** Inside your pipeline slice (e.g. `app/pipelines/modelling/patchcore/types.py`), define your data contracts:

   ```python
   from pydantic import BaseModel, Field


   class PatchCoreConfig(BaseModel):
       backbone: str = "resnet18"
       coreset_sampling_ratio: float = Field(default=0.1, gt=0.0, le=1.0)
       num_neighbors: int = Field(default=9, ge=1)
   ```

2. **Pipeline Integration:** Use these models to validate inputs in CLI entrypoints (`app/cli.py`) and Streamlit tabs.
3. **Core Configurations:** Check `app/core/config.py` to see how application-wide settings are defined and validated at startup.
