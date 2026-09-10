---
type: Concept
title: Architecture Transition
description: Explains the transition from a split FastAPI/Streamlit architecture to a monolithic Streamlit application.
tags: [architecture, streamlit, fastapi, simplification]
---

# Architecture Transition: FastAPI to Monolithic Streamlit

This document explains the transition from the legacy split architecture (FastAPI backend + Streamlit frontend) to the current unified **Monolithic Streamlit** application. 

---

## 1. The "Was" State: Split Architecture

Historically, the application was split into two separate services:
* **FastAPI Backend (Port 8000)**: Handled all model execution, training pipelines, caching logic, and JSON serialization.
* **Streamlit Frontend (Port 8501)**: Served as a dumb UI that issued HTTP requests to the FastAPI backend via a custom `APIClient` and rendered the JSON responses.

### Pain Points of the Split Architecture
1. **Serialization Overhead:** Complex NumPy arrays (like heatmaps and feature maps), PyTorch tensors, and large evaluation dataframes had to be constantly serialized to JSON or base64 over the network.
2. **State Duplication:** The same model caching logic had to be managed asynchronously across the network barrier. Streamlit session state and FastAPI caching mechanisms were often disjointed.
3. **Double Boilerplate:** Every new feature required writing an API endpoint, a Pydantic response model, and then a corresponding API client method in the Streamlit UI. 
4. **Race Conditions:** Running two concurrent processes via `just run` (using tools like `honcho` or background jobs) caused stdout/stderr interleaving, difficult debugging, and orphaned port processes on crash.

---

## 2. The "Is" State: Monolithic Streamlit

The application now runs entirely as a single process: a **Monolithic Streamlit** app (Port 8501).

### The New Design
1. **Direct Imports:** Instead of an `APIClient` making `requests.post()` calls, the Streamlit application directly imports the domain logic (e.g. `run_keras_cae_pipeline` or `run_patchcore_pipeline`).
2. **Native Python Objects:** We now pass raw NumPy arrays, pandas DataFrames, and Matplotlib figure objects directly between the modelling layer and the UI. There is no JSON/Base64 encoding step required for visualization.
3. **Simplified Setup:** Starting the app requires only one process (`streamlit run`). There is no complex orchestration needed to manage two interconnected services.

### Why this is Better (For Our Current Needs)
As an internal ML tooling and evaluation dashboard, we don't currently need to expose our anomaly detection models as a public REST API for external third parties. If we consider integrating with orchestration tools like MLflow, a direct monolithic python execution context remains highly compatible because MLflow primarily tracks executions via its Python SDK.

If we ever need a REST API in the future (e.g. for a production line integration), it is easier to expose a tiny FastAPI app that wraps the core pipeline methods, rather than forcing our interactive data science dashboard to suffer the overhead of HTTP serialization. 

---

## 3. Backward Compatibility & Shims

During the transition, we performed several cleanups:
* Removed `APIClient` and `fastapi` integrations from `app/ui`.
* Removed Pydantic JSON serialization logic in `app/pipelines/modelling/keras_cae/cae_pipeline.py`.
* Unified the `preprocessing_steps` configuration argument into `pipeline` across all models to better reflect the use of `PreprocessingPipeline`.

The core logic under `app/pipelines` remains isolated from the UI, ensuring that the **Vertical Slice Architecture** is maintained even in a monolithic state.
