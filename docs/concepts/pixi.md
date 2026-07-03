---
type: Concept
title: Pixi Package Management
description: Detailed explanation of Pixi, its benefits for Data Science/ML workloads, and usage.
tags: [package-manager, pixi, dependencies, ml]
---

# Pixi Package Management

## What is it?

[Pixi](https://pixi.sh/) is a blazing-fast, modern package manager and workflow tool built in Rust. It is developed by **prefix.dev**, a company founded by core maintainers of the Conda ecosystem (including developers of Mamba and conda-forge).

Under the hood, Pixi uses **[uv](https://github.com/astral-sh/uv)** (Astral's ultra-fast Python packaging tool) to resolve and install PyPI dependencies. This enables Pixi to merge the absolute best of both worlds:

1. **Conda/Mamba ecosystem's** ability to manage binary, system-level, and multi-language dependencies (like CUDA, compilers, or native C++ libraries).
2. **`uv`'s** blazing-fast PyPI package resolution and installation speeds.

It completely replaces traditional Python tooling like `pip`, `poetry`, `virtualenv`, and traditional `conda`.

## Who is behind it and why did they build it?

Pixi was created by prefix.dev, whose team has deep roots in the Conda and Mamba package management ecosystems.

### The Motivation

For years, developers working with Python for Data Science and Machine Learning faced a difficult trade-off:

- **Conda** was excellent for complex binary dependencies (such as GPU drivers, CUDA, and C++ libraries), but the classic Python-based client was notoriously slow, and its CLI/workflow experience felt dated compared to modern tools like Cargo (Rust) or npm (JavaScript).
- **Pip/Poetry** were fast and offered modern lockfile/workflow support, but they could not install system-level binaries or non-Python packages, leading to complex, fragile system configurations when setting up ML tools.

To bridge this gap, the creators of Pixi set out to build a tool that:

1. **Leverages Rust**: Rewriting the client from scratch in Rust for unmatched speed, safety, and single-binary distribution (no Python bootstrap required).
2. **Unifies Ecosystems**: Allowing developers to specify both Conda dependencies (from Conda-Forge) and PyPI dependencies (using `uv` as the underlying engine) in a single configuration file (`pixi.toml` or `pyproject.toml`).
3. **Improves Developer Experience (DX)**: Providing a modern, declarative workflow with first-class support for tasks, lockfiles, multi-platform builds, and environment switching.

## Why do we do it?

Python's standard package managers (like `pip` or `poetry`) are excellent for pure Python code. However, they struggle significantly when projects involve complex machine learning, data science, or native C/C++ dependencies.

### 1. Non-Python & Binary Dependencies

Machine Learning frameworks (like PyTorch or TensorFlow) rely heavily on system-level libraries like CUDA, cuDNN, OpenBLAS, or native compiler toolchains. `pip` cannot install system libraries. Pixi, leveraging the Conda-Forge ecosystem, can install **everything**—not just Python packages, but C++ libraries, compilers, and even tools like `just` or `git`, ensuring a 100% reproducible environment.

### 2. Deterministic Environments & Lockfiles

Pixi generates a `pixi.lock` file. This lockfile resolves dependencies not just for your current machine, but across multiple platforms (Linux, macOS, Windows). This guarantees that what works on your laptop will work exactly the same way in the GitHub Actions CI/CD pipeline, eliminating "it works on my machine" issues.

### 3. Environment Segregation (CPU vs GPU)

Pixi allows us to define different "environments" in `pyproject.toml`. We can have a heavy `cuda` environment for local model training with GPU support, and a lightweight `default` or `ci` environment (CPU only) for running automated tests or linting in the cloud, all managed from the same lockfile.

## How to use it

You rarely need to interact with `pixi` directly if you are using the provided `Justfile` commands, but understanding the core commands is helpful:

- **Installing the project:** `pixi install` (This is automatically run via `just setup`). It creates a hidden `.pixi` folder containing the isolated environments.
- **Running a command in the environment:** `pixi run <command>`. For example, `pixi run pytest` runs tests inside the isolated environment without needing to "activate" a virtualenv first. We often use `-e dev` to specify the development environment: `pixi run -e dev pytest`.
- **Adding a dependency:** `pixi add <package_name>`. This updates `pyproject.toml` and regenerates the `pixi.lock` file.
- **Adding a dev dependency:** `pixi add --feature dev <package_name>`.

*Note: Never use `pip install` directly in this repository. Always use `pixi add` to ensure the lockfile remains accurate.*

## CI/CD Integration & Automated Updates

To ensure absolute reproducibility of the Pixi environment in the cloud, our continuous integration pipelines integrate directly with the Pixi ecosystem.

### Node.js 24 Upgrade & Actions Compliance

GitHub Actions runners are continuously upgraded to secure, modern runtimes. To support the Node.js 24 runtime upgrade and prevent runner deprecation warnings, the workflows in this repository utilize modern versions of core actions:

- **`actions/checkout@v6`**: Upgraded to provide native support for the Node.js 24 runner environment, improved credential protection, and performance fixes.
- **`prefix-dev/setup-pixi@v0.10.0`**: Upgraded to align with modern Node.js runners and package resolving changes. This action installs the Pixi executable on the runner, configures shell integration, and manages caching of Pixi environments for fast CI runtimes.

### Automated Dependency Management via Dependabot

Maintaining updated GitHub Action versions manually is prone to oversight. To automate version upgrades, a [Dependabot configuration](file:///.github/dependabot.yml) is situated in the `.github/` folder:

- **Ecosystem**: `github-actions`
- **Schedule**: Weekly checks for new versions of checkout, setup-pixi, and other Actions.
- **Result**: Automatically creates Pull Requests to keep actions secure and compliant with the latest runtimes without manual intervention.
