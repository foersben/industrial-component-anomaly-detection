# Industrial Component Anomaly Detection

A high-performance, deep-learning-powered industrial computer vision system engineered explicitly for **Vertical Slice Architecture (VSA)**, zero-bypass quality enforcement, and native **Multi-Agent AI Collaboration** (optimized for autonomous assistants like Antigravity).

This repository contains the production code, research notebooks, and operational pipelines designed to identify, isolate, and classify defects in industrial components. The architecture isolates complex CUDA-enabled graphics and deep learning dependencies using **Pixi** and separates detection features into decoupled vertical feature slices.

---

## 📚 Detailed Documentation

For a comprehensive, beginner-friendly deep dive into all the concepts and tools used in this repository, run `just serve` and open http://localhost:9000/ (local preview), or view the GitHub Pages site deployed from `main` (see `.github/workflows/docs.yml`).

The generated documentation includes detailed guides on:
* **Concepts & Architecture:** Deep dives into Vertical Slice Architecture, Pixi, OKF, Linting & Type Safety, and Data Validation.
* **Guides:** How to properly use the Jupyter Notebooks workspace.
* **Project Roadmap:** Strategic engineering milestones, deliverables log, and Phase 1-5 operational breakdowns.

---

## 🏛️ System Philosophy & Architecture

### 1. Vertical Slice Architecture (VSA)

Unlike traditional horizontal layered architectures (Three-Tier, Onion, Hexagonal) that split code by technical concerns (controllers in one folder, services in another), this project organizes code into completely self-contained, vertical features.

* Each slice within `app/pipelines/` contains all components required to fulfill a specific business capability.
* **Slices Implemented:**
  * `app/pipelines/anomaly_binary/`: Focuses exclusively on binary segregation (Normal vs. Anomalous).
  * `app/pipelines/anomaly_classification/`: Multi-class vertical slice triggered upon anomaly detection to classify defect types.
* **The AI Advantage:** Horizontal architectures force an AI agent to hop across multiple directories to implement a single feature, dramatically increasing token consumption, context window fragmentation, and the risk of context drift. Under VSA, an agent operates within a strictly bounded, discrete folder structure without leaking logic across distant parts of the application tree.

### 2. The Zero-Bypass Quality Gate

This repository operates on a strict "fail-fast" principle. Code cannot be committed, pushed, or deployed unless it passes a localized gauntlet of formatting, strict type-checking, cryptographic secret scanning, and documentation compliance validation.

### 3. Open Knowledge Format (OKF v0.1)

To ensure that all internal repository knowledge stays completely portable and optimal for secondary AI ingestion loops, this repository follows **Google Cloud's OKF v0.1 spec**. All Markdown documentation files inside `docs/` and `.agents/` are treated as structured nodes in an interlinked Knowledge Graph, strictly verified by a custom local Python gatekeeper script.

### 4. Why Pixi for Data Science & ML?

Unlike traditional Python package managers (like `pip`, `poetry`, or `uv`) that focus strictly on Python packages, this repository utilizes **Pixi** to natively support complex machine learning and data science workloads:

* **Non-Python / Binary Dependency Management:** Machine learning projects often depend heavily on complex binary libraries (such as CUDA, OpenBLAS, or C++ wrappers). Pixi is built on the Conda ecosystem, enabling it to install system-level binary dependencies directly into the local environment without relying on system package managers.
* **Deterministic GPU/CPU Environments:** Using Pixi's native `environments` and `features` systems (defined in `pyproject.toml`), this template dynamically manages separate solve groups for GPU-enabled environments (e.g., PyTorch with CUDA 12.1) and lightweight CPU-only environments (e.g., for CI/CD runs).
* **Multi-Platform Consistency:** Pixi generates a unified `pixi.lock` file that maps dependencies across multiple target architectures (Linux, macOS, Windows) and environments, eliminating the "works on my machine" class of environment problems when deploying ML pipelines.

---

## 🛠️ Core Technology Stack & Tool Matrix

This ecosystem relies on modern, lightning-fast Rust-backed and asynchronous Python tooling to maintain a bulletproof environment.

| Tool | Component | Explicit Operational Task |
| --- | --- | --- |
| **Python 3.12+** | Runtime Environment | The core target engine. Fully locked into type checkers and compiler layers. |
| **pixi** | Package Manager | Resolves dependencies in milliseconds and forces absolute workspace isolation via a deterministic lock engine (`pixi.lock`). Completely replaces `pip`, `poetry`, `virtualenv`, and `conda`. |
| **Hatchling** | Build Backend | Modern PEP 621 compliant build backend managing pure source target mappings. |
| **FastAPI** | Core Framework | Asynchronous, high-performance web framework providing foundational API routing and dependency injection layers. |
| **Pydantic v2** | Data Validation | Strict data parsing, schema enforcement, and environment variable settings management (`pydantic-settings`). |
| **Ruff** | Linter & Formatter | Written in Rust, replacing `flake8`, `black`, and `isort`. Enforces a 120-character line limit and strict adherence to Google Python Style Guide docstring rules. |
| **Mypy** | Type Enforcement | Configured in `strict = true` mode with `explicit_package_bases = true` to guarantee total type safety across isolated packages without duplicate tracking overlaps. |
| **Pytest** | Test Engine | Runs asynchronous unit, integration, and E2E validation passes (`pytest-asyncio`) with strict coverage metrics tracking (`pytest-cov`). |
| **Zensical** | Docs Engine | Compiles technical code documentation via `mkdocstrings[python]` and validates graph link structural integrity using `mkdocs-htmlproofer-plugin`. |
| **Just** | Task Automation | A local recipe runner (`Justfile`) that orchestrates all workspace commands, replacing complex bash scripts or overloaded `Makefile` syntax. |

---

## 📂 Architecture & Directory Layout

```text
├── .agents/                # Autonomous Agent Operational Rules & Workflows
│   ├── roles/              # Explicit Multi-Agent persona state constraints
│   ├── rules/              # Global framework and behavioral guidelines
│   └── workflows/          # Task-specific execution paths (e.g., implement.md)
├── .vscode/                # Isolated workspace configurations & extension requirements
├── app/                    # Application Source Tree
│   ├── api/                # Global API routing and middleware definitions
│   ├── core/               # Immutable core configs (Pydantic Settings) and exceptions
│   ├── domain/             # Pure domain logic and models (zero external dependencies)
│   └── pipelines/          # Isolated, self-contained feature slices
│       ├── anomaly_binary/ # Binary anomaly detection (Normal vs Anomalous)
│       └── anomaly_classification/ # Defect class sorting engine
├── docs/                   # Markdown architecture guides, references, and system logs
├── scripts/                # Local automated continuous integration tools
│   └── validate_okf.py     # Script verifying documentation complies with Google OKF
├── tests/                  # Deterministic testing suite
│   ├── unit/               # Localized module verification logic
│   ├── integration/        # External adapter and pipeline interaction logic
│   └── conftest.py         # Session-scoped asynchronous loop management configuration
├── .clineignore            # Context protection limits for IDE coding assistants
├── .clinerules             # Absolute behavioral guardrails for AI agents
├── .pre-commit-config.yaml # Background git hook validation engine
├── Justfile                # Canonical task configuration registry
├── pyproject.toml          # Monolithic tool parameters (Ruff, Mypy, Pytest)
└── zensical.toml           # Technical documentation and site generation configuration

```

---

## 🔐 Enterprise Security Model

To prevent catastrophic credential leaks, this repository explicitly bans the storage of raw API keys or passwords in `.env` files.

**The Security Posture:**

1. **Local Vaulting:** All environment secrets must be stored externally using **KeePassXC** integrated with the OS **Secret Service API (D-Bus)** and an **SSH Agent**.
2. **Explicit Confirmation:** SSH keys must be loaded with explicit confirmation flags (`ssh-add -c`). This guarantees that if an autonomous IDE agent attempts to authenticate via Git or SSH in the background, a physical GUI prompt will block execution until a human clicks "Allow."
3. **Cryptographic Baseline:** The workspace uses `detect-secrets` via a `pre-commit` hook to scan every changed line for high-entropy strings.

* *Note:* The security baseline (`.secrets.baseline`) is **intentionally generated manually** during setup. Automating this generation would risk silently whitelisting real production secrets.

---

## 🤖 Multi-Agent Governance Network

This repository is designed to host a "Role-Playing State Machine" for single-thread AI IDE assistants (like Roo Code). Rather than relying on a complex external multi-agent framework, the agent shifts its own context constraints, skill sets, and write-permissions step-by-step using OKF-compliant Markdown profiles.

### The Persona Registry (`.agents/roles/`)

* **`01-orchestrator.md`:** The Lead. Analyzes user requests, generates technical OKF specs, and delegates state transitions. *Forbidden from writing functional code.*
* **`02-architect.md`:** The System Designer. Defines pure domain entities and core Pydantic configurations. *Forbidden from writing tests.*
* **`03-engineer.md`:** The Implementer. Scaffolds OpenCV and PyTorch logic strictly inside `app/pipelines/anomaly_binary/` and `app/pipelines/anomaly_classification/` to turn red tests green. Enforces Google-style docstrings. *Forbidden from touching core configs.*
* **`04-qa-automator.md`:** The Gatekeeper. Scaffolds failing TDD tests based on specs and verifies edge cases in `tests/`. *Forbidden from modifying `app/` logic.*

### OKF Graph Compliance

Every Markdown file in the system acts as a node in the agentic knowledge graph. Every file **must** begin with standard YAML frontmatter explicitly declaring a `type`, and must exclusively use relative paths for linking.

```markdown
---
type: SOP
title: Complex Feature Multi-Agent Lifecycle
description: Sequential multi-role pipeline for complex system feature generations.
tags: [agentic, development, workflow]
---
# Lifecycle Implementation...

```

*If a human or agent attempts to commit an unstructured text file lacking this metadata, the `validate_okf.py` pre-commit hook will instantly reject the commit.*

---

## 🚀 Getting Started (Human & Machine Initialization)

### 1. Prerequisites

Ensure the following tools are globally accessible on your host machine:

* `pixi`
* `just`
* `git`

### 2. IDE Setup (VS Code / Cursor)

Open the repository folder directly inside your editor:

```bash
code .

```

The workspace reads `.vscode/extensions.json` and will automatically prompt you to install the optimized extension bundle (including `Ruff`, `Strict Mypy`, `Error Lens`, and `nefrob.vscode-just-syntax`). **Accept and install all recommendations to ensure live editor highlighting matches the CI quality gates.**

### 3. Bootstrap Environment

Run the single wrapper recipe to automatically configure your localized virtual environment, synchronize all dev/runtime dependency channels, and securely map git hooks:

```bash
just setup

```

### 4. Download Dataset (MVTec ITODD)

To retrieve the dataset, we host the ~7.5 GB Base Package and 3D range/image data on Hugging Face (avoiding heavy Git LFS commits). To fetch it to your local workspace, run:

```bash
just download-data
```

For detailed instructions on the dataset scope, how to upload it to Hugging Face, or how it is structured, refer to the [Dataset Setup Guide](./docs/guides/dataset_setup.md).

### 5. Extract Dataset

After downloading the archives, extract them natively to your workspace:

```bash
just extract-data
```

### 6. Initialize Your Security Baseline

To activate the secret scanner tripwire, create your local cryptographic tracking baseline and register it with git:

```bash
pixi run -e dev detect-secrets scan > .secrets.baseline
git add .secrets.baseline

```

### 7. Link Visual Studio Code Interpreter

1. Open the VS Code Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`).
2. Type **Python: Select Interpreter**.
3. Choose the path pointing directly to the locally generated `.venv/bin/python` folder.

---

## ⚙️ Development Task Matrix

Use `just` to coordinate all workspace tasks. **Never call bare `pip`, `poetry`, or global environment commands.**

| Command | Action Performed |
| --- | --- |
| `just default` | Lists every automated command recipe currently available. |
| `just setup` | Installs isolated virtual environments, configures pre-commit hooks, creates the `.venv` symlink, and installs extensions. |
| `just install` | Alias mapping directly to the `setup` macro. |
| `just link-venv` | Recreates the local `.venv` symlink pointing to the active `.pixi/envs/dev` directory. |
| `just download-data` | Downloads the raw MVTec ITODD dataset from Hugging Face Hub. |
| `just extract-data` | Extracts the downloaded `.tar.xz` dataset packages locally. |
| `just hf-login` | Integrates with KeePassXC Secret Service to log in to Hugging Face Hub (falls back to interactive login). |
| `just upload-data` | Uploads a local directory back to the Hugging Face Hub dataset repository. |
| `just lint` | Sequentially auto-fixes lint errors, enforces layout formatting, evaluates strict types via Mypy, and runs the OKF compliance Python script. |
| `just test` | Runs the asynchronous test suite via Pytest with code coverage matrix evaluation. |
| `just format` | Safely forces code blocks to match global stylistic spacing layout parameters. |
| `just docs` | Compiles your docstrings and Markdown graph into a live local documentation page reflecting structural source modifications. |
| `just clean` | Purges localized compiler cache states, artifact allocations (`__pycache__`, `.mypy_cache`), and lock remnants. |

---

## 🛡️ Pre-Commit Integrity Framework

A robust suite of checks runs automatically before any commit can seal.

1. **Syntax & Whitespace:** Fixes trailing spaces and ensures single-newline EOF structures.
2. **Security Tripwires:** Runs `detect-secrets` against the `.secrets.baseline` file.
3. **Lexical Validation:** Executes `codespell` to catch typos in code, strings, and documentation via `tomli` parsing.
4. **Code Quality:** Triggers `ruff-check`, `ruff-format`, and `mypy` locally targeting `app/` and `scripts/` with `pass_filenames: false` to ensure global context evaluation.
5. **Knowledge Graph Integrity:** Triggers `validate_okf.py` to ensure all AI context boundaries remain parsable.
6. **Author Identity:** A bash gate confirming `commit.gpgsign = true` is active locally, ensuring all commits are cryptographically verified to a human author.

---

## ⚙️ CI/CD Pipeline Automation & Dependency Management

The continuous integration and documentation deployment pipelines are managed via GitHub Actions. All workflows are optimized for speed, reliability, and security.

### Node.js 24 & Modern Runner Compliance

To comply with GitHub's runner upgrades and deprecation of older Node.js runtimes, all workflow files use modern action versions:

* **`actions/checkout@v6`**: Used for repository checkout, providing improved credential security, stability fixes, and native Node.js 24 compatibility.
* **`prefix-dev/setup-pixi@v0.10.0`**: Used to bootstrap the Pixi package environment, complying with Node.js 24 upgrade paths.

### Automated Updates with Dependabot

To prevent version drift and runtime deprecation issues, this repository includes a Dependabot configuration (`.github/dependabot.yml`). Dependabot is configured to:

* Monitor all GitHub Actions dependencies.
* Scan for updates on a weekly schedule.
* Automatically open Pull Requests when new versions (like future major action releases) are published.

---

## ⚙️ CI/CD Pipeline Phases

The repository’s continuous integration workflow is managed via `.github/workflows/ci.yml` and is split into two highly optimized, cache-enabled automated phases.

### Job 1: Code & Context Quality Gate

Runs automatically on every `push` and `pull_request` targeting the `main` branch.

* Spawns an isolated `ubuntu-latest` runner equipped with Python 3.12 and `pixi` dependency environment.
* Triggers all pre-commit hooks globally across all workspace assets.
* *Automation Rule:* Environment-specific identity hooks (`check-identity`, `enforce-author-identity`) are **explicitly bypassed (`SKIP`)** inside the cloud execution layer, as virtual machine runners lack local human GPG credentials.
* Executes the complete `pytest` matrix to ensure all functional components remain verified.

### Job 2: Build & Deploy Documentation

Triggers **only** after Job 1 passes perfectly, and only upon direct pushes to the `main` branch.

* Locks dependencies using the `ci-dev` environment setup via `pixi`.
* Compiles your markdown guides and codebase docstrings via `zensical build`.
* Safely packages the resulting `./site` directory and deploys it natively to **GitHub Pages**, providing a centralized, universally accessible OKF-compliant documentation site for your system.

---

## 📄 License & Attribution

* **Codebase:** Distributed under the [MIT License](./LICENSE) (or your chosen code license).
* **Dataset (MVTec ITODD):** Distributed strictly under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License ([CC BY-NC-SA 4.0](./LICENSE-DATA.md)). This project is compliant with non-commercial usage terms.

### Academic Citation

If you use this dataset or codebase in scientific or academic work, please cite the original authors:

> Bertram Drost, Markus Ulrich, Paul Bergmann, Philipp Härtinger, and Carsten Steger. *Introducing MVTec ITODD — A Dataset for 3D Object Recognition in Industry*; in: IEEE International Conference on Computer Vision (ICCV), 2200-2208, October 2017.
