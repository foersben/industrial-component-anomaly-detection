---
type: SOP
title: Implement Experimental Notebook and Lab Infrastructure
description: Procedural workflow for establishing an isolated, automated, and quality-gated notebook environment within a Pixi-managed project.
tags: [architecture, infrastructure, jupyter, automation]
---

## Infrastructure Specification: Experimental Notebook Labs

You are an expert AI software architect and automation engineer. Your task is to implement a dedicated, isolated, and collaborative workspace for experimental Python notebooks (`.ipynb`) and research labs inside this repository, adhering strictly to the project's Zero-Bypass Quality Gate and tooling constraints.

### 🎯 Hard Operational Constraints

1. **No Global/Bare Tools:** NEVER use bare `pip`, `poetry`, or global python commands. Everything must execute via `pixi run -e dev` or `just` macros.
2. **Architecture Boundaries:** Keep all notebook workflows completely separate from production code (`app/`). Do not leak experimental imports into `app/core/` or `app/pipelines/`.
3. **No CI Blockers:** Notebooks must not break production deployment gates due to strict typing or formatting guidelines, unless explicitly configured.

---

## 🛠️ Step-by-Step Implementation Instructions

### Step 1: Directory Structure Strategy

Create a root-level `notebooks/` directory. Ensure it supports both team feature exploration and isolated individual developer scratchpads:

- `notebooks/shared/` — For team-wide pipeline prototyping.
- `notebooks/scratch/` — Add a `.gitignore` inside this subfolder if individuals want uncommitted local scratchpads, or track them explicitly.

### Step 2: Update Dependency Baseline (`pyproject.toml`)

Ensure the required packages are present and safely configured:

1. Verify `jupyter`, `pandas`, and `scikit-learn` exist under `[tool.pixi.dependencies]`.
2. Add `nbstripout` to the `[dependency-groups] dev` array to handle notebook output purging before commits:

   ```toml
   [dependency-groups]
   dev = [
       # ... existing tools
       "nbstripout>=0.8.0",
   ]
   ```

### Step 3: Configure Quality Gate Overrides (`pyproject.toml`)

To prevent messy notebook prototypes from breaking the `strict = true` Mypy engine or raising noisy Ruff warnings during production commits, update the lint/type-checking matrices:

```toml
[tool.ruff]
# Explicitly exclude scratch/shared notebooks from core enforcement if desired,
# OR leave it un-excluded if you want Ruff to auto-format .ipynb files cleanly.
extend-exclude = ["notebooks/scratch"]

[tool.mypy]
# Exclude notebooks from mandatory type enforcement to encourage speed of experimentation
exclude = ["notebooks/"]
```

### Step 4: Implement Automation Recipes (`Justfile`)

Add clean execution workflows to the `Justfile` so team members do not have to memorize long environment strings:

```makefile
# Launch Jupyter Lab within the authenticated dev environment context
lab:
    pixi run -e dev jupyter lab --ip=127.0.0.1 --port=8888

# Clean out temporary notebook checkpoint files and caches
clean-notebooks:
    find notebooks/ -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
```

*Integrate `clean-notebooks` into your existing `just clean` target macro.*

### Step 5: Secure Git History Against Notebook Bloat (`.pre-commit-config.yaml`)

Notebooks store binary metadata and image outputs that cause massive Git conflicts and bloat. Integrate `nbstripout` directly into the local pre-commit gauntlet. Append this block cleanly:

```yaml
  - repo: https://github.com/kynan/nbstripout
    rev: 0.8.0
    hooks:
      - id: nbstripout
        files: ^notebooks/
```

---

## 🚦 Verification Guardrails

Before declaring this task complete, you MUST execute the local validation checks:

1. Run `just lint` to verify that updates to `pyproject.toml` and your new structure do not trigger style or layout violations.
2. Run `just test` to guarantee the test suite remains 100% green and fully functional.
3. Manually execute `just lab --help` or a dry run check to ensure Pixi successfully resolves the environment layer.
