---
type: SOP
title: Complex Feature Multi-Agent Lifecycle Workflow
description: Sequential multi-role pipeline for complex system feature generations.
tags: [architecture, management, multi-agent]
---

# Complex Feature Implementation Workflow

This operational checklist outlines how to execute advanced system transformations by passing the active context through multiple specialized agent states.

## 📋 Phase 1: Planning (Role: Lead Orchestrator)

1. Read the user's technical request completely.
2. Adopt the **Lead Orchestrator** role instructions.
3. Generate a comprehensive architectural design matrix detailing the required domain model additions and save it directly to `docs/architecture/feature-spec.md`.
4. Define clear validation expectations and step criteria for the testing loops.

## 📋 Phase 2: Test Scaffolding (Role: QA Automation Engineer)

1. Drop the Orchestrator persona and adopt the **QA Automation Engineer** role constraints.
2. Review the feature specifications created in Phase 1.
3. Open `tests/` and construct failing test modules (`test_*.py`) that map exactly to the planned spec criteria.
4. Run `just test` to confirm that all new tests fail cleanly (`Red` state).

## 📋 Phase 3: Slicing Implementation (Role: Core Software Engineer)

1. Drop the QA persona and adopt the **Core Software Engineer** role constraints.
2. Review the failing test logs and the feature brief.
3. Scaffold a completely isolated vertical feature slice folder inside `app/pipelines/`.
4. Code the exact application logic required to satisfy the failing test conditions.
5. Ensure all public classes and functions contain strict type signatures and Google-style docstrings.

## 📋 Phase 4: Validation Gate (Role: QA Automation Engineer)

1. Drop the Engineer persona and re-adopt the **QA Automation Engineer** role.
2. Run the technical validation suite:

   ```bash
   just test
   ```
