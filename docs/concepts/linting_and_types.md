---
type: Concept
title: Linting & Type Safety
description: Detailed explanation of Ruff, strict Mypy, and zero-bypass quality gates.
tags: [linting, typing, ruff, mypy, quality]
---

# Linting & Type Safety (Ruff & Mypy)

## What is it?

This repository enforces an uncompromising standard for code quality. We use two primary tools to achieve this:

1. **Ruff:** An extremely fast Python linter and code formatter written in Rust. It consolidates and replaces legacy tools like `Flake8`, `Black`, `isort`, and `pylint`.
2. **Mypy (Strict Mode):** A static type checker for Python. We run it in its most aggressive configuration (`strict = true`).

## Why do we do it?

Python is dynamically typed and notoriously flexible. While this makes it fast to write, it makes it incredibly fragile in large enterprise systems or when multiple AI agents are generating code.

### 1. Speed & Consistency with Ruff

Before Ruff, developers had to wait for multiple slow Python-based tools to check their code. Ruff executes in milliseconds. We use it to enforce a single, objective style across the entire repository (120-character line limits, sorted imports, Google-style docstrings). This ends debates about code style during code reviews—the machine decides, and the machine is always right.

### 2. Bulletproof Interfaces with Strict Mypy

By enforcing `strict = true` in Mypy, we strip away Python's dynamic ambiguity. Every function argument, return type, and variable must have an explicit type hint.

* **Human Benefit:** You get perfect autocomplete in your IDE (VS Code) and catch `NoneType` errors before you even run the code.
* **AI Benefit:** AI agents rely heavily on type hints to understand how components connect. If an AI sees `def process(data):`, it has to guess what `data` is. If it sees `def process(data: UserProfile) -> TransactionResult:`, the AI has concrete boundaries and generates vastly more accurate code.

## How to use it

These tools act as a "Zero-Bypass Quality Gate." You cannot ignore them.

* **During Development:** The `.vscode/extensions.json` ensures your editor has the Ruff and Mypy extensions installed. You will see red squiggles immediately if you violate a type or formatting rule.
* **Manual Execution:** Run `just lint` to automatically fix formatting issues and run the Mypy type checker across the `app/` directory.
* **The Git Gate:** These tools run automatically as `pre-commit` hooks. If your code is not formatted properly or if Mypy detects a type mismatch, **you cannot commit the code**. The system will block the `git commit` command until you fix the errors.
