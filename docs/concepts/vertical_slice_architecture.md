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

## How to use it in this template

All feature slices are placed inside the `app/pipelines/` directory.

When you want to add a new feature, you:

1. Create a new directory under `app/pipelines/`, for example: `app/pipelines/process_payment/`.
2. Inside this directory, create the necessary files:
   * `models.py` (Pydantic schemas specifically for this feature)
   * `router.py` (The FastAPI endpoint)
   * `logic.py` (The core business rules and database interactions)
3. Wire the `router.py` into the main FastAPI application in `app/api/`.

By keeping the feature isolated, you ensure that if the payment process gets incredibly complex, it won't clutter the rest of the application.
