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

### 2. Deep Integration with FastAPI
FastAPI, our web framework, is built entirely around Pydantic. When you define a Pydantic model for an API request, FastAPI automatically generates the OpenAPI (Swagger) documentation, validates incoming requests, and serializes outgoing responses.

### 3. Immutable Environment Settings
We use `pydantic-settings` to manage all environment variables (database URIs, API keys, feature flags) inside the `app/core/` directory. Pydantic validates these variables at application startup. If a critical environment variable is missing or incorrectly formatted (e.g., a port number is provided as a word), the application will fail to start immediately ("fail-fast"), rather than crashing unpredictably hours later in production.

## How to use it
In Vertical Slice Architecture, Pydantic models (schemas) should be placed as close to the feature as possible.

1. **Define a Model:** Inside your feature slice (e.g., `app/pipelines/create_user/models.py`), define your data structures.
   ```python
   from pydantic import BaseModel, EmailStr

   class UserCreateRequest(BaseModel):
       username: str
       email: EmailStr
       age: int
   ```
2. **Use in Routing:** Import this model into your `router.py` to type-hint the FastAPI endpoint.
3. **Core Configurations:** Look at `app/core/` to see how global settings are defined and validated at startup.
