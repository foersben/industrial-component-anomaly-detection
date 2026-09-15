"""The application package for the industrial component anomaly detection system.

This package contains the CLI, UI, and evaluation pipelines.

- **CLI**: `app.cli`
- **UI**: `app.ui`
- **Pipelines**: `app.pipelines`
"""

from app.cli import main

__all__ = ["main"]

__version__ = "0.1.0"
