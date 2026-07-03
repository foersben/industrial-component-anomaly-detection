---
type: Concept
title: Open Knowledge Format (OKF)
description: Detailed explanation of OKF, structuring documentation for AI, and validation.
tags: [okf, documentation, ai, knowledge-graph]
---

# Open Knowledge Format (OKF)

## What is it?
The Open Knowledge Format (OKF) is an emerging specification (spearheaded by Google Cloud) for structuring markdown documentation. It requires that markdown files include specific YAML frontmatter (metadata at the top of the file) and follow strict linking rules. This turns a directory of flat markdown files into a highly structured, machine-readable **Knowledge Graph**.

## Why do we do it?
In the era of AI-assisted development, documentation is no longer just for human engineers. It is critical context for Large Language Models (LLMs) and autonomous agents.

If an AI agent needs to understand the system, raw, unstructured markdown can lead to hallucinations or lost context. By enforcing OKF:
1. **Contextual Awareness:** The YAML frontmatter (like `type: Concept` or `tags: [api, reference]`) immediately tells the AI exactly what kind of document it is looking at before it reads the content.
2. **Navigability:** OKF enforces strict relative pathing for links. This allows AI agents to confidently traverse the documentation graph (e.g., following a link from an API reference to an architectural concept) without encountering broken links or getting lost in absolute paths.
3. **Agent State Management:** As seen in our `.agents/roles/` directory, OKF structures act as "persona constraints." The AI reads its OKF role file to understand its exact permissions and boundaries for a given task.

## How to use it
Every markdown file in `docs/` and `.agents/` **must** comply with OKF standards.

### The Frontmatter Requirement
Every file must begin with YAML frontmatter containing a `type` field (by convention we also include `title` and `description`).

```markdown
---
type: SOP
title: Database Migration Guide
description: Step-by-step instructions for running alembic migrations.
tags: [database, operations, sop]
---
# Database Migration Guide...
```

### The Validation Gate
You do not have to guess if your documentation is compliant. This repository includes a custom pre-commit hook script (`scripts/validate_okf.py`).

When you commit changes to Markdown files under `docs/` or `.agents/`, this script runs automatically. If you create a markdown file without YAML frontmatter (including a `type`), or if you create a broken relative `.md` link, the commit will be rejected immediately, protecting the integrity of the Knowledge Graph.
