---
type: rules
---
# Workspace Rules

## Git Settings

* **Never bypass Git settings:** Do not use `--no-gpg-sign` or other flags to bypass user Git configurations (such as GPG commit signing). If a commit fails due to a missing GPG key/agent in the agent shell, report the issue to the user and ask them to perform the commit themselves rather than bypassing signature requirements.
