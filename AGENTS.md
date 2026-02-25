# Agent Workflow Notes

- Before every `git push`, run style checks on touched Python files:
  - `python -m black --check <touched_python_files>`
  - `python -m isort --check <touched_python_files>`
