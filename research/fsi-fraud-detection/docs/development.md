# Development Guide

This document covers setting up a development environment, running tests, and
configuring custom package registries. For end-user quick start instructions,
see the main [README](../README.md).

## Development Setup

After cloning, create a virtual environment and install all dependency groups
(including dev tools like `pytest`, `ruff`, `mypy`):

```bash
# Create a new virtual environment. The --seed flag installs pip, setuptools, and wheel for you.
uv venv --seed

# Install all dependencies including dev tools.
# The --frozen flag ensures the exact versions in the lockfile are used
# without any updates or checks for the latest versions.
uv sync --all-groups --frozen
```

## Running Tests

The test suite covers RNG distributions, static data loaders, anomaly
transformers, and end-to-end dataset generation (126 tests):

```bash
uv run pytest tests/ -q
```

## Linting & Type Checking

```bash
# Lint
uv run ruff check .

# Type check
uv run mypy .
```

## Custom Package Index

If your organization uses a private package registry, configure it in
`pyproject.toml`:

```toml
[[tool.uv.index]]
name = "<your index name>"
url = "<your index URL>"
explicit = true
```

See the [uv index documentation](https://docs.astral.sh/uv/concepts/projects/dependencies/#index)
for full configuration options.
