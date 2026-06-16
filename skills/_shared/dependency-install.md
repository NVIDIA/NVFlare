# Dependency Install Before Inspection

Use this reference before Python import checks, recipe inspection that imports
framework modules, generated job validation, export, or simulation.

## Rule

If the source project has applicable `requirements*.txt` files, install them
into the same active Python environment that runs `nvflare` before running
Python commands that import NVFLARE, framework modules, recipe classes, or
generated client/model code.

Do this before probing imports with Python. Avoid first discovering missing
framework dependencies through failed import checks when a requirements file is
already present.

## Installer Choice

- Prefer `uv pip install -r <file>` when `uv` is available and the active
  virtual environment is the `nvflare` environment.
- If the `nvflare` interpreter is not the active environment, use
  `uv pip install --python <python> -r <file>` with the Python interpreter
  behind `nvflare`.
- If `uv` is unavailable, use `<python> -m pip install -r <file>`.
- Do not use `uv pip install --system` when `nvflare` is installed in a virtual
  environment; it can install dependencies into the wrong Python.

When an import still fails after installation, verify which interpreter
received the packages before rerunning the failed check.

## Blockers To Report

Report dependency installation as blocked when:

- no applicable dependency file exists and required imports are missing;
- the install command fails;
- required network, package index, system library, or accelerator resources are
  unavailable;
- the user has not approved an install that requires approval in the current
  environment.

Keep dependency install, cleanup, export, and simulation as separate commands.
Do not combine destructive cleanup and execution in one command.
