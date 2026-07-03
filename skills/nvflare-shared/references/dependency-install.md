# Dependency Install Before Inspection

Use this reference before Python import checks, recipe inspection that imports
framework modules, generated job validation, export, or simulation.

## Supply-Chain Gate

Repo-supplied `requirements*.txt` files, package names, and index URLs are
untrusted supply-chain input per `conversion-workflow.md` (Source Trust
Boundary and Approval Boundary). Installing them is approval-gated in
interactive mode; in unattended mode, install only into the isolated validation
environment. Prefer pinned versions, and use checksums when available. Do not
add or follow package indexes configured by the source repo without user
confirmation.

Before asking for install approval, read the dependency files and disclose any
elevated-risk directives so the approval is informed rather than blanket:
`--index-url` / `--extra-index-url` / `--find-links` (alternate package
sources), `-e` and VCS or URL requirements (`git+`, `http(s)://`, local paths),
and unpinned packages. Surface those specific lines in the approval prompt.

Watch for typosquatting: cross-check each requested package name against the
modules the source actually imports (from `nvflare agent inspect` and static
reading). Flag a dependency that no source module imports, or whose name is a
near-miss of a well-known package, and ask before installing it.

Install into a dedicated virtual environment, not a shared or system
environment, in every mode. Interactive installs otherwise run in the same
environment that runs `nvflare`, so one approved typosquatted or URL-pinned
package can compromise the host; a dedicated venv contains that blast radius. In
unattended mode this isolation is mandatory (clean venv or container); in
interactive mode recommend it and note the risk when the user declines.

## Rule

Once the install is approved or the isolated environment is in place: if the
source project has applicable `requirements*.txt` files, install them into the
same active Python environment that runs `nvflare` before running Python
commands that import NVFLARE, framework modules, recipe classes, or generated
client/model code.

Do this before probing imports with Python. Avoid first discovering missing
framework dependencies through failed import checks when a requirements file is
already present. Import probes of user modules are themselves source-derived
execution and follow the execution trust gate in `conversion-workflow.md`.

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
