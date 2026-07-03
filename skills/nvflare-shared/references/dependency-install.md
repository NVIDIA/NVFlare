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
and unpinned packages. Describe each such directive in the approval prompt by
directive type, package name, source host, and the risk it carries — never by
quoting the raw line. Requirement lines can embed credentials: strip userinfo
(`user:token@`), query strings, and fragments from every URL before disclosure,
per the redaction rule in `conversion-workflow.md`. Note that a credential was
present and redacted; do not reproduce its value in prompts, reports, or logs.

Watch for typosquatting: cross-check each requested package name against the
modules the source actually imports (from `nvflare agent inspect` and static
reading). Flag a dependency that no source module imports, or whose name is a
near-miss of a well-known package, and ask before installing it.

Install into a dedicated validation environment (a clean venv or container),
not a shared or system environment, in every mode. Installing source
requirements into a shared host environment means one approved typosquatted or
URL-pinned package can compromise the host; a dedicated environment contains
that blast radius. In unattended mode this isolation is mandatory; in
interactive mode recommend it and note the risk when the user declines. When
the user declines isolation and directs the install into a shared environment,
that shared environment becomes the validation environment for the rules below.

## Rule

Once the install is approved or the isolated environment is in place: if the
source project has applicable `requirements*.txt` files, install them into the
dedicated validation environment before running Python commands that import
NVFLARE, framework modules, recipe classes, or generated client/model code.
Install `nvflare` into that same environment if it is not already present, and
run `nvflare` commands, import probes, export, and simulation from it — not
from a separate shared or system environment.

Do this before probing imports with Python. Avoid first discovering missing
framework dependencies through failed import checks when a requirements file is
already present. Import probes of user modules are themselves source-derived
execution and follow the execution trust gate in `conversion-workflow.md`.

## Installer Choice

- Prefer `uv pip install -r <file>` when `uv` is available and the dedicated
  validation environment is the active virtual environment.
- If the validation environment is not active, use
  `uv pip install --python <python> -r <file>` with the validation
  environment's Python interpreter.
- If `uv` is unavailable, use `<python> -m pip install -r <file>` with the
  validation environment's interpreter.
- Do not use `uv pip install --system`; it installs into the system Python and
  defeats the isolation requirement.

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
