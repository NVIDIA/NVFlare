# Validation Evidence Guidance

Use this reference before declaring a generated or exported NVFLARE job valid.

## Local Validation

- Use `python job.py` for local recipe or SimEnv validation when supported.
- Prefer synthetic data flags or small fixtures when the original dataset is
  unavailable.
- Keep validation commands single-purpose. Run dependency installation, cleanup,
  export, and simulation as separate commands.
- If validation cannot run, save the conversion as a draft and report the
  concrete blocker.

## Preflight Before Full Simulation

Before spending time on full simulation, run cheap checks when applicable:

- compile generated Python files;
- construct or instantiate the selected recipe;
- export to a temporary directory;
- inspect exported server/client app folders and expected config files;
- verify generated files required by server and client code are packaged;
- run the framework-specific model compatibility check defined by the framework
  skill.

Use preflight results to fix packaging, config, or model-state issues before
running a full simulation.

## Evidence To Report

Before calling a generated job correct, report:

- selected recipe and the `nvflare recipe show` command used to inspect it;
- changed files and why they were changed;
- local validation command and pass/fail status;
- export command, export directory, and exported folder inspection result when
  export is in scope;
- metric values or a clear explanation that metrics were unavailable;
- exact evidence paths for simulation workspace, generated result files,
  server-side metrics artifacts, logs, and global-model artifacts when present;
- unresolved blockers such as unavailable data, missing dependencies, failed
  validation, or required user approval.

When a generated job does not run as expected, keep the failure as evaluation
evidence. Record the user request, selected recipe, files changed, validation
command, failure output summary, root-cause hypothesis, and follow-up fix or
blocker.
