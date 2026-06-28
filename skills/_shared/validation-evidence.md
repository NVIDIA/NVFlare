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

## Terminal Simulation Evidence

Do not report local simulation or job validation as successful while
`python job.py`, simulator, or another generated run command is still running.
If using background execution, wait for task completion, record the exit code,
then inspect server/client logs and generated artifacts before reporting
success.

Required success evidence is process exit code 0, terminal FL evidence such as
the server log reaching a Finished state, and metrics evidence such as
`metrics_summary.json` or a concrete explanation for why metrics are
unavailable. Progress messages, scheduled wakeups, and active processes are not
completion evidence.

If the run exceeds the allowed time, report it as blocked or timed out with the
current command status, log evidence, and artifact evidence. Do not describe a
timed-out or still-running simulation as done.

## Preflight Before Full Simulation

Before spending time on full simulation, run cheap checks when applicable:

- compile generated Python files;
- construct or instantiate the selected recipe;
- export to a temporary directory;
- inspect exported server/client app folders and expected config files;
- verify generated files required by server and client code are packaged;
- run local partition sanity checks when generated site splits or data
  partitions are introduced;
- run the framework-specific model compatibility check defined by the framework
  skill.

Use preflight results to fix packaging, config, or model-state issues before
running a full simulation.

## Evidence To Report

Before calling a generated job correct, report:

- selected recipe and the `nvflare recipe show` command used to inspect it;
- changed files and why they were changed;
- local validation command, process exit code, and terminal-state evidence;
- export command, export directory, and exported folder inspection result when
  export is in scope;
- metric values from metrics artifacts, or a clear explanation that metrics
  were unavailable;
- exact evidence paths for simulation workspace, generated result files,
  server-side metrics artifacts, server/client logs, and global-model artifacts
  when present;
- unresolved blockers such as unavailable data, missing dependencies, failed
  validation, or required user approval.

When a generated job does not run as expected, keep the failure as evaluation
evidence. Record the user request, selected recipe, files changed, validation
command, failure output summary, root-cause hypothesis, and follow-up fix or
blocker.
