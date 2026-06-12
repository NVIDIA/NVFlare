# Shared NVFLARE Job Lifecycle Guidance

Use this reference for framework-agnostic conversion, validation, export, and
POC handoff behavior. Framework skills should combine it with framework-specific
model, training-loop, and aggregation guidance.

## Natural Request Parsing

Users may describe work in product terms, for example: "Here is my training
code. Convert it to FLARE FL code, run it with 3 simulated sites on this
dataset, split the dataset evenly, use FedAvg, and train for 3 rounds."

Extract recipe intent, site count, rounds, dataset path, split policy, training
arguments, validation intent, and approval boundaries before asking follow-up
questions. Ask only when a missing value changes the generated job or runtime
behavior.

## Conversion Workflow Contract

- Run `nvflare agent inspect <path> --format json` before editing.
- Use the user-requested target location for generated FLARE job source.
- Keep edits scoped to training, model, job, and small config files.
- Preserve user data paths and require user confirmation before changing them.
- Translate natural user requests into concrete recipe, site-count, dataset,
  split, training, validation, and export settings.
- Use the standard generated source names `client.py`, `job.py`, and `model.py`
  when model code is copied or wrapped. Keep original source files as
  references unless the user explicitly asks to rewrite them.
- Ask before changing private data paths, replacing dataset access, using
  non-fixture data for validation, or submitting to POC, production, or
  startup-kit based runtimes.
- Do not generate Python solely to wrap `nvflare` CLI commands or scrape human
  CLI output.
- Do not require `rg` to be installed. Use `rg` when available; otherwise use
  `nvflare agent inspect`, `find`, `git ls-files`, or a small Python search.

## Generated Job Layout

Use the current project or job source root as the default generated FLARE source
location. Add or update standard source files such as `client.py`, `job.py`,
`model.py`, `requirements.txt`, and small config files beside the existing
training files unless the user explicitly asks for another target directory.
Do not create an extra wrapper folder such as `nvflare_jobs/<job_name>/` by
default. Preserve original training files such as `train.py` as references
instead of renaming or overwriting them unless the user explicitly asks for an
in-place rewrite.

Do not put exported jobs, simulation workspaces, generated model artifacts, or
temporary vocab/cache files in the original source root by default. Use explicit
runtime locations under `/tmp/nvflare/` unless the user provides another path:

- exported job config: `/tmp/nvflare/job_config/<job_name>/`;
- simulation workspace: `/tmp/nvflare/workspaces/<job_name>/`;
- generated validation outputs or evaluation records:
  `/tmp/nvflare/results/<job_name>/`.

## Local Validation

- Use `python job.py` for local recipe or SimEnv validation when supported.
- Prefer synthetic data flags or small fixtures when the original dataset is
  unavailable.
- Before import checks, export, or simulation, install applicable
  source-provided `requirements*.txt` files into the same active Python
  environment. Prefer `uv pip install -r <file>` when `uv` is available;
  otherwise use `python -m pip install -r <file>` or the repository's documented
  equivalent. If an import still fails, verify which environment received the
  install before rerunning the failed check.
- Once an applicable requirements file is found and framework imports are
  missing, install it before any Python command that imports framework-specific
  NVFLARE modules such as `nvflare.app_opt.pt.*`, recipe classes, or generated
  client/model code.
- Treat missing dependencies as blockers only when no applicable dependency file
  exists, install fails, system/GPU resources are unavailable, or required
  approval/network access is unavailable.
- Keep validation commands single-purpose. Run cleanup, dependency install,
  export, and simulation as separate commands; do not combine destructive
  cleanup and execution such as `rm -rf <workspace> && python job.py`.
- After successful simulation, inspect the server workspace metrics directory.
  Standard aggregation recipes write
  `/tmp/nvflare/workspaces/<job_name>/server/simulate_job/metrics/metrics_summary.json`
  for final/best aggregate metrics and `round_metrics.jsonl` for per-round or
  per-site evidence when present. Report both categories and paths separately;
  if either file is absent, say so and fall back to bounded stdout/stderr or
  server logs.
- Report command, status, result directory, and dependency or data blockers.

## Preflight Before Full Simulation

Before `python job.py`, run cheap checks first:

- Compile generated Python files.
- Construct the recipe and export to a temporary directory.
- Inspect exported server config for model path and constructor args.
- Verify `app/custom` includes files imported by server and client code; add
  explicit file packaging only when export misses a required file.
- Check server/client model `state_dict` compatibility.

Use preflight results to fix packaging, config, or model-state issues before
spending time on full simulation.

## Export

- Use `python job.py --export --export-dir <dir>` to export a generated job.
  These are NVFLARE job system arguments across recipes, algorithms, and
  frameworks. Do not declare them as generated job-local arguments.
- If a generated `job.py` defines local command-line options, its local parser
  must tolerate NVFLARE system arguments such as `--export` and `--export-dir`.
  With `argparse`, use `parse_known_args()` or an equivalent approach. Do not
  add local `--export` or `--export-dir` arguments, and do not let local parsing
  reject or consume them before the NVFLARE job/export layer handles export.
  Treat this as a generation-time requirement; validation should confirm the
  behavior rather than discovering it through a failed export.
- Default `<dir>` to `/tmp/nvflare/job_config/<job_name>` unless the user
  provides an export directory.
- If writing explicit Job API code without a recipe execution helper, call
  `job.export_job(<dir>)` directly when needed.
- Inspect the exported folder for server/client app folders and expected config
  files before recommending submission.

## Validation Evidence

Before calling a generated job correct, report:

- selected recipe and the `nvflare recipe show` command used to inspect it;
- changed files and why they were changed;
- local validation command and pass/fail status;
- export command, export directory, and exported folder inspection result when
  export is in scope;
- metric values or a clear explanation that metrics were unavailable;
- exact evidence paths: simulation workspace, generated result files,
  server-side metrics artifacts, logs, and global-model artifacts. Include both
  `metrics_summary.json` and `round_metrics.jsonl` paths when present;
- unresolved blockers such as unavailable data, missing dependencies, or
  required user approval.

If `python job.py` cannot run, the conversion may still be saved as a draft, but
report it as unvalidated and name the concrete blocker.

## POC Handoff

Users may approve a runtime handoff after simulation, for example: "Simulation
looks good. Start POC and submit the exported job" or "I have a POC workspace
here; submit the job to it."

Treat this as explicit POC approval. Validate the exported job folder first,
then use the supplied POC workspace or start POC as requested, submit the job,
and wait or monitor if requested.

Report the POC workspace, submitted job folder, job ID, final status or current
status, command evidence, and log/result paths. For POC or production runs in a
terminal state, use `nvflare job download <job_id> -o <dir> --format json` to
download consumable result artifacts. Read `data.artifacts.global_model`,
`data.artifacts.metrics_summary`, and `data.artifacts.round_metrics` from the
JSON response when present instead of constructing server paths manually.
`round_metrics` is optional, and missing artifact categories should be reported
from `data.missing_artifacts` without treating a successful download as failed.
If the POC run fails, record the failure as evaluation evidence.

## Approval Boundary

POC or production submission is outside a conversion skill's default action.
Ask for explicit user approval before using submit or runtime-start commands.

## Evaluation Records

When a generated job does not run as expected, keep the failure as evaluation
evidence instead of treating it as a one-off note. Record the user request,
selected recipe, files changed, validation command, failure output summary,
root-cause hypothesis, and follow-up fix or blocker.

If the failure represents a repeatable skill gap, add or update an eval case,
benchmark gap, fixture, or reference note so future skill runs are tested
against the same scenario.
