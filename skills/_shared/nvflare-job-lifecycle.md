# Shared NVFLARE Job Lifecycle Guidance

Use this reference for framework-agnostic conversion, validation, export, and
POC handoff behavior. Framework skills should combine it with framework-specific
model, training-loop, and aggregation guidance.

Load the smaller shared references when the task reaches that phase:

- `dependency-install.md` before Python import or inspect commands;
- `runtime-output-guidance.md` before choosing generated source, export, or
  runtime workspace locations;
- `validation-evidence.md` before validation and final conversion acceptance;
- `metrics-and-artifact-reporting.md` before final metric or artifact reporting.

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

## Site Data Partitioning

When converting single-node training code to multiple simulated or federated
sites, preserve any existing user-provided site split. If no split exists and
the user asks the agent to create one, prefer a deterministic seeded shuffle and
use a stratified split when classification labels are available. Do not use a
simple stride or contiguous split as the default because it can create biased
site partitions from ordered data.

Report the split policy, seed, site count, and any reason stratification was not
used. Treat private data movement as part of the approval boundary: do not copy
private site data into generated artifacts unless the user explicitly asks.

## Source Of Truth Boundary

Use the active skill and its references for conversion workflow, safety rules,
client API or patch patterns, exchange format expectations, generated layout,
and evidence reporting. For current recipe names and parameters, use
`nvflare recipe list --format json` and
`nvflare recipe show <recipe-name> --format json` as the source of truth.

Only consult NVFLARE library source, such as `site-packages/nvflare/**`, when
the skill references and structured CLI metadata do not answer a specific
question. If source reading is necessary, name the unanswered question and
record it as a skill/reference gap so future runs can avoid the source dive.

## Generated Job Layout

Follow `runtime-output-guidance.md` for generated source layout, runtime
workspace placement, generated validation outputs, and export directory
defaults.

## Execution Environments

Recipe jobs run through an execution environment from `nvflare.recipe`, usually
`SimEnv` for local simulation and validation, `PocEnv` for local POC systems, or
`ProdEnv` for production systems. Default to `SimEnv` during conversion
validation. Use `PocEnv` or `ProdEnv` only after the approval boundary below.

For recipe-based generated jobs, build the recipe and call `recipe.execute(env)`
from `job.py`. For example:

```python
env = SimEnv(num_clients=num_clients, num_threads=num_clients, workspace_root=workspace_root)
recipe.execute(env)
```

This lets `python job.py` run local validation and lets NVFLARE recipe/job
system export arguments such as `--export` and `--export-dir` be handled by the
NVFLARE layer. Do not define generated job-local `--export` or `--export-dir`
arguments.

## Local Validation

- Use `python job.py` for local recipe or SimEnv validation when supported.
- Prefer synthetic data flags or small fixtures when the original dataset is
  unavailable.
- Before Python import checks, export, or simulation, follow
  `dependency-install.md`.
- Treat missing dependencies as blockers only when no applicable dependency file
  exists, install fails, system/GPU resources are unavailable, or required
  approval/network access is unavailable.
- Keep validation commands single-purpose. Run cleanup, dependency install,
  export, and simulation as separate commands; do not combine destructive
  cleanup and execution such as `rm -rf <workspace> && python job.py`.
- After successful simulation, follow `metrics-and-artifact-reporting.md`.

## Preflight Before Full Simulation

Follow `validation-evidence.md` for generic preflight checks. Framework skills
own framework-specific compatibility checks such as model state, data loading,
or metric serialization.

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
- Default `<dir>` according to `runtime-output-guidance.md` unless the user
  provides an export directory.
- If writing explicit Job API code without a recipe execution helper, call
  `job.export_job(<dir>)` directly when needed.
- Inspect the exported folder for server/client app folders and expected config
  files before recommending submission.

## Validation Evidence

Follow `validation-evidence.md` and `metrics-and-artifact-reporting.md`. If
`python job.py` cannot run, the conversion may still be saved as a draft, but
report it as unvalidated and name the concrete blocker.

## POC Handoff

Users may approve a runtime handoff after simulation, for example: "Simulation
looks good. Start POC and submit the exported job" or "I have a POC workspace
here; submit the job to it."

Treat this as explicit POC approval. Validate the exported job folder first,
then use the supplied POC workspace or start POC as requested, submit the job,
and wait or monitor if requested.

Report the POC workspace, submitted job folder, job ID, final status or current
status, command evidence, and log/result paths. For terminal POC or production
runs, follow `metrics-and-artifact-reporting.md` for downloaded artifact
handling. If the POC run fails, record the failure as evaluation evidence.

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
