# Milestone 8 Detailed Implementation Proposal: Lifecycle Skills + Lightning Conversion

## Objective

Milestone 8 should move from a lifecycle-only wave to a controlled expansion
wave that proves the agent skill system can support:

1. the first customer lifecycle skills, and
2. a second framework conversion skill, `nvflare-convert-lightning`.

This gives the base skill set two real conversion paths before broader framework
expansion. PyTorch Lightning is close enough to PyTorch to reuse shared guidance,
but different enough to validate that the architecture is not overfit to one
PyTorch Client API pattern.

## Current Upstream Context

After rebasing onto `upstream/main`, the native agent command surface exists
under `nvflare/tool/agent/`:

- `nvflare agent info`
- `nvflare agent inspect`
- `nvflare agent doctor`
- `nvflare agent skills install`
- `nvflare agent skills list`

The latest `nvflare agent inspect` implementation already has an important
handoff point for Milestone 8:

- when `conversion_state == "exported_job"`, `_skill_selection()` recommends
  `nvflare-job-lifecycle`;
- when a framework is detected and the code is `not_converted`,
  `_skill_selection()` recommends the matching `nvflare-convert-*` skill;
- `FRAMEWORK_SKILLS` is the deterministic CLI routing map that must add
  Lightning when the Lightning skill lands.

This means Milestone 8 should not invent a new lifecycle command surface. The
CLI should remain deterministic evidence and routing support; the skills should
own procedural workflow guidance.

## Milestone 8 Scope Change

Rename the milestone from:

```text
Milestone 8: Customer Lifecycle Skill Wave
```

to:

```text
Milestone 8: Base Conversion Expansion and Customer Lifecycle Skill Wave
```

Milestone 8 deliverables become:

- `nvflare-convert-lightning`
- `nvflare-poc-workflow`
- `nvflare-job-lifecycle`

Milestone 9 should remove Lightning from its framework-conversion list and keep
the remaining broader framework skills:

- `nvflare-convert-tensorflow`
- `nvflare-convert-huggingface`
- `nvflare-convert-xgboost`
- `nvflare-convert-sklearn`
- `nvflare-convert-survival-analysis`

## Why Lightning Moves Into Milestone 8

Lightning is a good second base conversion skill because it stresses the design
without expanding the catalog too broadly.

Plain PyTorch conversion uses explicit Client API model exchange:

- receive `FLModel`;
- load `params` into a `torch.nn.Module`;
- train/evaluate;
- send `FLModel` with updated `params`, metrics, and metadata.

Lightning conversion should use the Lightning integration:

- `import nvflare.client.lightning as flare`;
- `flare.patch(trainer)`;
- Lightning `Trainer` owns fit/validate/test execution;
- `FLCallback` handles receive/send around trainer events;
- optional `ClientLogger` handles FLARE-backed tracking.

This validates that the skill system can support two different correct API
patterns:

- low-level framework exchange for PyTorch;
- framework-native trainer patching for Lightning.

## Implementation Split

Milestone 8 should be implemented in two separated tracks:

1. Skill implementation: files under repo-root `skills/`.
2. CLI/product implementation: deterministic support in `nvflare/tool/agent`,
   `nvflare recipe`, `nvflare job`, and future promoted helper commands.

The CLI must not become a hidden agent runtime, benchmark runner, or public skill
evaluator. It should provide inspectable facts, readiness checks, install/list
support, and stable JSON contracts that skills can rely on.

## Track A: Skill Implementation

### A1. Add `nvflare-convert-lightning`

Create:

```text
skills/nvflare-convert-lightning/
  SKILL.md
  references/
    lightning-detection.md
    lightning-conversion.md
    lightning-validation.md
    lightning-ddp-and-tracking.md
  evals/
    evals.json
    files/
```

The initial skill should stay concise. Detailed code patterns, examples, and
edge cases belong under `references/`.

### A2. Lightning Skill Trigger Contract

Positive triggers:

- user asks to convert PyTorch Lightning code to FLARE;
- project imports `pytorch_lightning` or `lightning.pytorch`;
- project defines `LightningModule`, `LightningDataModule`, or uses
  `Trainer`;
- user asks to federate a Lightning trainer, module, checkpoint, DDP job, or
  Lightning validation/evaluation workflow.

Negative triggers:

- plain `torch.nn.Module` and manual training loop without Lightning should
  route to `nvflare-convert-pytorch`;
- existing exported FLARE job should route to `nvflare-job-lifecycle`;
- failed or suspicious existing job should route to `nvflare-diagnose-job`;
- TensorFlow/Keras, XGBoost, sklearn, Hugging Face, and NumPy/custom loops stay
  out of scope unless no stronger skill exists.

### A3. Lightning Conversion Contract

The Lightning skill should guide agents to:

- inspect before editing with `nvflare agent inspect <path> --format json`;
- install project requirements before importing user code, using `uv pip` when
  available;
- identify the existing `LightningModule`, `LightningDataModule`, trainer
  construction, callbacks, checkpointing, metrics, and logger usage;
- preserve user data paths unless the user explicitly asks to change them;
- keep generated runtime workspaces outside the source tree by default;
- use the standard generated source names `client.py`, `job.py`, and `model.py`
  when new files are needed;
- avoid wrapping `nvflare` CLI commands in generated Python;
- validate locally before claiming conversion complete;
- report metrics/artifact paths only when evidence exists.

The generated Lightning client path should prefer:

```python
import nvflare.client.lightning as flare

flare.init()  # optional: only when get_site_name() or other pre-patch context is needed
trainer = Trainer(...)
flare.patch(trainer)

while flare.is_running():
    # Optional: call receive() only when round/site/task metadata is needed.
    # Do not pass input_model to trainer; the patched trainer loads the
    # global model internally.
    input_model = flare.receive()
    trainer.validate(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)  # when test evidence is requested/available
```

Do not force manual `FLModel` construction for normal Lightning training. The
Lightning integration owns model load/send through callbacks after
`flare.patch(trainer)`. An explicit `flare.receive()` in the patched loop is
optional metadata/task access; the skill must not make it a second model-load
path, generate an additional manual receive/send exchange, or pass the returned
`FLModel` into the `Trainer`.

For DDP, use the repository's rank-synchronized loop shape:

```python
flare.patch(trainer)

while True:
    is_running = flare.is_running()
    is_running = trainer.strategy.broadcast(is_running, src=0)
    if not is_running:
        break

    input_model = flare.receive()  # optional, only when round/task metadata is needed
    trainer.validate(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)  # when test evidence is requested/available
```

### A4. PyTorch Recipe Reuse

Lightning is a PyTorch training framework, not a separate FLARE recipe family.
The Lightning skill should reuse the same PyTorch recipe discovery and
selection rules as `nvflare-convert-pytorch`:

```bash
nvflare recipe list --framework pytorch --format json
nvflare recipe show <recipe-name> --format json
```

The difference is the client/training integration path:

- `nvflare-convert-pytorch` owns manual Client API / `FLModel` exchange for
  plain PyTorch loops.
- `nvflare-convert-lightning` owns Lightning `Trainer`, `LightningModule`,
  `LightningDataModule`, callbacks, loggers, checkpoints, and
  `flare.patch(trainer)`.

Initial supported Lightning workflows should still be selected from PyTorch
recipe intent:

- standard horizontal training with the PyTorch FedAvg recipe;
- evaluation-only with the PyTorch FedEval recipe;
- single-GPU or ordinary single-process training may use in-process execution;
- PyTorch distributed training, including Lightning DDP and plain PyTorch DDP,
  should use external process launch such as `launch_external_process=True`
  instead of trying to run distributed workers inside an in-process executor;
- optional tracking handoff through `add_experiment_tracking` or
  `nvflare.app_opt.lightning.loggers.ClientLogger` only when the user's task
  asks for tracking or the existing code already uses it.

Recipe selection must be driven by user workflow intent, not by the mere
existence of Lightning. Do not create a Lightning-only recipe-selection
reference unless future FLARE releases add Lightning-specific recipes.

This external-process rule is PyTorch-family guidance shared by
`nvflare-convert-pytorch` and `nvflare-convert-lightning`. TensorFlow
multi-GPU/distributed behavior is not part of Milestone 8; future TensorFlow
guidance should verify the correct execution model before reusing this rule,
though separate process launch is the likely starting point for distributed
training scripts.

### A5. Lightning References

`references/lightning-detection.md` should cover:

- recognizing `LightningModule`, `LightningDataModule`, and `Trainer`;
- distinguishing Lightning projects from plain PyTorch modules and manual
  training loops;
- preserving a negative handoff to `nvflare-convert-pytorch` when no Lightning
  evidence exists.

`references/lightning-conversion.md` should cover:

- where to call `flare.patch(trainer)`;
- using `flare.init()` and `flare.get_site_name()` only when needed;
- using `flare.receive()` in the patched loop only for FL task progression,
  round/site logging, or task metadata, not for manual model loading;
- avoiding extra manual `FLModel` send/receive when the patched trainer already
  owns model exchange;
- preserving callbacks/loggers/checkpoint callbacks;
- validating global model before local training when server-side model
  selection needs validation metrics;
- avoiding repeated expensive setup inside the FL round loop.

`references/lightning-validation.md` should cover:

- dependency installation first;
- local `python job.py` SimEnv validation;
- export validation;
- metric evidence;
- runtime output locations;
- known SimEnv limitations.

`references/lightning-ddp-and-tracking.md` should cover:

- DDP rank behavior;
- the `while True` DDP loop that broadcasts `flare.is_running()` from rank 0
  before receive/validate/fit;
- GPU/CPU fallback;
- TensorBoard/MLflow logger preservation;
- `ClientLogger` limitations.

### A6. Lifecycle Skills

Milestone 8 lifecycle skills should be implemented separately from Lightning.
These are not conversion skills and they do not add new public CLI commands.
They are workflow playbooks that teach the agent when to use the existing
NVFLARE CLI and how to report evidence safely.

Add two lifecycle skill directories:

```text
skills/nvflare-poc-workflow/
  SKILL.md
  references/
    poc-lifecycle.md
    poc-readiness.md
    poc-cleanup-and-restore.md
  evals/
    evals.json
    files/

skills/nvflare-job-lifecycle/
  SKILL.md
  references/
    job-validation.md
    submit-monitor-download.md
  evals/
    evals.json
    files/
```

#### `nvflare-poc-workflow`

Purpose:

- bridge from generated-job simulation (`python job.py` with SimEnv) to a local
  FLARE system with separate server and client processes on localhost;
- start that small local FLARE system safely;
- verify server/client readiness before job submission;
- let the user experience job lifecycle operations, such as submit, monitor,
  and download, in a local FLARE system;
- stop or clean up POC state when the user asks;
- recover orphaned local FLARE server/client processes when a previous POC
  workspace was overwritten or can no longer track the processes it started;
- prevent accidental overwrite or cleanup of an active POC workspace.

Primary users:

- data scientists who have validated a generated job in simulation and want the
  next local system-level step;
- developers testing examples or generated jobs against real local FLARE
  processes rather than SimEnv;
- agents that have completed conversion and need a running local system before
  handing off to job submission.

Positive trigger prompts:

- "I ran `python job.py`; what is the next local system test?"
- "Run this exported job with separate local server and client processes."
- "Show me how this job behaves in a local FLARE system instead of SimEnv."
- "I want to submit, monitor, and download this job in a local FLARE system."
- "Start a local server and clients for this exported job."
- "Check whether my local FLARE system is running."
- "Stop the local FLARE system."
- "Clean up the local FLARE system from the last run."
- "The POC workspace was overwritten; find and stop leftover FLARE processes."
- "I cannot restart POC because old server/client processes are still running."
- "Find orphaned local FLARE processes and ask me before killing them."
- "I want to start a fresh local FLARE system, but one may already be running."
- "Can I run another local POC in a different workspace?"
- "Clean up this POC workspace, but first check whether jobs are still running."
- "Create a local POC workspace and start FLARE."

Trigger from evidence:

- user asks for the next step after `python job.py` or SimEnv simulation;
- user asks for separate local server/client processes, localhost
  system-level testing, or local job lifecycle operations;
- user explicitly asks for POC prepare/start/stop/cleanup;
- user reports that a previous POC workspace was deleted, overwritten, or cannot
  track server/client processes anymore;
- user asks to start a new POC while an existing POC may already be running;
- user asks to clean or overwrite a POC workspace that may still have active
  jobs, server, or client processes;
- `nvflare agent doctor` reports no active/running local startup kit and the
  user wants to submit against a local system;
- bounded process inspection finds likely local FLARE POC server/client
  processes after normal stop/status commands cannot account for them;
- a conversion skill has exported a job and the user asks to run it in a local
  FLARE system, not just SimEnv.

Non-triggers:

- first-time NVFLARE install or broken CLI: out of lifecycle-skill scope; use
  ordinary product docs and general agent setup guidance before this skill runs;
- job validation/submit/monitor/download after a system is already running:
  route to `nvflare-job-lifecycle`;
- remote/startup-kit use: route to `nvflare-job-lifecycle` for identity/config
  checks, not local POC setup;
- cloud/Kubernetes/multi-machine provisioning: out of scope for this skill.

What we add:

- a concise `SKILL.md` that defines POC as local-only and requires explicit
  user confirmation before destructive cleanup, overwriting an active workspace, or
  starting a second local POC alongside an existing one. It should not inline
  the detailed cleanup/orphan-process procedure;
- `references/poc-lifecycle.md` with prepare/start/status/stop/cleanup
  walkthroughs using current NVFLARE CLI commands;
- `references/poc-readiness.md` with server/client readiness checks through
  `nvflare agent doctor --online --format json` and existing status commands;
- `references/poc-cleanup-and-restore.md` with safe cleanup boundaries,
  workspace reporting, restoration guidance where supported, and orphan-process
  cleanup rules.

If Milestone 8 needs a smaller first delivery, the bounded orphan-process
recovery portion can split into a follow-up. The core POC handoff remains
prepare/start/status/stop/cleanup for known workspaces plus readiness checks
before job submission.

Context-loading rule:

- The main `SKILL.md` should contain only routing, safety gates, and the minimum
  lifecycle map. It should load `references/poc-lifecycle.md` for
  prepare/start/status, `references/poc-readiness.md` only for readiness/status
  verification, and `references/poc-cleanup-and-restore.md` only when the user
  asks to stop, clean, overwrite, recover orphaned processes, or when evidence
  shows an active/conflicting workspace.
- Conversion skills should not load POC references directly. They should
  mention the POC handoff briefly and route to `nvflare-poc-workflow` only when
  the user asks for the next local system-level step after SimEnv/export.
- This is a shared lifecycle skill for any exported FLARE job, independent of
  framework or recipe. PyTorch, Lightning, TensorFlow, XGBoost, sklearn, and
  future conversion skills should all use the same handoff pattern instead of
  carrying framework-local copies of POC lifecycle guidance.
- `nvflare-job-lifecycle` should not load POC cleanup guidance unless its
  submit/monitor path discovers that the local POC system itself must be
  stopped, cleaned, or recovered.

Expected workflow:

1. Confirm whether the user wants prepare, start, status, stop, cleanup, or
   orphan-process recovery, or handoff to job submission.
2. Run `nvflare agent doctor --format json` to understand local readiness.
3. Before start or cleanup, inspect whether the target workspace already has
   running server/client processes, active jobs, occupied ports, or startup-kit
   state.
4. If starting a new POC, identify target workspace, site count, and whether an
   existing workspace may be reused. If another local POC is already running,
   report its workspace/process/port evidence and ask whether to stop it, reuse
   it, or start a separate POC in a different workspace.
5. If cleaning up a workspace with active jobs, report the active job evidence
   and ask whether to wait, stop the jobs/system, or abort cleanup.
6. If stopping or cleaning up, run the normal POC CLI stop/status/restore path
   first.
7. If the workspace cannot track prior processes, perform bounded process
   discovery for local FLARE server/client processes. Report the exact command
   lines, PIDs, cwd/port hints when available, and why they look related to the
   POC workspace.
8. Ask for explicit user confirmation before killing any process. Prefer targeted PIDs over broad
   name-based termination.
9. Run the appropriate POC CLI commands.
10. Verify readiness with `nvflare agent doctor --online --format json` when a
   system is expected to be running.
11. Report server/client status, workspace path, active startup kit context, and
   next handoff.
12. If the user wants to submit a job, hand off to `nvflare-job-lifecycle`.

Evidence to report:

- POC workspace path;
- server/client process or status summary;
- startup kit/config selected by the CLI;
- connected client count when available;
- active job evidence before cleanup;
- active workspace/port/process evidence when another POC is already running;
- doctor online summary when available;
- cleanup actions performed or intentionally skipped;
- orphan-process evidence, user-confirmed PIDs killed, and processes intentionally
  left alone.

#### `nvflare-job-lifecycle`

Purpose:

- operate on an existing or exported FLARE job;
- validate before submit;
- submit, monitor, inspect, download, and summarize results;
- enforce startup-kit identity/config checks.

Primary users:

- data scientists with an exported FLARE job folder;
- users who want to submit a generated job to a local or remote FLARE system;
- agents that receive `nvflare agent inspect` output with
  `conversion_state == "exported_job"`;
- operators who need bounded job status, logs, stats, or downloaded outputs.

Positive trigger prompts:

- "Submit this exported FLARE job."
- "Validate this job before I submit it."
- "Monitor job 123 until it finishes."
- "Download the result for the last job."
- "Show me logs/stats for this failed job."
- "Run this job on my POC system."
- "Submit this job to the remote FLARE system."

Trigger from evidence:

- `nvflare agent inspect <path> --format json` reports `exported_job`;
- the user points to a job folder containing FLARE job config/app structure;
- the user asks to submit/monitor/download job results;
- a conversion skill has exported a job and the user asks to run it outside
  SimEnv;
- `nvflare-poc-workflow` reports a running local system and the next requested
  action is job submission or monitoring.

Non-triggers:

- user gives raw training code and asks for conversion: route to a framework
  conversion skill;
- user asks to prepare/start/stop a local POC system: route to
  `nvflare-poc-workflow`;
- user asks to install or repair NVFLARE locally: out of lifecycle-skill scope;
  use ordinary product docs and general agent setup guidance;
- deep root-cause analysis of a failed job with logs already collected: route
  to `nvflare-diagnose-job` when diagnosis is the lead task.

What we add:

- a concise `SKILL.md` with target/context checks and a validation-first
  workflow;
- `references/job-validation.md` with read-only job structure checks, exported
  manifest/fingerprint handling when present, required files, and common
  packaging mistakes;
- `references/submit-monitor-download.md` with current `nvflare job` command
  usage, status polling guidance, bounded log/stat collection, and result
  download. It should include a short result-summary section that reports
  metrics and artifact paths only from observed outputs.

`references/job-validation.md` should validate:

- Job root resolution:
  - confirm the supplied path exists and is a directory or supported job
    archive/input accepted by existing `nvflare job` commands;
  - run `nvflare agent inspect <path> --format json` and use
    `conversion_state`, `target_type`, `job.exported_job_markers`, and findings
    as the first evidence source;
  - if the user gives a parent directory, identify whether there is exactly one
    valid child job; if multiple valid jobs exist, ask which one to use instead
    of guessing.
- Required job markers:
  - find one `meta.*` file recognized by FLARE config formats, such as
    `meta.json`, `meta.conf`, `meta.yml`, `meta.yaml`, and template/default
    variants where the CLI supports them;
  - find server/client config files in app directories named by `deploy_map`
    when possible, such as `app/config/config_fed_server.*`,
    `app/config/config_fed_client.*`, `app_server/config/config_fed_server.*`,
    and site app equivalents;
  - report missing `meta`, missing `deploy_map`, missing server config, or
    missing client config as blockers before submit.
- Config parseability and shape:
  - parse FLARE config files with structured parsers or existing NVFLARE helper
    paths, not ad hoc text scraping;
  - verify `deploy_map` points to app directories that exist;
  - verify app directories assigned to `server` have server config and app
    directories assigned to site/client roles have client config;
  - warn when `min_clients`, deploy-map sites, or expected client count look
    inconsistent.
- Referenced files and packaging:
  - statically check that obvious local Python scripts, custom modules, config
    includes, and relative paths referenced from configs exist under the job
    bundle;
  - flag absolute user-machine paths, references outside the job bundle, and
    likely private data paths as warnings or blockers depending on submit target;
  - flag symlinked job content and sensitive file names/extensions using the
    same no-follow/redaction posture as `nvflare agent inspect`.
- Launcher/resource hints:
  - summarize `resource_spec` and `launcher_spec` when present;
  - for Docker/K8s launcher hints, check that required fields such as image
    references are present, but do not build images or validate remote cluster
    access in this reference.
- Export metadata:
  - if `_export_manifest.json`, fingerprint metadata, or future export receipts
    are present, verify that listed files still exist and report mismatches;
  - if absent, continue with structural validation and note that provenance
    metadata is unavailable.

Validation result categories:

- `blocker`: missing meta/config, unparsable config, unresolved deploy-map app,
  missing referenced source file required to launch, dangerous path traversal, or
  a submit target mismatch that is very likely to fail.
- `warning`: missing optional provenance metadata, ambiguous parent directory,
  possible absolute data path, resource hints that may not match the target, or
  optional metrics/artifact paths not yet present.
- `ok`: the job shape is submit-ready based on static evidence.

What `job-validation.md` must not do:

- import or execute user training code;
- run training, simulation, or submit;
- select recipes or rewrite `client.py`/`job.py`;
- decide model quality, metric correctness, or convergence;
- perform deep root-cause analysis of failed runtime logs. Route that to
  `nvflare-diagnose-job` when diagnosis becomes the lead task.

`references/submit-monitor-download.md` should validate target readiness before
any submit:

- Runtime target availability:
  - determine the runtime target the user intends to use;
  - run `nvflare agent doctor --online --format json` or the current equivalent
    read-only status command before submit/monitor/download;
  - if no local POC is running and the user intended local execution, stop and
    route to `nvflare-poc-workflow` rather than trying to submit;
  - if the target server is unreachable, report the endpoint evidence and do
    not submit.
- Startup-kit and active context:
  - identify the active startup kit/config path used by the CLI;
  - if the startup-kit path does not exist, cannot be read, or is not the one
    the user intended, stop and report the mismatch;
  - if multiple possible kits are present, ask which one to use rather than
    guessing;
  - do not rewrite startup-kit configuration as part of submit/monitor/download.
- Role, identity, and study authorization:
  - summarize user/admin/site identity from the active context without printing
    secrets;
  - verify that the active role is appropriate for the requested operation:
    job submit/monitor/download should use an admin/user context, not a site
    client-only context;
  - if the user names a study, verify that the active identity/config can target
    that study before submit;
  - if the active kit lacks the right role or study access, report a blocker and
    ask the user to activate the correct kit or choose a different target.
- Submit behavior:
  - only submit after `job-validation.md` has no blockers;
  - record the exact job path, target system, study argument, submit command,
    returned job ID, and status.
- Monitor/log/stat behavior:
  - poll status with bounded intervals and a clear stop condition;
  - collect bounded logs/stats only when requested, when needed for final
    evidence, or when status becomes failed/stalled;
  - if the job fails or stalls and the user asks why, hand off to
    `nvflare-diagnose-job`.
- Download behavior:
  - download only after the job reaches a terminal state or when the user
    explicitly requests partial outputs;
  - report download directory, returned artifact paths, and whether expected
    model/metrics/log files were present.
- Result summary after download:
  - summarize only observed evidence from command output, downloaded result
    directories, logs/stats, or files explicitly produced by the job;
  - report final/best metrics, round or per-site metrics, model/checkpoint
    paths, logs/stats paths, and missing evidence when present;
  - if a conversion skill already reported local SimEnv metrics, keep that
    local-validation evidence separate from submitted-job or downloaded-result
    evidence;
  - do not generate metrics, infer metrics from the agent's own claims, or
    duplicate framework-specific metric-generation guidance.

Preflight result categories:

- `blocker`: target FLARE system not running/reachable, wrong or missing
  startup kit, active role cannot submit/monitor/download, requested study is
  not authorized by the active context, or job validation has blockers.
- `warning`: target is reachable but status is degraded, study context was not
  specified, optional log/stat commands are unavailable, or download destination
  already exists and requires user choice.
- `ok`: target context is clear and the requested lifecycle action can proceed.

Command routing:

The `--format` flag is a global CLI output flag. The skill should verify each
job subcommand's supported output modes with `nvflare job <subcommand>
--schema` before emitting machine-readable commands. For example, `monitor`
advertises `json` and `jsonl` in schema even though its subparser help only
lists monitor-local options.

| User intent | Preferred command | Notes and boundaries |
| --- | --- | --- |
| Submit an exported job | `nvflare job submit -j <exported_job_root> [--study <name>] [--submit-token <token>] --format json` | Requires `job-validation.md` with no blockers. `exported_job_root` is the exact submit-ready directory produced by `python job.py --export --export-dir <exported_job_root>`. |
| Recover/lookup a submitted job from a retry token | `nvflare job list --submit-token <token> [--study <name>] --format json` | Use when submit may have succeeded but the agent lost the job ID. Do not resubmit until lookup is checked. |
| List recent jobs | `nvflare job list [--study <name>] [-m <num>] --format json` | Read-only; useful before monitor/download when the user does not know the job ID. |
| Inspect one job's metadata/status | `nvflare job meta <job_id> [--study <name>] --format json` | Read-only status and metadata. Prefer before logs/download when job state is unclear. |
| Wait for terminal status | `nvflare job wait <job_id> [--study <name>] [--timeout <sec>] [--interval <sec>] --format json` | Use for non-streaming wait. Bound timeout unless the user asks to wait indefinitely. |
| Monitor progress | `nvflare job monitor <job_id> [--study <name>] [--timeout <sec>] [--interval <sec>] --format jsonl` | Use when streaming progress is useful. Keep polling bounded and summarize terminal status. |
| Collect running stats | `nvflare job stats <job_id> [--study <name>] [--site server\|<name>\|all] --format json` | Read-only; collect bounded stats when requested or needed for evidence. |
| Collect logs | `nvflare job logs <job_id> [--study <name>] [--site server\|<name>\|all] [--tail <N>] [--since <timestamp>] [--max-bytes <N>] --format json` | Read-only server-side log retrieval. Use bounds by default; do not shell into sites. |
| Download results | `nvflare job download <job_id> [--study <name>] [-o <dir>] [--force] --format json` | Prefer after terminal state. `--force` requires explicit user choice if destination exists. |
| Abort a running job | `nvflare job abort <job_id> [--study <name>] [--force] --format json` | Mutating. Run only when explicitly requested and after restating target, job ID, and expected effect. |
| Delete a job | `nvflare job delete <job_id> [--study <name>] [--force] --format json` | Mutating/destructive. Run only when explicitly requested and after restating target, job ID, and expected effect. |
| Clone a job | `nvflare job clone <job_id> [--study <name>] --format json` | Mutating or state-creating depending on server behavior. Ask before use unless the user explicitly requested clone. |

The reference should point to `nvflare job <subcommand> --schema` or the CLI
docs for full option details instead of duplicating the full CLI manual.
Preferred agent usage should request JSON/JSONL output where the command schema
supports it, so downstream evidence is structured.

Expected workflow:

1. Inspect the supplied job path with `nvflare agent inspect <path>
   --format json`.
2. Validate job shape and exported metadata when present.
3. Run `nvflare agent doctor --format json`; use `--online` before submit or
   monitoring when a running system is required.
4. Identify the target runtime context from CLI/startup-kit evidence.
5. Submit or monitor using existing `nvflare job` commands.
6. Download results/logs/stats when requested or required for final evidence.
7. Report job ID, status, command evidence, metrics found, artifact paths, and
   unresolved blockers.

Evidence to report:

- inspected job path and classification;
- validation result and missing/invalid files;
- startup-kit identity/config context when relevant;
- job ID and status transitions;
- logs/stats/downloaded result paths;
- metric/artifact paths and values when present;
- limitations, such as SimEnv not proving remote auth/network behavior.

The `nvflare-job-lifecycle` skill should explicitly match the existing CLI
handoff from `nvflare agent inspect`: exported jobs already recommend this skill.

### A7. Shared Guidance Refactor

Avoid duplicating content between PyTorch and Lightning.

Shared files should own cross-framework behavior, while family-shared files own
behavior reused by related framework skills only:

```text
skills/_shared/
  nvflare-job-lifecycle.md
  dependency-install.md
  runtime-output-guidance.md
  validation-evidence.md
  metrics-and-artifact-reporting.md
  pytorch-model-exchange.md
```

`pytorch-model-exchange.md` is not global guidance. Only
`nvflare-convert-pytorch` and `nvflare-convert-lightning` should load it. It
should cover state-dict/tensor exchange rules common to the PyTorch family:

- `state_dict`;
- `FLModel.params`;
- preserving PyTorch tensors rather than converting model parameters to NumPy;
- loading received parameters with `load_state_dict(...)`;
- keeping metrics JSON/scalar-friendly;
- separating checkpoint files from FL parameter exchange.

PyTorch-specific files should keep only plain-PyTorch details:

- manual Client API receive/send;
- PyTorch dataloaders, optimizers, and training loops;
- PyTorch recipe references.

Lightning-specific files should keep only Lightning-specific details:

- `LightningModule`;
- `Trainer`;
- `LightningDataModule`;
- `flare.patch`;
- callbacks/loggers/checkpoints;
- DDP/multi-GPU.

### A8. Deduplication and Conflict Control

Lightning shares large parts of the conversion journey with PyTorch and the job
lifecycle skill. Milestone 8 should treat that shared content as a product
contract, not copied prose. The goal is that an agent sees one source of truth
for each decision.

Ownership rules:

| Topic | Owner | PyTorch skill | Lightning skill | Job lifecycle skill |
| --- | --- | --- | --- | --- |
| Dependency installation before inspection/import | `_shared/dependency-install.md` | Reference only | Reference only | Reference only |
| Runtime output location outside source tree by default | `_shared/runtime-output-guidance.md` | Reference only | Reference only | Reference only |
| PyTorch-family tensor/state-dict exchange | `_shared/pytorch-model-exchange.md` | Reference only | Reference only | Does not load |
| Validation evidence, metrics, artifact paths | `_shared/validation-evidence.md` and `_shared/metrics-and-artifact-reporting.md` | Framework-specific examples only | Framework-specific examples only | Existing/exported job evidence only |
| SimEnv limitations | `_shared/nvflare-job-lifecycle.md` | Reference only | Reference only | Owns operational implications |
| Recipe discovery and selection pattern | Framework conversion skill | Owns PyTorch recipes | Owns Lightning-appropriate PyTorch recipes | Does not select recipes for new conversion |
| Generated `client.py` conversion pattern | Framework conversion skill | Manual Client API / `FLModel` exchange | `flare.patch(trainer)` and Lightning callbacks | Does not edit generated client code |
| Generated `job.py` for a new converted job | Framework conversion skill | Owns PyTorch generated job | Owns Lightning generated job | Validates existing/exported jobs only |
| Submit/monitor/download against a running FLARE system | `nvflare-job-lifecycle` | Hand off after export or user request | Hand off after export or user request | Owns |
| POC prepare/start/stop/cleanup/recovery | `nvflare-poc-workflow` | Handoff only | Handoff only | No, except readiness handoff |

The POC row applies to every current and future framework conversion skill, not
only PyTorch and Lightning. Framework skills should know when to suggest the
handoff, but the operational details remain in `nvflare-poc-workflow`.

Conflict rules:

- `nvflare-convert-lightning` must not instruct manual `FLModel` send/receive
  as the default path. That belongs to `nvflare-convert-pytorch`.
- `nvflare-convert-pytorch` must not mention `flare.patch(trainer)` except as a
  negative-routing hint that Lightning should be used instead.
- `nvflare-job-lifecycle` must not tell the agent to rewrite framework training
  code or generate a new recipe job from raw training code.
- Shared guidance files must not include framework-specific code snippets beyond
  tiny neutral examples. Full snippets belong in the owning framework skill.
- If a shared instruction changes, PyTorch and Lightning should inherit it by
  reference. Do not paste the updated wording into both skills.
- If PyTorch and Lightning need different advice for the same topic, the shared
  file should state the neutral rule and each framework reference should state
  only the framework-specific exception.

Lint and review checks:

- Add a doc-crosslink check that framework skills reference shared guidance
  rather than duplicating known shared section headings.
- Add a same-category overlap check between `nvflare-convert-pytorch` and
  `nvflare-convert-lightning` requiring adjacent negative trigger cases.
- Add command-drift checks for shared commands once, then require framework
  skills to link to the shared command section.
- Add a reviewer checklist item: "Does this skill duplicate shared lifecycle,
  dependency, runtime-output, or validation guidance?"
- Add an eval case where the prompt mentions both `torch` and `Trainer`; the
  expected lead skill is Lightning when Lightning APIs are present.
- Add an eval case where the prompt has plain `torch.nn.Module` plus
  `DataLoader`; the expected lead skill is PyTorch and Lightning must not
  trigger.

## Track B: CLI/Product Implementation

### B1. Extend `nvflare agent inspect` Framework Detection

Update `nvflare/tool/agent/inspector.py`:

- detect `pytorch_lightning` and `lightning.pytorch` imports;
- detect symbols:
  - `LightningModule`;
  - `LightningDataModule`;
  - `Trainer`;
  - `Callback`;
  - `ModelCheckpoint`;
- detect calls:
  - `Trainer(...)`;
  - `trainer.fit(...)`;
  - `trainer.validate(...)`;
  - `trainer.test(...)`;
  - `flare.patch(trainer)`;
- distinguish plain PyTorch from Lightning when both `torch` and Lightning are
  present;
- classify Lightning code as `not_converted`, `partial_client_api`,
  `client_api_converted`, `flare_job`, or `exported_job` using Lightning-aware
  signals.

Update the deterministic framework map:

```python
FRAMEWORK_SKILLS = {
    "pytorch": "nvflare-convert-pytorch",
    "pytorch_lightning": "nvflare-convert-lightning",
}
```

Do not list unimplemented future framework skills in runtime routing unless the
skill exists and is packaged.

### B2. Preserve Existing Job-Lifecycle Recommendation

Do not change the existing exported-job handoff:

- exported job -> `nvflare-job-lifecycle`;
- not-converted Lightning project -> `nvflare-convert-lightning`;
- not-converted PyTorch project -> `nvflare-convert-pytorch`;
- safety findings -> include `nvflare-orient` as supporting guidance.

This keeps lifecycle handling separate from conversion handling.

### B3. Improve Inspection Output For Lightning

Add inspect output fields or evidence buckets that are still generic enough for
future frameworks:

```json
{
  "frameworks": [
    {
      "name": "pytorch_lightning",
      "confidence": 0.95,
      "evidence": [
        {"type": "import", "symbol": "pytorch_lightning.Trainer", "path": "train.py"},
        {"type": "class_base", "symbol": "LightningModule", "path": "model.py"}
      ]
    }
  ],
  "conversion_state": "not_converted",
  "skill_selection": {
    "recommended_skills": ["nvflare-convert-lightning"]
  }
}
```

Avoid raw source content in JSON output. Keep current redaction and bounded
evidence behavior.

### B4. Native Skill Install/List

No new public install/list subcommands are needed for Milestone 8.

The existing CLI remains:

```bash
nvflare agent skills install --agent codex|claude [--skill <name>] [--dry-run] --format json
nvflare agent skills list --agent codex|claude --format json
```

Implementation requirements:

- newly added skills are included in the manifest when packaged;
- normal dev/test builds keep `evals/`;
- release builds can filter analysis-only files when `NVFL_RELEASE=1`;
- `skills list` reports all packaged Milestone 8 skills;
- no public runtime skill-evaluation, skill-performance, or skill-benchmarking
  subcommands are added.

### B5. Dev Tools Admission Lints

Update `dev_tools/agent/skills/checks/`:

- include `nvflare-convert-lightning` in conversion-category overlap checks;
- require an adjacent negative case between PyTorch and Lightning;
- validate all Lightning skill command references;
- validate links to shared references;
- enforce frontmatter and 200-line `SKILL.md` limit;
- keep `evals/` as analysis metadata, not runtime instructions.

### B6. Optional Future CLI Candidates

Do not implement these as public CLI commands in Milestone 8, but design the
skills so repeated deterministic patterns can later be promoted:

- exported-job validation helper;
- metrics-summary extraction;
- support-bundle redaction;
- job artifact manifest/fingerprint inspection.

These should remain helper-script or skill-reference concepts unless a clear
product command contract is accepted.

## Test Plan

### Skill Tests

- frontmatter validation for all new skills;
- markdown link validation;
- command drift validation;
- trigger overlap validation:
  - Lightning positive;
  - PyTorch positive;
  - Lightning prompt must not trigger PyTorch;
  - PyTorch prompt must not trigger Lightning;
  - exported job must recommend lifecycle, not conversion;
- release-filter test proves `evals/` is not required for runtime-installed
  skills.

### CLI Tests

Add or extend tests under `tests/unit_test/tool/agent/`:

- inspect detects `pytorch_lightning` import;
- inspect detects `lightning.pytorch` import;
- inspect detects `LightningModule` subclass;
- inspect detects `Trainer(...)` usage;
- inspect recommends `nvflare-convert-lightning` for not-converted Lightning;
- inspect recommends `nvflare-convert-pytorch` for plain PyTorch;
- inspect recommends `nvflare-job-lifecycle` for exported job even when
  Lightning files are present;
- inspect does not leak source content or secrets in Lightning evidence;
- agent command registry still advertises only info, inspect, doctor,
  skills install, and skills list.

### Packaging Tests

- dev wheel includes Lightning `evals/`;
- release build with `NVFL_RELEASE=1` filters Lightning `evals/`;
- release-filtered Lightning skill installs successfully;
- release-filtered installed markdown has no broken runtime references;
- `skills list` includes Lightning when present and no unrelated third-party
  skill conflicts are reported.

### Benchmark/Review Evidence

Before marking Milestone 8 complete:

- run PyTorch and Lightning conversion cases with Codex and Claude;
- compare with-skills vs without-skills for correctness, runtime, dependency
  behavior, generated structure, token usage, and metric evidence;
- record evidence summaries in the benchmark/reviewer output location or mark
  the skill draft/internal in handoff metadata;
- do not use the Lightning skill as a template for additional framework skills
  until it has benchmark/reviewer evidence.

## Documentation Changes

Update:

- `docs/design/agent_implementation_plan.md`
  - rename Milestone 8;
  - add `nvflare-convert-lightning` to M8 deliverables;
  - remove Lightning from M9;
  - add explicit CLI-vs-skill implementation split.

- `docs/design/agent_integration.md`
  - change Lightning tier from generic `next` to M8/base expansion;
  - mention that `agent inspect` routes Lightning projects to the Lightning
    conversion skill and exported jobs to lifecycle.

- `docs/design/agent_skill_authoring.md`
  - keep Lightning in the conversion table;
  - mark it as Milestone 8 rather than later framework wave;
  - add shared-guidance factoring rules for PyTorch and Lightning.

- `docs/design/agent_skill_evaluation.md`
  - add PyTorch-vs-Lightning adjacent trigger cases;
  - add runtime-vs-analysis packaging checks for the new skill.

## Detailed Implementation Plan

Milestone 8 should be implemented in stages, with skill content and CLI support
kept separate. The critical path is: design alignment -> shared guidance
factoring -> deterministic inspect/routing support -> Lightning skill content
-> PyTorch/Lightning conversion benchmark -> lifecycle skill content -> lint and
packaging -> full benchmark evidence.

### Stage 1: Scope And Branch Hygiene

Goal: make the PR scope clear before adding files.

Work:

- update the four design docs listed above;
- remove `nvflare-setup-local`, `production-approval.md`, and standalone
  `metrics-and-artifacts.md` from the Milestone 8 plan;
- confirm no public `skills evaluate`, `skills benchmark`, or
  `skills performance` commands are in scope;
- identify which changes belong in the skills/agent PR versus the benchmark
  harness PR.

Dependencies: none.

Can run in parallel: no; this is the scope gate for the rest of the work.

Exit criteria:

- design docs agree on the three Milestone 8 skills:
  `nvflare-convert-lightning`, `nvflare-poc-workflow`, and
  `nvflare-job-lifecycle`;
- no stale Lightning-as-Milestone-9 or production-approval references remain.

### Stage 2: Shared Guidance Refactor

Goal: avoid duplicating PyTorch, Lightning, lifecycle, dependency, runtime, and
evidence guidance before adding the new Lightning skill.

Work:

- add or revise `_shared/dependency-install.md`;
- add or revise `_shared/runtime-output-guidance.md`;
- add or revise `_shared/validation-evidence.md`;
- add or revise `_shared/metrics-and-artifact-reporting.md`;
- add `_shared/pytorch-model-exchange.md` for PyTorch-family tensor/state-dict
  rules only;
- update `nvflare-convert-pytorch` to reference shared guidance instead of
  duplicating it;
- keep TensorFlow and future non-PyTorch skills from loading
  `_shared/pytorch-model-exchange.md`.

Dependencies: Stage 1.

Can run in parallel:

- shared dependency/runtime/evidence guidance can be edited in parallel with
  CLI inspect work from Stage 3;
- PyTorch skill cleanup should happen before Lightning content is finalized.

Exit criteria:

- PyTorch skill still passes current lint/evals;
- shared references have no framework-specific full examples except where the
  reference is explicitly PyTorch-family scoped;
- `nvflare-convert-pytorch` behavior is preserved.

### Stage 3: Deterministic Inspect And Routing Support

Goal: make `nvflare agent inspect` produce enough evidence for agents to choose
Lightning, PyTorch, lifecycle, or diagnosis skills without loading every skill.

Work:

- detect `pytorch_lightning` and `lightning.pytorch` imports;
- detect `LightningModule`, `LightningDataModule`, and `Trainer` evidence;
- keep plain PyTorch detection separate from Lightning detection;
- preserve exported-job detection as higher priority than raw framework
  detection;
- recommend:
  - exported job -> `nvflare-job-lifecycle`;
  - Lightning not-converted project -> `nvflare-convert-lightning`;
  - plain PyTorch not-converted project -> `nvflare-convert-pytorch`;
- add bounded, redacted evidence entries without raw source leakage.

Dependencies: Stage 1.

Can run in parallel:

- can run while Stage 2 shared guidance is being edited;
- tests can be drafted in parallel with the implementation.

Exit criteria:

- inspect tests cover both Lightning import styles, Lightning class evidence,
  Trainer evidence, plain PyTorch fallback, and exported-job priority;
- command registry still exposes only the intended agent CLI surface.

### Stage 4: Lightning Skill Content

Goal: add Lightning as a thin PyTorch-family framework adapter, not a duplicate
of the PyTorch skill.

Work:

- add `skills/nvflare-convert-lightning/SKILL.md`;
- add Lightning references:
  - `lightning-detection.md`;
  - `lightning-conversion.md`;
  - `lightning-validation.md`;
  - `lightning-ddp-and-tracking.md`;
- reuse PyTorch recipe selection through the existing PyTorch recipe discovery
  rules;
- reference `_shared/pytorch-model-exchange.md` for tensor/state-dict exchange;
- reference shared dependency/runtime/evidence guidance;
- add Lightning evals.

Dependencies:

- Stage 2 should be mostly complete, especially
  `_shared/pytorch-model-exchange.md`;
- Stage 3 should define inspect evidence and recommendation names.

Can run in parallel:

- Lightning content can start while Stage 3 tests are being finalized;
- DDP/tracking reference can be drafted after the core conversion reference.

Exit criteria:

- Lightning skill does not instruct manual `FLModel` exchange as the default;
- Lightning uses PyTorch recipes and does not define a separate recipe family;
- PyTorch/Lightning negative trigger cases are adjacent and explicit.
- Lightning eval fixtures assert the patched-loop contract: no generated
  manual `FLModel` send/receive path, no `input_model` passed to `Trainer`, and
  optional `flare.receive()` used only for round/task metadata.

### Stage 5: Two-Conversion-Skill Verification Checkpoint

Goal: measure the behavior of the two base conversion skills,
`nvflare-convert-pytorch` and `nvflare-convert-lightning`, before adding POC or
job-lifecycle skills. This keeps the first benchmark question focused on
conversion-skill coverage and shared guidance, without extra operational skill
context.

Work:

- run skill lint/admission tests for PyTorch and Lightning;
- run `nvflare agent inspect` tests for Lightning/PyTorch routing;
- run install/list tests for PyTorch and Lightning skills;
- run packaging checks needed to install the Lightning skill in dev mode;
- run the private deterministic checkpoint:

  ```bash
  python -m dev_tools.agent.skills.checks.milestone8_checkpoint \
    --repo-root . \
    --benchmark-evidence <stage5-benchmark-evidence.json> \
    --format json
  ```

- run focused PyTorch and Lightning conversion benchmark cases with Codex and
  Claude, without adding lifecycle/POC skills to the comparison;
- record PyTorch and Lightning observed correctness, runtime, dependency
  behavior, generated structure, token usage, and metric evidence in the
  benchmark/reviewer output location.

Dependencies:

- Stage 2 shared guidance refactor;
- Stage 3 inspect/routing support;
- Stage 4 Lightning skill content.

Can run in parallel:

- Codex and Claude conversion runs can run independently;
- PyTorch and Lightning benchmark cases can run independently once the skill
  install image/environment is ready.

Exit criteria:

- Lightning works well enough to continue, or it is marked draft/internal with
  clear blockers;
- PyTorch behavior is not regressed by shared guidance refactoring;
- token/runtime effect of adding the Lightning skill is understood before POC
  lifecycle content is added.

The checkpoint command intentionally does not run agent benchmarks. It validates
the deterministic source checks and then requires a separate evidence JSON with
four run records: Codex/PyTorch, Codex/Lightning, Claude/PyTorch, and
Claude/Lightning. Each run record must include the agent, skill, correctness,
runtime seconds, dependency behavior, generated structure, token usage, metric
evidence, and artifact location. Without this evidence the checkpoint reports
`incomplete`, not `ok`.

### Stage 6: Lifecycle Skill Content

Goal: add shared operational playbooks after conversion-skill effects are known,
without adding new public commands or duplicating conversion skill behavior.

Work:

- add `skills/nvflare-poc-workflow/SKILL.md`;
- add POC references:
  - `poc-lifecycle.md`;
  - `poc-readiness.md`;
  - `poc-cleanup-and-restore.md`;
- add `skills/nvflare-job-lifecycle/SKILL.md`;
- add job lifecycle references:
  - `job-validation.md`;
  - `submit-monitor-download.md`;
- put result summary guidance inside `submit-monitor-download.md`, not in a
  separate metrics reference;
- add lifecycle evals.

Dependencies:

- Stage 1 for scope;
- Stage 5 two-conversion-skill verification checkpoint;
- existing `nvflare agent inspect`, `nvflare agent doctor`, and `nvflare job`
  command contracts.

Can run in parallel:

- POC workflow and job lifecycle can be authored in parallel after Stage 5;
- lifecycle evals can be drafted once the trigger/negative-trigger lists are
  stable.

Exit criteria:

- lifecycle skills are concise and load references only when needed;
- POC cleanup/orphan-process recovery is isolated to the cleanup reference;
- orphan-process recovery can be deferred without blocking Lightning conversion
  or job lifecycle delivery;
- job lifecycle validates/submits/monitors/downloads existing jobs but does not
  rewrite framework training code;
- lifecycle skill text does not change the conversion-skill benchmark question
  that Stage 5 already measured.

### Stage 7: Admission Lints, Packaging, And Release Filtering

Goal: make the new skills installable and keep analysis-only eval files out of
release-filtered publication while preserving dev/test behavior.

Work:

- update skill manifest/build logic for the new skill directories;
- keep `evals/` in normal dev/test builds;
- filter `evals/` only for `NVFL_RELEASE=1` release builds;
- verify release-filtered installed skills do not reference removed runtime
  files;
- update `dev_tools/agent/skills/checks/` for Lightning trigger overlap,
  command drift, links, frontmatter, and runtime-vs-analysis packaging;
- update install/list tests for new skills.

Dependencies:

- Stage 4 and Stage 6 skill directories;
- packaging filter decision from the skills/agent PR.

Can run in parallel:

- lint rule updates can start while skill content is stabilizing;
- release-filter tests can be written once file paths are known.

Exit criteria:

- `skills list` includes the Milestone 8 skills;
- release-filtered install works without `evals/`;
- normal dev/test build still keeps analysis metadata.

### Stage 8: Targeted Verification

Goal: catch deterministic regressions before full lifecycle benchmark runs.

Run targeted checks in this order:

1. Skill lint/admission tests for changed skills.
2. `nvflare agent inspect` unit tests.
3. `nvflare agent skills install/list` tests.
4. Packaging/release-filter tests.
5. Markdown link checks for touched docs and skill references.
6. Project style check with `./runtest.sh -s`.

Dependencies: Stages 2-7.

Can run in parallel:

- skill lint tests and inspect tests can run independently;
- packaging tests should run after the skill tree settles.

Exit criteria:

- targeted tests pass;
- style check passes or any skipped check is explicitly documented with reason.

### Stage 9: Full Benchmark And Review Evidence

Goal: prove the full Milestone 8 skill set improves real agent behavior before
calling lifecycle skills public-ready.

Work:

- keep the Stage 5 PyTorch/Lightning conversion evidence as the baseline for
  conversion-skill impact;
- run lifecycle-specific cases with Codex and Claude:
  - local POC prepare/start/status/stop/cleanup;
  - exported job validation;
  - submit/monitor/download/result-summary against a running local system;
- rerun conversion cases only if Stage 6 or Stage 7 changed shared references
  loaded by conversion skills;
- compare with-skills versus without-skills for:
  - structure correctness;
  - job execution status;
  - final and round metric evidence;
  - generated-code quality;
  - dependency install behavior;
  - runtime and non-install runtime;
  - token usage and skill-loading cost;
- record evidence in benchmark/reviewer output or mark the skill
  draft/internal in handoff metadata;
- run reviewer pass focused on trigger boundaries, duplicated context, and
  result evidence.

Dependencies:

- Stage 8 green;
- benchmark harness branch or local harness availability.

Can run in parallel:

- Codex and Claude runs can run independently;
- lifecycle cases can run independently once the local runtime environment is
  available.

Exit criteria:

- each public-ready Milestone 8 skill has benchmark/reviewer evidence;
- any skill without evidence is marked draft/internal and is not used as a
  template for later framework skills.

### Parallel Work Map

| Stage | Lane | Can Start After | Work | Blocks |
| --- | --- | --- | --- | --- |
| 1 | Scope and branch hygiene | none | design scope, PR split, removed items | all later stages |
| 2 | Shared guidance | Stage 1 | dependency/runtime/evidence/model-exchange refs, PyTorch cleanup | Stage 4 final content, Stage 5 |
| 3 | Inspect/CLI support | Stage 1 | Lightning evidence detection, routing tests | Stage 4 routing text, Stage 5 |
| 4 | Lightning skill | Stage 2 shared guidance draft + Stage 3 routing names stable | SKILL/references/evals | Stage 5 |
| 5 | PyTorch/Lightning benchmark checkpoint | Stage 4 | Codex/Claude PyTorch and Lightning conversion runs without POC/lifecycle skills | Stage 6 |
| 6 | Lifecycle skills | Stage 5 | POC and job lifecycle SKILL/references/evals | Stage 7, Stage 9 |
| 7 | Lints/packaging | Stage 4 and Stage 6 skill paths stable | admission checks, release filter, install/list tests | Stage 8 |
| 8 | Targeted verification | Stage 7 | targeted tests, markdown links, style | Stage 9 |
| 9 | Full benchmark evidence | Stage 8 | Codex/Claude lifecycle and any needed conversion reruns | public-ready signoff |

### Dependency Summary

- Lightning content depends on shared PyTorch-family model-exchange guidance.
- Lightning routing depends on inspect detecting Lightning separately from plain
  PyTorch.
- Lifecycle skill implementation should wait until the PyTorch/Lightning
  conversion checkpoint has measured conversion correctness, token, and runtime
  effects without POC/lifecycle skill context.
- Lifecycle skills depend on existing `inspect`, `doctor`, and `job` command
  behavior, but do not require new public CLI commands.
- Release filtering depends on skill packaging paths being stable.
- Benchmark evidence depends on install/list, inspect routing, and lint checks
  passing first.

## Non-Goals

- Do not add a generic conversion skill.
- Do not add public runtime skill-evaluation, skill-performance, or
  skill-benchmarking subcommands.
- Do not move benchmark execution into normal runtime skills.
- Do not make `nvflare-job-lifecycle` own framework conversion.
- Do not make `nvflare-convert-lightning` own POC setup, job submission, or
  broad diagnosis.
- Do not add TensorFlow/XGBoost/sklearn/Hugging Face in the same change.

## Acceptance Criteria

Milestone 8 is complete when:

- `nvflare-convert-lightning`, `nvflare-poc-workflow`, and
  `nvflare-job-lifecycle` exist with valid frontmatter, references, and evals;
- `nvflare agent inspect` recommends Lightning conversion for Lightning projects
  and lifecycle for exported jobs;
- PyTorch and Lightning trigger overlap is covered by deterministic lint and
  eval cases;
- release-filtered installed skills work without `evals/`;
- no unsupported public skill-evaluation commands are exposed;
- benchmark/reviewer evidence exists for each public-ready Milestone 8 skill, or
  the skill is explicitly marked draft/internal.
