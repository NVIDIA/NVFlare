# NVFLARE Agent Stage 1: Agent-Ready Foundation

## Document Control

| Field | Value |
| --- | --- |
| Created date | 2026-04-27 |
| Last updated | 2026-04-27 |
| Parent design | [nvflare_agent_consumer_readiness.md](nvflare_agent_consumer_readiness.md) |
| Implementation plan | [nvflare_agent_foundation_implementation_plan.md](nvflare_agent_foundation_implementation_plan.md) |
| Related CLI design | [nvflare_cli.md](nvflare_cli.md) |
| Stage | 1 - Agent-ready foundation |

## Goal

Make NVFLARE reliable for external coding agents that are acting as developer
tools. The target user may only have installed NVFLARE with:

```bash
python -m pip install nvflare
```

They should not need to clone the NVFLARE GitHub repository to use agent skills,
inspect local code, prepare POC, submit jobs, or diagnose common failures.

Stage 1 is about making existing FLARE capabilities easy for agents to discover,
invoke, and verify. It is not about making FLARE itself an agentic runtime.

## Scope

In scope:

- Packaged agent skills installed from the installed `nvflare` package.
- Stable CLI JSON/schema/error contracts for agent execution.
- `job.py` generation and export conventions for agent-created jobs.
- Read-only local inspection of user code and job folders.
- Local environment readiness checks.
- Local install, POC setup, and job lifecycle workflows through the normal CLI.
- Setup/deployment guidance for multi-machine, Docker, K8s, and cloud-provider
  environments, with explicit human handoff points for production actions.
- Production-safe guidance that uses scoped startup-kit identity selection and
  explicit approval boundaries.
- Diagnosis workflows using job metadata, logs, stats, downloaded artifacts, and
  system state.

Out of scope:

- MCP server implementation. MCP is a later stage unless it avoids agent-written
  Python glue for functionality that has no CLI.
- Auto-research experiment registry, sweeps, resumability, and report generator.
- Fully automatic conversion of arbitrary ML code without user review.
- Agents handling private keys, bypassing authorization, or approving production
  actions automatically.
- Replacing Client API, Recipe API, FedJob, SimEnv, or Session APIs.

## Stage 1 Assumptions

PR #4479 is assumed to close the core admin CLI baseline after it merges:

- `nvflare job ...`
- `nvflare system ...`
- `nvflare study ...`
- `nvflare poc ...`
- `nvflare config kit ...`

Those commands are the operational substrate for Stage 1. The Stage 1 work below
adds agent-facing skills, conventions, inspection, and readiness/diagnosis
helpers on top of that substrate. It should not reintroduce one-off wrappers for
every CLI command.

## Design Principles

### CLI Is the Execution Contract

Agents should use CLI commands for admin operations because they are
reproducible, visible to users, and work in shells, CI, notebooks, and coding
agents. All public agent-used commands must support:

- `--format json` for exactly one JSON envelope on stdout for single-result
  commands.
- `--format jsonl` for streaming commands, where each line is one complete JSON
  event object.
- `--schema` for machine-readable command discovery.
- stable error codes and documented exit codes.
- schema metadata for `streaming`, `output_modes`, and `idempotent`.
- no required prompts in automation paths.
- progress, warnings, and diagnostics on stderr in JSON mode.

`--schema` output must be a concrete JSON object, not raw help text. Minimum
shape:

```json
{
  "schema_version": "1",
  "command": "nvflare job submit",
  "description": "Submit a job to the selected FLARE server.",
  "arguments": [],
  "flags": [
    {"name": "job-folder", "short": "j", "type": "path", "required": true},
    {"name": "format", "type": "enum", "values": ["json"], "required": false},
    {"name": "idempotency-key", "type": "string", "required": false}
  ],
  "output_modes": ["json"],
  "streaming": false,
  "idempotent": false,
  "idempotency_key_supported": true,
  "mutating": true,
  "examples": [
    "nvflare job submit -j ./jobs/hello-pt --idempotency-key <uuid> --format json"
  ]
}
```

Agents should use `nvflare job wait <job_id> --timeout <seconds> --format json`
when they need one final result, and `nvflare job monitor <job_id> --timeout
<seconds> --format jsonl` when they need progress events. `job monitor` exits
when the job reaches `COMPLETED`, `FAILED`, or `ABORTED`; timeout emits a final
JSONL event with `terminal: true` and `status: "TIMEOUT"`. Agent workflows
should submit jobs with an idempotency key:

```bash
nvflare job submit -j <job_folder>/<job_name> --idempotency-key <uuid> --format json
```

Skills must not blindly retry non-idempotent commands.

### Code Inspection Security Boundary

`agent inspect` helps an agent understand user code, but user code may contain
credentials, private endpoints, data paths, and proprietary model details. The
command must return structural facts, not raw source content:

- default redaction is on for JSON and skill outputs.
- `--redact on|off` controls whether literal evidence values may be shown.
- secret-like string literals, tokens, passwords, keys, and connection strings
  are never included in default output.
- `ABSOLUTE_DATA_PATH` findings should include file/line and pattern type, not
  the full path when redaction is enabled.
- conversion skills must require a review for hardcoded credentials and private
  data paths before export.

Sensitive users should run agent-assisted conversion in a local or approved
enterprise agent environment.

### Skills Are Workflow Playbooks

Skills are not one wrapper per command. They teach agents how to combine
existing commands and APIs safely:

- when to use Client API, Recipe API, FedJob, SimEnv, POC, or production.
- which files to inspect.
- which edits are safe.
- which commands to run.
- how to validate before submission.
- how to diagnose common failures.

Skills must also be machine-readable. Each skill should include structured
frontmatter for inputs, outputs, CLI commands used, approval checkpoints, and
recovery categories. The Markdown body explains the workflow, but the
frontmatter is what agents and tests use to validate that the skill still
matches the installed CLI.

Minimum skill frontmatter:

```yaml
---
skill_version: "1.0.0"
min_flare_version: "2.8.0"
max_flare_version: null
maintainer: "nvflare-cli"
status: active
tier: conversion_core
inputs:
  - name: target_path
    type: path
outputs:
  - name: export_dir
    type: path
agent_requirements:
  python_execution: required
  file_editing: required
  shell_execution: required
  network_access: optional
approval_checkpoints:
  - id: review_converted_code
    risk_level: low
    confirm_phrase: null
cli_commands_used:
  - nvflare agent inspect
  - nvflare job submit
recovery_categories:
  - FIXABLE_BY_CODE
  - FIXABLE_BY_CONFIG
---
```

Every skill should support a dry-run style plan. The plan should list commands,
files to create or modify, mutating steps, approval checkpoints, and estimated
duration before the agent performs risky work.

### Stage 1 Workflow Categories

Stage 1 skills and docs should be organized around these categories:

| Category | Stage 1 Treatment |
| --- | --- |
| Setup and deployment | Provide `nvflare-setup-local` in the first bundle; document multi-machine, Docker, K8s, and cloud deployment skills as Stage 1+ with explicit human handoff and environment checks |
| Inspect and convert code | Provide `agent inspect` plus Client API/PyTorch conversion guidance; extend to other frameworks through Stage 1+ skills |
| Generate, validate, and export jobs | Standardize `job.py`, `SimEnv`, and `--export --export-dir` as the agent path |
| Identity, provisioning, and studies | Use `nvflare config kit`, distributed provisioning commands, and `nvflare study` with authorization-aware skill behavior |
| Job lifecycle and operations | Use the normal `nvflare job` and `nvflare system` command surfaces after environment setup |
| Diagnosis and troubleshooting | Start with diagnosis skills that collect meta/logs/stats/system state; promote to CLI diagnosis when stable |

Skill recommendations must consider agent capability. A CLI-only agent can run
`job list` or `doctor`, but cannot follow a skill that edits `client.py` or runs
`python job.py`. Skills should declare `agent_requirements`, and
`nvflare agent skills list --filter capability=cli_only --format json` should
let constrained agents discover eligible workflows.

Setup/deployment must not be hidden inside job workflow skills. Local install,
POC, multi-machine, Docker, K8s, and cloud-provider environments have different
prerequisites, network assumptions, startup-kit placement, and approval
boundaries. Agents need explicit guidance before they submit or operate jobs in
those environments.

### Python APIs Are for Job Code and Local Simulation

Agents should write Python code when generating or modifying user job artifacts:

- Client API for training script integration.
- Recipe API or FedJob for job construction.
- SimEnv for local validation.

Agents should not generate Python admin glue for routine operations that already
have CLI coverage, such as job submit/list/logs or system status.

### GitHub Clone Is Not Required

Released users should get Stage 1 skills and templates from the installed
`nvflare` package. Source-tree overrides are allowed for NVFLARE developers, but
must be explicit.

Canonical templates live under `nvflare/agent/templates/` in the wheel and are
loaded with `importlib.resources`. The `examples/` tree is not part of the
agent runtime contract.

## User Workflows

### Workflow 1: Install Skills

Expected released-user flow:

```bash
python -m pip install nvflare
nvflare agent skills install --agent codex
nvflare agent skills validate --agent codex --format json
```

The command belongs under the agent namespace and should support both named
agent shortcuts and explicit target directories:

```bash
nvflare agent skills install --agent claude
nvflare agent skills install --agent codex
nvflare agent skills install --agent openclaw
nvflare agent skills install --target /path/to/agent/skills/nvflare
nvflare agent skills list --format json
nvflare agent skills validate --format json
```

Do not add a top-level `nvflare install-skills` command. Skill installation is
agent-specific functionality, and keeping it under `nvflare agent` avoids adding
another broad top-level command.

The installer copies packaged skill files from the installed Python package,
using package resources, not a repo checkout. It should support:

- named agent shortcuts such as `--agent codex`.
- explicit `--target` directories for project-local or custom installs.
- `--dry-run`.
- backup of an existing NVFLARE-managed skill directory before overwrite.
- clear warning if a target location is not writable.
- version compatibility checks from skill metadata.
- runtime validation against the installed NVFLARE version and command schemas.

## End-to-End Stage 1 Use Cases

The sections below show how `agent inspect` and packaged skills fit together in
real workflows. Skill names are shown as conceptual playbooks. The actual agent
may invoke them automatically based on the user request and the inspection
result.

### Use Case A: Convert Centralized Training Code and Run in POC

User request:

```text
Convert this training code to FLARE and run it locally with two clients.
```

Agent flow:

1. Install or verify skills:

   ```bash
   nvflare agent skills install --agent codex
   nvflare agent doctor --format json
   ```

2. Inspect the user's project:

   ```bash
   nvflare agent inspect ./my_training_repo --format json
   ```

   The agent reads:

   - `frameworks` to select the conversion skill, such as
     `nvflare-convert-pytorch`, `nvflare-convert-lightning`,
     `nvflare-convert-tensorflow`, `nvflare-convert-huggingface`,
     `nvflare-convert-xgboost`, `nvflare-convert-sklearn`, or
     `nvflare-convert-numpy`.
   - `conversion_state` to decide whether to convert, complete a partial Client
     API conversion, or skip conversion.
   - `flare_usage.job_py` and `flare_usage.export_supported` to decide whether
     to generate or update `job.py`.

3. Apply the selected conversion skill:

   ```text
   nvflare-convert-<framework>
   nvflare-generate-job
   nvflare-local-validation
   ```

   The agent edits user files only where needed, usually `client.py`,
   `model.py`, and `job.py`.

4. Validate locally with `SimEnv`:

   ```bash
   python job.py
   ```

5. Export the job definition:

   ```bash
   python job.py --export --export-dir <job_folder>
   ```

6. Prepare and start POC:

   ```bash
   nvflare poc prepare --format json
   nvflare poc start --format json
   nvflare poc wait-ready --timeout 60 --format json
   nvflare config kit show --format json
   nvflare agent doctor --online --format json
   ```

   If `poc start --format json` is made blocking, `poc wait-ready` can be a
   no-op or an alias. The agent workflow needs one deterministic readiness
   point before online doctor or job submission.

7. Submit and wait for the exported job:

   ```bash
   nvflare job submit -j <job_folder>/<job_name> --idempotency-key <uuid> --format json
   nvflare job wait <job_id> --timeout 3600 --format json
   nvflare job logs --site all <job_id> --tail 200 --format json
   nvflare job download <job_id> --output-dir ./<job_id> --format json
   ```

   SimEnv validation is not enough for production readiness. It does not test
   TLS/authentication, startup-kit selection, network behavior, site-local data,
   per-site resource limits, or long-running client process behavior. Before
   production submission, agents should also run the exported job in POC and
   mark `_export_manifest.json` with `poc_validated: true` only after the POC
   job succeeds. Production submission should warn when this field is false or
   missing.

   Local validation must also check FL semantics, not only process success. The
   `nvflare-local-validation` skill should verify receive -> apply -> train ->
   send ordering and compare round-start/end metrics from `metrics_summary.json`
   when available. A large jump between one round's end loss and the next
   round's start loss, or no global improvement across rounds, should produce a
   warning before production submission.

8. If the job fails, run the diagnosis skill:

   ```text
   nvflare-diagnose-job
   ```

   The skill collects:

   ```bash
   nvflare job meta <job_id> --format json
   nvflare job logs --site all <job_id> --tail 200 --format json
   nvflare job stats <job_id> --format json
   nvflare system status --format json
   nvflare system resources --format json
   ```

### Use Case B: Existing Code Already Uses Client API

User request:

```text
This repo is already converted to FLARE. Run it and tell me if it works.
```

Agent flow:

1. Inspect:

   ```bash
   nvflare agent inspect ./repo --format json
   ```

2. If `conversion_state == "client_api_converted"`, skip conversion and use:

   ```text
   nvflare-generate-job
   nvflare-local-validation
   nvflare-poc-workflow
   nvflare-job-lifecycle
   ```

3. If `conversion_state == "partial_client_api"`, use the framework conversion
   skill only to complete the missing Client API pieces before generating or
   updating `job.py`.

### Use Case C: Existing FLARE Job Folder

User request:

```text
Submit this FLARE job and collect logs.
```

Agent flow:

1. Inspect:

   ```bash
   nvflare agent inspect ./jobs/hello-pt --format json
   ```

2. If `conversion_state == "exported_job"`, skip conversion and submit
   directly:

   ```bash
   nvflare config kit show --format json
   nvflare job submit -j ./jobs/hello-pt --idempotency-key <uuid> --format json
   nvflare job wait <job_id> --timeout 3600 --format json
   nvflare job logs --site all <job_id> --tail 200 --format json
   ```

3. If `conversion_state == "flare_job"`, the path is source code with `job.py`
   but not necessarily an exported job definition. Export first:

   ```bash
   python job.py --export --export-dir <job_folder>
   nvflare job submit -j <job_folder>/<job_name> --idempotency-key <uuid> --format json
   ```

### Use Case D: Production Submit with Existing Startup Kit

User request:

```text
Submit this exported job to my production FLARE system.
```

Agent flow:

1. Check identity and system readiness:

   ```bash
   nvflare config kit list --format json
   nvflare config kit show --format json
   nvflare agent doctor --online --format json
   ```

2. If the default identity is not the one the user wants, ask which startup kit
   to use. Agent workflows should prefer a scoped per-command selector rather
   than mutating the global active kit:

   ```bash
   nvflare agent doctor --online --kit-id <id> --format json
   ```

   `nvflare config kit use <id>` remains useful as a human default, but it
   writes to the shared `~/.nvflare/config.conf`. Skills should avoid it in
   concurrent agent, notebook, or shared-machine workflows.

3. Confirm production submission with the user.

4. Preflight the target study:

   ```bash
   nvflare study list --kit-id <id> --format json
   ```

   Proceed only if the target study exists and reports `can_submit_job: true`.
   Otherwise surface a structured authorization/study finding before job
   submission.

5. Re-run online doctor if the previous snapshot is stale.

6. Submit, wait, and collect evidence:

   ```bash
   nvflare job submit -j <job_folder>/<job_name> --kit-id <id> --idempotency-key <uuid> --format json
   nvflare job wait <job_id> --kit-id <id> --timeout 3600 --format json
   nvflare job logs --site all <job_id> --kit-id <id> --tail 200 --format json
   nvflare job download <job_id> --kit-id <id> --output-dir ./<job_id> --format json
   ```

### Use Case E: Federated Statistics Before Training

User request:

```text
Before training, run federated statistics to understand the data distribution.
```

Agent flow:

1. Use `nvflare-federated-statistics` to create or adapt a federated statistics
   job using the standard example layout.
2. Validate locally when possible.
3. Export the job definition.
4. Submit through the same job lifecycle commands.
5. Download and summarize the resulting statistics artifacts.

### Use Case F: Notebook-Based Conversion and Submit

Notebook users need the same workflow with extra guardrails:

1. Verify the notebook kernel and shell can see the same NVFLARE installation:

   ```bash
   nvflare --version
   nvflare agent doctor --format json
   ```

2. Avoid `nvflare config kit use` inside notebook cells. It changes the global
   active kit for every process under the user account. Prefer per-command
   `--kit-id <id>` or a process-scoped startup-kit override once implemented.
3. Export from `job.py` and submit with the same CLI commands used by shell
   workflows.
4. Poll status from a short-running cell instead of blocking the kernel
   indefinitely:

   ```bash
   nvflare job meta <job_id> --kit-id <id> --format json
   ```

5. After a kernel restart, run `nvflare agent doctor --online --format json`
   before re-running `poc prepare`; POC processes may still be running.

A Stage 1 notebook example should show safe POC validation, scoped identity,
bounded monitoring, logs, and job download.

### Workflow 2: Inspect Local Code or Job

```bash
nvflare agent inspect ./my_training_repo --format json
nvflare agent inspect ./jobs/hello-pt --format json
nvflare agent inspect ./job.py --format json
nvflare agent inspect ./my_training_repo --confirm-framework pytorch --redact on --format json
```

`agent inspect` is read-only. It inspects the path supplied by the user. It does
not inspect the installed NVFLARE package, fetch GitHub content, submit jobs, or
modify files.

It should report:

- detected framework(s) and likely entry points.
- whether the code already uses FLARE Client API, regardless of framework.
- whether code appears to be plain training code, partially converted Client API
  code, complete Client API code, FLARE job source, or a submit-ready exported
  job folder.
- whether the directory follows the standard FLARE example layout.
- likely train/evaluate/data-loading functions.
- detected job files and missing job files.
- whether `job.py` supports `--export --export-dir`.
- whether `job.py` imports or constructs `SimEnv` for local validation.
- recipe fit: Recipe-compatible, requires FedJob/ScriptRunner, or uncertain.
- multi-GPU or distributed training patterns and their FL impact.
- recommended next commands.
- dynamic framework resolution warnings and user confirmation needs.

### Workflow 3: Convert Training Code to FLARE

The skill-guided conversion path is:

1. Inspect the user code.
2. Choose the lowest-risk integration lane.
3. Back up files that will be edited into `.nvflare_bak/<timestamp>/`.
4. Patch the training script with Client API boundaries where possible.
   Conversion edits must be idempotent and must not duplicate existing FLARE
   imports or `flare.init()`/`receive()`/`send()` calls.
5. Generate or update `job.py`.
6. Run `SimEnv`.
7. Export the job folder.
8. Submit with CLI only after local validation passes.

The skill output should include `backup_path`, and users or agents should be
able to revert a failed conversion with:

```bash
nvflare agent skills revert --backup .nvflare_bak/<timestamp>/ --format json
```

Preferred `job.py` convention:

```bash
python job.py
python job.py --export --export-dir <job_folder>
nvflare job submit -j <job_folder>/<job_name> --idempotency-key <uuid> --format json
```

This convention is not a new job model. It should formalize the structure used
by packaged templates under `nvflare/agent/templates/`:

1. parse CLI arguments.
2. construct the appropriate Recipe object.
3. construct `SimEnv` for local execution.
4. run the recipe locally, or export the job definition.
5. submit the exported job through `nvflare job submit`.

Source-tree examples can remain developer references, but released skills must
load canonical templates through
`importlib.resources.files("nvflare.agent.templates")` so users with only a pip
install do not need a GitHub checkout.

Generated `job.py` should use `SimEnv` for local execution. It should not use
`PocEnv` or `ProdEnv` as the normal agent automation path. POC and production
submission stay in the CLI layer.

Recipe/FedJob decision rules should be explicit:

- use Recipe when the algorithm is a supported standard FL workflow and the
  training loop can receive/send model state normally.
- use FedJob/ScriptRunner when the existing script should run as a launched
  client task or job assembly needs more direct control.
- use custom Executor/Controller when aggregation, server-side per-round
  decisions, non-model tasks, or orchestration cannot be expressed by Recipe or
  ScriptRunner.
- return `recipe_fit.assessment = "uncertain"` and ask the user when static
  evidence is mixed.

Recipe selection must use metadata, not keyword matching. `nvflare recipe show`
and filtered `recipe list` should expose algorithm, aggregation,
client requirements, framework support, heterogeneity support, privacy
compatibility, tags, parameters, optional dependencies, and example/template
references. The generate-job skill can then run:

```bash
nvflare recipe list --filter framework=huggingface --filter privacy=differential_privacy --format json
```

Existing examples use a few export flag variants such as `--export_job` and
`--export_config`. Stage 1 templates should standardize on
`--export --export-dir`, while `agent inspect` may recognize the existing
variants when inspecting older examples.

### Workflow 3A: Packaged Template Layout

The canonical Stage 1 templates should live in the installed package under
`nvflare/agent/templates/`. The source-tree `examples/hello-world` examples can
mirror the same structure for developers, but skills should not depend on the
source tree.

Recommended layout:

```text
nvflare/agent/templates/<template_name>/
  README.md
  job.py
  client.py
  model.py
  prepare_data.py
  download_data.py
  requirements.txt
```

Not every job needs every file. For example, `model.py`, `prepare_data.py`, and
`download_data.py` may be omitted when the model is inline, data is already
available, or no download step is needed. But new Stage 1 templates should use
these names when the concepts exist.

File responsibilities:

| File | Role |
| --- | --- |
| `README.md` | Human and agent-facing explanation, setup, run/export/submit commands, expected outputs |
| `job.py` | Recipe/FedJob construction, SimEnv local run, export path |
| `client.py` | Client-side training/evaluation entry point using Client API or recipe-compatible script |
| `model.py` | Model definition and model-state helpers |
| `prepare_data.py` | Local data preparation or partitioning for examples |
| `download_data.py` | Optional dataset download step when data is not bundled |
| `requirements.txt` | Example-specific Python dependencies |

Recommended README layout:

1. What this example does.
2. File structure.
3. Setup and dependency installation.
4. Data preparation.
5. Job recipe overview.
6. Local `SimEnv` run command.
7. Export command.
8. POC/production submit command.
9. Expected outputs and artifact locations.
10. Troubleshooting notes.

### Workflow 4: Run POC and Submit Job

POC setup and job operations remain separate:

```bash
nvflare poc prepare --format json
nvflare poc start --format json
nvflare poc wait-ready --timeout 60 --format json
nvflare config kit show --format json
python job.py --export --export-dir <job_folder>
nvflare job submit -j <job_folder>/<job_name> --idempotency-key <uuid> --format json
nvflare job wait <job_id> --timeout 3600 --format json
nvflare job logs --site all <job_id> --tail 200 --format json
nvflare job download <job_id> --output-dir ./<job_id> --format json
nvflare poc stop --restore-kit --format json
```

`poc start --format json` should return actual bound addresses and port conflict
metadata:

```json
{
  "server_address": "localhost:6011",
  "admin_address": "localhost:6012",
  "default_port": 6009,
  "port_conflict": true,
  "warnings": ["Default port 6009 was in use; POC started on 6011"]
}
```

The POC skill should capture these addresses and pass them explicitly to online
readiness checks, for example with `nvflare agent doctor --online --server
<address> --format json`, instead of assuming the active startup kit still
points to the intended POC server.

`job download --format json` should return artifact paths rather than forcing
agents to guess:

```json
{
  "download_path": "./job-abc123",
  "artifacts": {
    "global_model": "job-abc123/server/models/global_model.pt",
    "metrics_summary": "job-abc123/server/metrics_summary.json",
    "client_logs": {"site-1": "job-abc123/site-1/log.txt"}
  },
  "missing_artifacts": []
}
```

After `poc prepare/start`, the POC admin startup kit is available through the
startup-kit registry. The same `nvflare job ...` commands are used for POC and
production; there should not be a separate POC job API. Agents should use a
scoped kit selector when multiple identities are possible.

If POC prepare changes the human default kit, it must record the prior kit id
and the stop workflow must restore it:

```bash
nvflare poc prepare --format json
nvflare poc stop --restore-kit --format json
```

Skills should store `pre_poc_kit_id` in `.nvflare_workflow_state.json` so they
can restore the user's prior context after interruption.
The same state file should store any generated job `idempotency_key` before
submit so the next session can recover the job id if the submit response is
lost. `job list --idempotency-key <uuid> --format json` and `job meta` should
surface this key.

The export step is required. `nvflare job submit` consumes the exported job
definition folder, not the source `job.py` directly.

Exported job folders should be verifiable. Stage 1 export should produce:

- `_export_manifest.json`: required files, source `job.py` path/hash, export
  timestamp, NVFLARE version, exporter, `poc_validated`, and optional
  `poc_validation` details.
- `job_fingerprint.json`: NVFLARE/Python/framework versions, recipe metadata,
  source hash, and export timestamp.

`agent inspect` on an exported folder should validate both files when present,
confirm the required files exist, classify the target as `exported_job`, and
warn if the current source `job.py` no longer matches the manifest hash.
`nvflare job submit` should eventually validate `_export_manifest.json` before
accepting an exported folder.

### Workflow 5: Diagnose Failure

The first Stage 1 implementation can rely on skills that collect data through
existing CLI commands:

```bash
nvflare job meta <job_id> --format json
nvflare job logs --site all <job_id> --tail 200 --format json
nvflare job stats <job_id> --format json
nvflare system status --format json
nvflare system resources --format json
```

A future `nvflare job diagnose <job_id> --format json` can aggregate this into
one command, but it is not required before skills can provide the initial
diagnosis workflow.

The diagnosis skill should include a failure-pattern catalog, not only raw log
collection. Initial patterns should cover:

| Pattern | Evidence | Recovery Category |
| --- | --- | --- |
| `CUDA_OOM` | `CUDA out of memory` | `FIXABLE_BY_CODE` |
| `AUTH_FAILURE` | certificate or authentication rejection | `FIXABLE_BY_CONFIG` |
| `ROUND_TIMEOUT` | round timeout or no client response | `ENVIRONMENT_FAILURE` |
| `IMPORT_ERROR` | `ModuleNotFoundError` or import failure | `FIXABLE_BY_CODE` |
| `ABSOLUTE_DATA_PATH` | local path used on a remote site | `FIXABLE_BY_CODE` |
| `STARTUP_KIT_EXPIRED` | certificate validity failure | `FIXABLE_BY_CONFIG` |
| `PARTIAL_LOG_VISIBILITY` | one or more sites do not expose logs | `REQUIRES_USER_APPROVAL` |

`nvflare job logs --format json` should report availability per site:

```json
{
  "logs_truncated": true,
  "sites": {
    "site-1": {"available": true, "lines": 200},
    "site-2": {"available": false, "reason": "log_forwarding_disabled"},
    "site-3": {"available": false, "reason": "permission_denied"}
  }
}
```

Logs must be bounded. `job logs` should support `--tail`, `--since`, and
`--max-bytes`; the default without an explicit bound is at most 500 lines per
site. Diagnosis skills should request a bounded slice such as `--tail 200`.

User training errors and FLARE framework errors should be distinguishable in
logs. Exceptions caught around user training code should be prefixed
`[USER_CODE_EXCEPTION]`, while FLARE framework/runtime errors should be prefixed
`[FLARE]`. Findings should include `source: "user_code"`,
`"flare_framework"`, or `"unknown"`.

When logs are partial, diagnosis should warn that evidence is incomplete and
should not present a confident root cause for sites whose logs are unavailable.

### Workflow 6: Use Production Identity Safely

Agents should discover identities through the startup kit registry:

```bash
nvflare config kit list --format json
nvflare config kit show --format json
```

Agents should not treat `nvflare config kit use` as workflow state. It mutates
the user's global `~/.nvflare/config.conf`, so concurrent agents, Jupyter
notebooks, and shared machines can race. Stage 1 should add scoped identity
selection to `job`, `system`, `study`, and `agent doctor --online`, for example
`--kit-id <id>` or `--startup-kit <path>`. Skills should use that scoped
selector. `config kit use` remains a human convenience for setting a default
identity.

In production, agent workflows should use dedicated reduced-privilege startup
kits. Do not give an agent a project-admin kit unless the workflow truly needs
admin authority. Prefer member or lead roles for bounded job and study
workflows, and treat per-agent authorization scopes such as read-only or
study-scoped tokens as future platform work.

Production submission, shutdown, delete, distributed approval, and key-generation
flows require explicit user approval. Skills should surface approval checkpoints
instead of silently running production-impacting commands.

Approval checkpoints should require a confirm phrase for high-risk actions. For
example, production job submit can require the user to type
`submit to production` after the skill shows job folder, server, study, identity,
risk level, and whether the action is reversible.

These approval checkpoints are Stage 1 skill behavior, not platform-enforced
authorization gates. An agent or script can bypass them by calling the CLI
directly. Production-submit skills must state this limitation and recommend
dedicated reduced-privilege startup kits. Server-side approval enforcement is
future platform work.

Setup/deployment is its own skill category. `nvflare-setup-local` should cover
`pip install nvflare`, local doctor checks, POC prepare/start, and readiness
verification. Multi-machine, Docker, K8s, and cloud deployment skills should
guide environment setup and validation before job submission. They should list
hosts, ports, startup kits, container images, K8s artifacts, cloud networking,
storage, and human handoff points explicitly, rather than hiding those choices
inside a generic job workflow.

## Stage 1 Command Surfaces

| Surface | Stage 1 Role | Status |
| --- | --- | --- |
| `nvflare job ...` | Submit, list, wait, monitor, logs, stats, download, abort, clone, delete | Closed by PR #4479 after merge; add JSONL monitor terminal/timeout events, bounded logs, download artifact paths, and idempotency-key submit contract |
| `nvflare system ...` | Status, resources, version, shutdown/restart with safety gates | Closed by PR #4479 after merge |
| `nvflare study ...` | Study lifecycle and membership workflows | Closed by PR #4479 after merge |
| `nvflare poc ...` | Local POC prepare/start/stop/clean | Closed by PR #4479 after merge |
| `nvflare config kit ...` | Startup kit registration, inspection, and human default identity selection | Closed by PR #4479 after merge |
| scoped startup-kit selector | Per-command identity override for `job`, `system`, `study`, and online doctor | New Stage 1 work |
| `nvflare provision`, `nvflare cert`, `nvflare package` | Multi-machine and distributed provisioning inputs for deployment skills | Existing/in-progress CLI surfaces |
| `nvflare recipe show/list --filter` | Queryable recipe metadata for agent recipe selection | New Stage 1 work |
| `nvflare cert request-status` | Poll distributed provisioning request state | Follow-on distributed provisioning work |
| `nvflare agent skills install` | Install packaged skills from installed package | New Stage 1 work |
| `nvflare agent skills list` | Show packaged and installed skill versions/status | New Stage 1 work |
| `nvflare agent skills validate` | Validate installed skills against current NVFLARE version and command schemas | New Stage 1 work |
| `nvflare agent skills get` | Retrieve one skill on demand to reduce agent context load | New Stage 1 work |
| `nvflare agent skills revert` | Restore files from a skill-created `.nvflare_bak/<timestamp>/` backup | New Stage 1 work |
| `nvflare agent workflow plan` | Produce dependency-checked skill sequence for a user goal | New Stage 1 work |
| `nvflare agent inspect <path>` | Read-only local project/job inspection | New Stage 1 work |
| `nvflare agent doctor` | Local FLARE readiness and startup-kit registry check | New Stage 1 work |
| `nvflare agent doctor --online` | Bounded server/client readiness snapshot using the selected startup kit and optional explicit server address | New Stage 1 work |
| Recipe job convention/templates | Shared `job.py`, file layout, and README structure based on packaged `nvflare/agent/templates` | Stage 1 documentation and skill-template work |
| `nvflare cert renew` | Renew or replace expiring startup-kit certificates | Follow-on operational lifecycle work |

Distributed certificate requests need expiration metadata because approval can
take hours or days. `nvflare cert request --format json` should return
`request_dir`, `request_id`, `expires_at`, and a warning when approval must
happen before a deadline. `nvflare cert request-status --request-dir <path>
--format json` should return `status`, `expires_at`, and
`time_remaining_hours`. The distributed-provisioning skill stores these fields
in workflow state and emits `CERT_REQUEST_EXPIRING_SOON` when less than one hour
remains.

## Packaged Skills

Skills should ship in the installed package, for example:

```text
nvflare/agent/skills/
  nvflare-orient/SKILL.md
  nvflare-convert-client-api/SKILL.md
  nvflare-convert-pytorch/SKILL.md
  nvflare-migrate-from-distributed/SKILL.md
  nvflare-convert-lightning/SKILL.md
  nvflare-convert-huggingface/SKILL.md
  nvflare-convert-tensorflow/SKILL.md
  nvflare-convert-xgboost/SKILL.md
  nvflare-convert-sklearn/SKILL.md
  nvflare-convert-numpy/SKILL.md
  nvflare-generate-job/SKILL.md
  nvflare-local-validation/SKILL.md
  nvflare-setup-local/SKILL.md
  nvflare-poc-workflow/SKILL.md
  nvflare-deploy-multimachine/SKILL.md
  nvflare-deploy-docker/SKILL.md
  nvflare-deploy-k8s/SKILL.md
  nvflare-deploy-cloud/SKILL.md
  nvflare-identity-and-config/SKILL.md
  nvflare-job-lifecycle/SKILL.md
  nvflare-study-workflow/SKILL.md
  nvflare-system-operations/SKILL.md
  nvflare-federated-statistics/SKILL.md
  nvflare-diagnose-job/SKILL.md
  nvflare-distributed-provisioning/SKILL.md
  nvflare-production-submit/SKILL.md
  nvflare-privacy-security/SKILL.md
```

Skills should be tiered to avoid loading the entire bundle into an agent's
context:

| Tier | Skills | Default Loading |
| --- | --- | --- |
| Essential | orient, setup-local, job-lifecycle, poc-workflow, identity-and-config | Always |
| Conversion core | convert-client-api, generate-job, local-validation | When conversion intent is detected |
| Framework-specific | PyTorch, distributed migration, Lightning, TensorFlow/Keras, Hugging Face, XGBoost, scikit-learn, NumPy | When inspect selects the framework |
| Advanced | distributed provisioning, privacy/security, deployment, system operations | On explicit request |

Agents can fetch one skill on demand:

```bash
nvflare agent skills get --name nvflare-convert-pytorch --format markdown
```

Skills are maintained as release artifacts. Each skill has a maintainer, follows
the NVFLARE release cycle, and must be audited when a CLI schema it depends on
changes. Users can report skill bugs or install local overrides:

```bash
nvflare agent skills report-bug --skill nvflare-convert-pytorch --format json
nvflare agent skills install --local --skill nvflare-convert-pytorch ./my-skill/
```

Each skill should include frontmatter:

```yaml
---
skill_version: "1.0.0"
min_flare_version: "2.8.0"
max_flare_version: null
maintainer: "nvflare-cli"
status: active
tier: framework_specific
agent_requirements:
  python_execution: required
  file_editing: required
  shell_execution: required
  network_access: optional
cli_commands_used:
  - nvflare agent inspect
json_fields_used:
  - data.conversion_state
depends_on:
  - skill: nvflare-convert-client-api
    required_output: conversion_state
    required_values: [client_api_converted]
produces:
  - field: export_dir
    type: path
---
```

Each skill should include:

- when to use it.
- when not to use it.
- required files and assumptions.
- exact commands.
- JSON fields to extract.
- validation checklist.
- expected artifacts.
- safe edit boundaries.
- troubleshooting table.
- error recovery categories and default recovery behavior.
- approval checkpoints and confirm phrases for high-risk actions.
- dependencies and produced outputs for workflow routing.
- maintainer, status, and agent capability requirements.

`nvflare-orient` is stateful. It should begin by checking
`.nvflare_workflow_state.json` in the current project. If a state file exists,
the skill should summarize the workflow, current step, artifacts, and pending
approvals, then ask whether to resume or restart. If no state exists, it runs
the full discovery flow. Approved and denied checkpoints should also be stored
there so the next skill can continue without re-asking or silently skipping an
approval gate.

`nvflare agent workflow plan --goal "<goal>" --format json` should produce an
ordered skill sequence before execution. The plan validates `depends_on`,
`produces`, required inputs, approval checkpoints, and unresolved questions.

## `agent doctor`

`nvflare agent doctor --format json` is a local readiness command. It should not
require a running FLARE server.

Checks:

| Check | Failure Example |
| --- | --- |
| NVFLARE import/version | Python package not importable |
| CLI command registry | `nvflare job --schema` fails |
| Active startup kit config | no active kit, stale path, invalid kit |
| Optional dependency summary | missing framework dependencies needed by selected recipe |
| Packaged skill availability | skills missing from installed package |
| Skill installation status | installed skills absent or stale |
| Skill compatibility | `nvflare agent skills validate` reports version mismatch or missing command schema |
| POC workspace config | configured workspace missing or stale |

Output should be one JSON envelope with `findings` and `next_steps`.

`nvflare agent doctor --online --format json` is part of Stage 1. It should
reuse existing commands and collect a bounded snapshot:

- active startup kit.
- server connection/auth.
- server status.
- server and client versions.
- connected clients.
- resources.
- job list summary.
- study list summary when authorized.
- startup kit certificate expiration and warnings.
- authorized study roles and submit capability when available.
- startup-kit expiration details, status, and renewal command when supported.
- `snapshot_timestamp` and `snapshot_ttl_seconds`.

It must not submit, abort, delete, download, restart, shut down, change identity
selection, modify config, or read private keys.

Long-running monitor and diagnosis skills should re-run a lightweight online
doctor check periodically. Credential expiration, stopped POC processes, or
server restart should be reported as environment/config issues rather than
misclassified as generated-code failures.

Production-submit skills must re-run `agent doctor --online` immediately before
submit if more than `snapshot_ttl_seconds` has elapsed since the last snapshot.
They should also verify the target study appears in the study summary with
`can_submit_job: true`.

When expiration is near or already reached, doctor output should include:

```json
{
  "snapshot_timestamp": "2026-04-27T10:00:00Z",
  "snapshot_ttl_seconds": 60,
  "startup_kit_expiration": {
    "expires_at": "2026-06-27T00:00:00Z",
    "days_remaining": 61,
    "status": "ok",
    "renewal_command": "nvflare cert renew --kit-id <id> --format json"
  }
}
```

## `agent inspect`

`nvflare agent inspect <path> --format json` is a local static inspection helper.
One concrete Stage 1 use case is to tell an agent whether user training code has
already been converted to FLARE Client API or whether conversion is still needed.
This must work across common framework styles, including PyTorch, PyTorch
Lightning, TensorFlow/Keras, NumPy, scikit-learn, XGBoost, and other code that
can use the Client API receive/send pattern.

Output should include:

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "path": "./my_training_repo",
    "kind": "training_repo",
    "redaction": "on",
    "frameworks": [
      {
        "name": "pytorch",
        "confidence": 0.85,
        "supporting_evidence": [{"file": "train.py", "line": 3, "pattern": "import torch"}],
        "contradicting_evidence": []
      }
    ],
    "entrypoints": ["train.py"],
    "flare_usage": {
      "client_api": {
        "detected": false,
        "evidence": []
      },
      "recipe_api": false,
      "job_py": false,
      "sim_env": false,
      "export_supported": false
    },
    "conversion_state": "not_converted",
    "recipe_fit": {
      "assessment": "compatible",
      "evidence": ["standard train/eval loop detected"],
      "recommendation": "Use Recipe API"
    },
    "training_mode": {
      "multi_gpu": false,
      "framework": "single_process",
      "evidence": [],
      "fl_impact": null
    },
    "findings": [],
    "recommended_skills": [
      "nvflare-convert-pytorch",
      "nvflare-generate-job"
    ],
    "recommended_next_steps": [
      "Use nvflare-convert-pytorch skill",
      "Generate job.py with SimEnv and export support"
    ]
  }
}
```

Inspection should be conservative. If the command is unsure, it should return a
finding with a hint rather than guessing.

Inspect should flag hardcoded local data paths because they often pass local
simulation but fail on remote FL sites. Static checks should report
`ABSOLUTE_DATA_PATH` findings for Unix absolute paths, Windows absolute paths,
`Path.home()`, and `os.path.expanduser("~")` patterns.

The initial `kind` values should be:

| Value | Meaning |
| --- | --- |
| `training_repo` | Directory containing training source files |
| `training_script` | Single training script |
| `job_source` | Source directory containing `job.py` and job construction code |
| `exported_job` | Submit-ready exported job folder |
| `workspace` | Mixed project workspace where a narrower path should be inspected |
| `unknown` | Inspection cannot classify the target safely |

Stage 1 inspection must be static only. It may read files, scan text, parse
Python AST, inspect filenames/layout, and parse simple config or README content.
It must not import or execute user modules or framework packages. For example,
it must not run `import train`, `import torch`, or `import tensorflow` as part of
inspection. Importing user code can trigger data downloads, GPU allocation,
training startup, environment-specific failures, or other side effects. If static
inspection cannot determine a fact, return `unknown` with evidence and hints.

Inspection must not follow symlinks by default. At minimum, it must never follow
a symlink that resolves outside the inspected root. Symlinks should be reported
as findings such as `SYMLINK_SKIPPED` with the link name, sanitized target, and
action `skipped`.

Static inspection must also report dynamic patterns instead of guessing:
Hydra/OmegaConf instantiation, dynamic imports, `getattr` dispatch,
`torch.compile` wrappers, external training functions, and dynamic Client API
dispatch should produce `DYNAMIC_FRAMEWORK_RESOLUTION` findings. When confidence
is below threshold, the skill should ask the user to re-run with
`--confirm-framework <name>` before editing code.

Client API detection should look for static evidence such as:

- `import nvflare.client as flare` or equivalent imports.
- `flare.init()`.
- `flare.receive()`.
- `flare.send(...)`.
- `FLModel` usage.
- a loop driven by `flare.is_running()` when present.

The `conversion_state` values should be:

| Value | Meaning |
| --- | --- |
| `not_converted` | Framework code found, but no Client API evidence |
| `partial_client_api` | Some Client API calls found, but receive/send/init pattern is incomplete |
| `client_api_converted` | Client API init/receive/send pattern appears complete |
| `flare_job` | A `job.py` or job directory already defines FLARE job construction |
| `exported_job` | A submit-ready exported job folder exists and should be passed directly to `nvflare job submit -j` |
| `unknown` | Inspection cannot determine the state safely |

`recipe_fit.assessment` values:

| Value | Meaning |
| --- | --- |
| `compatible` | A standard recipe appears sufficient |
| `requires_fedjob` | Static evidence suggests ScriptRunner, FedJob, or custom Executor/Controller is needed |
| `uncertain` | Evidence is mixed; ask the user before generating `job.py` |

Framework detection should also be static and evidence-based. Stage 1 should
recognize at least these framework families:

| Framework Family | Static Evidence Examples | Typical Next Skill |
| --- | --- | --- |
| PyTorch | `import torch`, `torch.nn`, `torch.optim`, `DataLoader`, `state_dict` | `nvflare-convert-pytorch` |
| PyTorch Lightning | `lightning`, `pytorch_lightning`, `LightningModule`, `Trainer` | `nvflare-convert-lightning` |
| TensorFlow/Keras | `tensorflow`, `tf.keras`, `keras.Model`, `model.fit` | `nvflare-convert-tensorflow` |
| Hugging Face | `transformers`, `Trainer`, `AutoModel`, `AutoTokenizer`, `datasets` | `nvflare-convert-huggingface` |
| XGBoost | `xgboost`, `xgb.train`, `XGBClassifier`, `DMatrix` | `nvflare-convert-xgboost` |
| scikit-learn | `sklearn`, `fit(`, `predict(`, `Pipeline`, estimator imports | `nvflare-convert-sklearn` |
| NumPy/custom | `numpy`, custom loops, and no stronger PyTorch/Lightning/TensorFlow/Hugging Face/XGBoost/sklearn evidence | `nvflare-convert-client-api` |

If multiple frameworks are detected, `agent inspect` should return all of them
with evidence and confidence. The agent skill chooses the route; inspect should
not hide mixed-framework projects.
Framework detection should evaluate stronger framework evidence first. NumPy is
common inside PyTorch, TensorFlow, Hugging Face, XGBoost, and scikit-learn
projects, so NumPy/custom should be a fall-through classification only when no
stronger framework family is detected.
Confidence should be numeric from `0.0` to `1.0`, with supporting and
contradicting evidence. When evidence conflicts, the skill should ask the user
which framework or entrypoint is primary before editing code.

Inspect should also detect large-model and multi-GPU patterns:

| Pattern | Static Evidence | FL Impact |
| --- | --- | --- |
| DataParallel/DDP | `DataParallel`, `DistributedDataParallel`, `torch.distributed.init_process_group` | State dict keys may use `module.` prefixes; use unwrapped state for aggregation |
| FSDP/sharded model | `FullyShardedDataParallel`, `FSDP`, sharded checkpoint APIs | Full state dict gathering may be needed before sending model parameters |
| Accelerate | `accelerate.Accelerator` | State extraction should follow the framework wrapper's unwrap APIs |
| Gradient accumulation | `accumulation_steps`, delayed `optimizer.step()` patterns | Send model after optimizer update boundary, not per micro-batch |

Distributed-to-federated migration is separate from ordinary PyTorch
conversion. DDP/FSDP/Accelerate code already has distributed communication that
must be removed or isolated before adding FLARE Client API. When inspect detects
these wrappers, `training_mode.conversion_path` should recommend
`nvflare-migrate-from-distributed`, with `conversion_difficulty: "hard"` and
special concerns such as `module.` prefixes, FSDP full-state-dict gathering,
and removal of collective calls.

Framework-specific conversion skills must include exact code patterns before
they are released. PyTorch should show `load_state_dict`/`state_dict`,
optimizer-state guidance, DDP/DataParallel prefix handling, FSDP full-state-dict
handling, and metric collection. Lightning, Hugging Face, TensorFlow/Keras,
XGBoost, scikit-learn, and NumPy/custom skills should each include their
framework-native state extraction/application, metric collection, ordering
constraints, and a minimal before/after example.

## Error and Output Contract

Stage 1 commands follow the CLI contract from `nvflare_cli.md`:

- stdout in JSON mode contains exactly one JSON envelope for single-result
  commands.
- streaming commands use `--format jsonl`, one JSON object per line.
- command schema uses the normalized `--schema` JSON shape and declares
  arguments, flags, `streaming`, `output_modes`, `idempotent`,
  `idempotency_key_supported`, and `mutating`.
- stderr carries progress and human diagnostics.
- errors include stable code, message, hint, and `recovery_category`.
- invalid arguments exit with `INVALID_ARGS`.
- missing data such as unknown job logs exits with the correct data-not-found
  code, not a parser error.

New `agent` commands must follow the same contract from the first implementation.
The error-code registry should map every public code to one recovery category:
`RETRYABLE`, `FIXABLE_BY_CONFIG`, `FIXABLE_BY_CODE`,
`REQUIRES_USER_APPROVAL`, `ENVIRONMENT_FAILURE`, or `UNKNOWN`.

## Implementation Plan

### Phase 1: Agent CLI Skeleton

1. Add `nvflare agent` command dispatch and `--schema` tests.
2. Add shared JSON envelope handling for new agent commands.
3. Add scoped startup-kit selection design/tests for commands that need online
   identity.

### Phase 2: Agent Inspect

1. Add read-only `nvflare agent inspect <path> --format json`.
2. Implement basic classifiers:
   - Python training script.
   - Python package/repo.
   - FLARE job directory.
   - `job.py`.
3. Detect common frameworks and FLARE usage.
4. Detect standard example layout files such as `README.md`, `job.py`,
   `client.py`, `model.py`, `prepare_data.py`, and `download_data.py`.
5. Detect export support in `job.py`.
6. Detect `recipe_fit`, `training_mode`, and absolute data paths.
7. Add tests with small fixture repos and job folders.

### Phase 3: Agent Doctor

1. Add `nvflare agent doctor --format json`.
2. Implement local checks without server dependency.
3. Add skill compatibility checks through `nvflare agent skills validate`.
4. Add `--online` through existing CLI/shared helper calls after local doctor is
   stable.
5. Add tests for missing active kit, stale kit, valid POC kit, package skill
   availability, server unreachable, authentication failure, reachable POC
   server, server/client version reporting, and optional command authorization
   findings.

### Phase 4: Minimum Skill Bundle and Installer

1. Write `nvflare-orient` and `nvflare-convert-client-api` after `agent inspect`
   and local doctor exist.
2. Add `nvflare/agent/skills/`.
3. Add installer, list, and validate commands.
4. Use package resources to copy skills from installed package.
5. Support `--dry-run`, explicit target directory, validation, and backup before
   overwrite.
6. Add unit tests for install, backup, permission failure, version filtering,
   and validate output.

### Phase 5: Skill-Guided Conversion Examples

1. Document the common recipe `job.py` and template layout conventions from
   `nvflare/agent/templates`.
2. Use packaged `nvflare/agent/templates/hello-pt` as the first PyTorch
   conversion example for the initial useful skill bundle.
3. Ensure generated `job.py` follows the standardized
   `--export --export-dir` convention.
4. Add docs showing `SimEnv` first, then export, then CLI submit.
5. Add one POC end-to-end smoke example driven by CLI.

### Phase 6: Diagnosis Workflow

1. Add diagnosis skill that collects meta/logs/stats/system state.
2. Define a finding format used by skills and future `job diagnose`.
3. Add common failure patterns as a skill table first.
4. Promote to `nvflare job diagnose` only when patterns and data sources are
   stable enough.

## Tests and Validation

Stage 1 is complete when these scenario tests pass:

1. A user installs NVFLARE from a wheel and installs packaged skills without a
   repo checkout.
2. An agent runs `nvflare agent doctor --format json` and receives actionable
   findings.
3. An agent inspects a simple PyTorch training repo and selects the Client API
   conversion path.
4. An agent generates `job.py`, runs `SimEnv`, exports a job folder, and submits
   it through `nvflare job submit`.
5. An agent prepares POC, starts it, submits an exported job, waits or monitors
   with JSONL, reads logs/stats, downloads results, stops POC, and restores the
   prior kit.
6. An agent runs job/system/study commands with a scoped startup-kit selector
   and does not mutate the global active kit.
7. A deliberately failing job produces structured findings through the diagnosis
   skill.
8. Production-impacting actions require explicit user approval in the skill
   workflow.
