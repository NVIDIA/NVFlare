# NVFLARE Agent Consumer Readiness

## Document Control

| Field | Value |
| --- | --- |
| Original draft date | 2026-03-25 |
| Current design date | 2026-04-27 |
| Last updated | 2026-04-27 |
| Related CLI design | [nvflare_cli.md](nvflare_cli.md) |
| Agent foundation design | [nvflare_agent_foundation.md](nvflare_agent_foundation.md) |
| Stage 1 implementation plan | [nvflare_agent_foundation_implementation_plan.md](nvflare_agent_foundation_implementation_plan.md) |
| Related but separate goal | `nvflare_agentic_ai_readiness.md` |

## Problem Statement

NVFLARE is increasingly consumed through coding agents such as Claude, Codex, and
OpenClaw. These agents are not FLARE runtime participants. They are developer-side
automation systems that read code, modify training scripts, run local validation,
submit jobs, inspect results, and help users iterate.

The current CLI work makes FLARE scriptable, but "agent ready" requires more than
machine-readable command output. An agent needs to discover the FLARE surface,
choose the right integration path, convert existing deep learning code safely,
validate locally, run POC or production workflows, diagnose failures, and preserve
security boundaries without relying on an interactive human console.

This document defines what it means for NVFLARE to be ready for agent consumers,
what skills and tools are needed, when MCP is useful, what gaps remain, and how
to support agent-driven research workflows.

This is distinct from making NVFLARE itself an agentic runtime. In this document
the agent is the user of NVFLARE. NVFLARE's job is to be an excellent local tool
that agents can discover, understand, invoke, and verify without requiring a
human to read prose documentation during every step.

## Scope

In scope:

- External coding agents consuming NVFLARE as a local developer tool.
- Agent-assisted conversion of existing centralized ML/DL code into FLARE jobs.
- Agent-driven local validation through recipes, FedJob, SimEnv, POC, and the CLI.
- Agent-driven operational workflows through `nvflare job`, `nvflare system`,
  `nvflare study`, `nvflare cert`, `nvflare package`, and `nvflare config`.
- Skills, structured CLI contracts, optional MCP server, and supporting docs.
- Auto-research loops: generate experiment variants, run jobs, collect results,
  compare metrics, and produce a reproducible report.

Out of scope:

- Autonomous agents running inside FLARE server/client processes as training
  components.
- A generic one-click converter that rewrites arbitrary ML code without user
  review.
- Replacing Python APIs such as Client API, FedJob, Recipe API, or Session.
- Replacing the CLI with MCP. MCP is an additional transport over the same
  contracts.
- Agents handling production private keys or bypassing approval boundaries.

## Readiness Stages

Agent-consumer readiness has three progressive stages. Each stage builds on the
previous one; Stage 1 is the immediate product-readiness requirement.

| Stage | Name | Who Benefits | Description |
| --- | --- | --- | --- |
| 1 | [Agent-ready foundation](nvflare_agent_foundation.md) | Any data scientist using a coding agent | FLARE is easy to set up, inspect, run, and use for code conversion through CLI contracts and packaged skills |
| 2 | MCP integration | Workflows where CLI coverage is insufficient or stateful tool access matters | FLARE exposes typed tools/resources so agents do not need to generate Python glue for operations that lack a CLI |
| 3 | Auto-research | Research teams | Agents can plan, run, compare, and report FL experiment series using stable metrics/artifact contracts |

Stage 1 makes FLARE understandable to agents. Stage 2 makes FLARE natively
callable by agents. Stage 3 uses that foundation for long-running autonomous
research workflows.

### Motivating Prompts

| Stage | Example User Prompt | What FLARE Must Provide |
| --- | --- | --- |
| 1 | "Set up NVFLARE locally with 2 clients and convert this PyTorch training script to FedAvg." | Skills, examples, `SimEnv`, POC setup, exported job submission, structured diagnosis |
| 2 | "This workflow needs FLARE functionality that has no CLI yet; expose it to my agent without making it write Python glue." | Typed MCP tools over shared helper contracts, scoped identity discovery, progress updates |
| 3 | "Compare FedAvg and FedProx on this dataset across 5 simulated sites and write a results report." | Experiment plan/artifact conventions, metrics retrieval, run comparison, report inputs |

## What Agents Need from NVFLARE

For an agent to use NVFLARE reliably, the product must provide:

- discoverable entry points: a small, well-named set of CLI commands, skills,
  examples, and Python APIs.
- machine-readable command contracts: JSON output, schemas, exit codes, and
  stable error codes.
- guided examples: deterministic skills and before/after examples that agents
  can instantiate.
- explicit environment state: selected/default identity, startup kit, POC workspace,
  server/client status, versions, jobs, logs, and artifacts.
- feedback loops: local validation, server-side status, job logs/stats, and
  diagnosis commands before proceeding to the next step.
- long-running workflow support: resumable state, artifact manifests, and
  incremental metrics for research loops.

## Definition of Agent Ready

NVFLARE is agent ready when an agent can complete the following without using
the interactive admin console or scraping human-only output:

1. Discover the local FLARE version, available commands, schemas, recipes,
   examples, startup kit configuration, and current system state.
2. Inspect an existing training codebase and choose the lowest-risk FLARE
   integration path.
3. Generate or modify only the necessary user files: client training script,
   `job.py`, optional project/provisioning files, and local validation scripts.
4. Validate the job locally with import checks, simulation, and focused tests.
5. Prepare or connect to a POC/production environment.
6. Submit, monitor, inspect logs, download results, and diagnose failures.
7. Iterate on code or configuration based on structured feedback.
8. Produce a report that records commands, versions, artifacts, metrics, and
   remaining manual decisions.

Readiness is measured by whether this loop is reliable for agents, not only
whether a human can drive it manually.

## Agent Consumer Categories

Agent readiness should be organized around workflow categories, not around one
skill per CLI command.

| Category | Agent Goal | Typical Surfaces |
| --- | --- | --- |
| Setup and deployment | Install NVFLARE, choose local/POC/multi-machine/Docker/K8s/cloud path, verify prerequisites, and bring up a usable environment | `pip install nvflare`, `nvflare agent doctor`, `nvflare poc ...`, `nvflare provision`, Docker/K8s/CSP deployment docs and artifacts |
| Inspect and convert code | Read user training code, detect framework and conversion state, then guide Client API or recipe changes | `nvflare agent inspect`, Client API, Recipe/FedJob APIs, framework-specific examples |
| Generate, validate, and export jobs | Create `job.py`, run local `SimEnv`, and export a submit-ready job folder | `python job.py`, `python job.py --export --export-dir ...`, examples/templates |
| Identity, provisioning, and studies | Select the correct startup kit identity, request/package startup kits, and manage study context where authorized | `nvflare config kit ...`, `nvflare cert ...`, `nvflare package`, `nvflare study ...` |
| Job lifecycle and operations | Submit, monitor, inspect, abort/delete where approved, read logs/stats, and download artifacts | `nvflare job ...`, `nvflare system ...` |
| Diagnosis and troubleshooting | Collect evidence and produce actionable findings without scraping human output | `nvflare job meta/logs/stats/download`, `nvflare system status/resources/version`, diagnosis skills |
| Research loops | Compare runs, track metrics/artifacts, and report results | Future metrics/artifact manifests, experiment conventions, possible MCP/session tools |

The setup/deployment category is intentionally separate from job lifecycle.
Agents must first know whether they are working in local POC, multi-machine,
containerized, Kubernetes, or CSP infrastructure before they can safely submit
or diagnose jobs.

## Skill Contract

Skills are workflow playbooks, but they must not be only prose. Each packaged
skill should have a parseable contract in frontmatter so agents and tests can
understand inputs, outputs, approvals, and CLI dependencies without scraping
Markdown paragraphs.

Minimum frontmatter schema:

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
    required: true
outputs:
  - name: export_dir
    type: path
depends_on:
  - skill: nvflare-convert-client-api
    required_output: conversion_state
    required_values: [client_api_converted]
produces:
  - field: export_dir
    type: path
  - field: job_name
    type: string
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
json_fields_used:
  - inspect_result.conversion_state
  - job_submit.data.job_id
recovery_categories:
  - FIXABLE_BY_CODE
  - FIXABLE_BY_CONFIG
---
```

The Markdown body can still explain the workflow, but named steps should
reference the structured inputs and outputs. For example, a conversion skill can
say that `inspect_result.conversion_state == "not_converted"` proceeds to the
Client API patch step, while `partial_client_api` proceeds to completion checks.
This gives tests a way to lint skills against the installed `--schema` output
and prevents skill drift when CLI flags or fields change.

Skills should form a workflow state machine, not an unstructured list. The
`depends_on` and `produces` fields define which skill may run next and which
outputs satisfy downstream inputs. Approval checkpoints should be written to the
workflow state file with an approved/denied status so the next skill can consume
the approval without re-asking.

Stage 1 should add a planner command:

```bash
nvflare agent workflow plan --goal "<goal>" --format json
```

The command returns an ordered skill sequence, dependency checks, required
inputs, produced outputs, approval checkpoints, and unresolved questions before
the agent starts mutating files or submitting jobs.

`nvflare-orient` must check agent capabilities before recommending a skill. A
CLI-only agent should not be routed to a code-editing skill; a read-only agent
should not be routed to a conversion skill that edits `client.py` or `job.py`.
`nvflare agent skills list` should support capability filtering:

```bash
nvflare agent skills list --filter capability=cli_only --format json
```

## Skills Maintenance Policy

Packaged skills are part of the NVFLARE user contract and need explicit
ownership:

- each skill declares a maintainer team or GitHub handle in frontmatter.
- skill updates follow the NVFLARE release cycle.
- breaking CLI schema changes require a skill compatibility audit before
  release.
- community PRs can add or update framework-specific skills when they include
  examples and contract tests.
- deprecated framework variants, such as unsupported TensorFlow versions, should
  be marked with `status: deprecated` and a replacement hint.
- users can report issues through a command or issue template:

```bash
nvflare agent skills report-bug --skill nvflare-convert-pytorch --format json
```

Users should also be able to install a local override without modifying the
package-bundled skill:

```bash
nvflare agent skills install --local --skill nvflare-convert-tensorflow ./my-fixed-skill/
```

## Error Recovery Contract

Multi-step workflows need a shared recovery taxonomy. CLI errors already include
stable codes and hints; agent skills should map those codes into recovery
categories so agents know whether to retry, edit code, select a different
identity, ask the user, or stop.

| Recovery Category | Meaning | Default Agent Action |
| --- | --- | --- |
| `RETRYABLE` | Transient timeout, connection drop, or temporary server unavailability | Retry with bounded backoff, then re-run `agent doctor --online` |
| `FIXABLE_BY_CONFIG` | Wrong selected identity, stale startup kit, wrong study, missing local config | Adjust scoped identity/config or ask user which identity/study to use, then re-check |
| `FIXABLE_BY_CODE` | Bad generated code, missing import, wrong data path, invalid job folder | Re-inspect, patch/regenerate the affected files, and validate again |
| `REQUIRES_USER_APPROVAL` | Production action, irreversible action, key generation, distributed approval | Surface an approval checkpoint and wait |
| `ENVIRONMENT_FAILURE` | POC partially started, port conflict, GPU unavailable, disk/resource failure | Stop the workflow and report environment remediation steps |
| `UNKNOWN` | Error does not match a known category | Collect diagnosis data and ask for human review |

The `hint` field should remain human-readable. The JSON error envelope should
also include `recovery_category` as a first-class field so skills do not need to
parse hint text:

```json
{
  "schema_version": "1",
  "status": "error",
  "code": "JOB_FOLDER_NOT_FOUND",
  "message": "Job folder './jobs/hello-pt' not found.",
  "hint": "Run python job.py --export --export-dir ./jobs first.",
  "recovery_category": "FIXABLE_BY_CODE"
}
```

The CLI contract should publish an error-code to recovery-category mapping, and
contract tests should verify that every public error code has a category.

## Approval Checkpoints

Production-impacting or irreversible workflows need a structured approval
protocol. A skill should emit an approval checkpoint before continuing:

```json
{
  "checkpoint_type": "user_approval_required",
  "action": "production_job_submit",
  "details": {
    "job_folder": "./jobs/hello-pt",
    "server": "prod-flare.company.com",
    "study": "research_study",
    "identity": "alice@company.com"
  },
  "risk_level": "high",
  "reversible": false,
  "confirm_phrase": "submit to production"
}
```

The agent must surface the checkpoint and wait for the exact confirmation
phrase. Approval records should include timestamp, identity, action, risk level,
and the accepted confirm phrase. Skills can keep this audit in their workflow
state file; production systems may later add server-side audit integration.

## Design Principles

### Tool-First Architecture

An agent can use NVFLARE through three surfaces: CLI, MCP, and Python APIs. They
serve different use cases and should not be treated as competing replacements.

| Surface | How an Agent Uses It | Best Fit | Risk if Used as the Only Surface |
| --- | --- | --- | --- |
| CLI | Emits a command such as `nvflare job submit -j ./job --idempotency-key <uuid> --format json` and branches on a structured envelope | Short operations, CI, shell-capable agents, reproducible workflows | Repeated calls reopen sessions; long-running coordination is clumsy |
| MCP | Calls a typed tool such as `nvflare.job.submit` or a tool for functionality that has no CLI yet | Missing-CLI operations, long-running agent sessions, progress events, typed resources, policy gates | Must not become a divergent API contract |
| Python API | Generates or edits Python code using Client API, FedJob, Recipe API, SimEnv, or Session | User job code, local simulation, embedding FLARE in larger systems | High token cost and more failure points when agents must write one-off admin glue |

Decision: use CLI as the Stage 1 contract for short, stateless operations and
all common admin workflows. Deprioritize MCP for simple IDE convenience. Add MCP
only when it avoids forcing agents to generate Python glue for functionality
that has no CLI, or when session continuity, streaming progress, or shared
resources materially improve the workflow. Python APIs remain the right surface
for generated job code and local simulation, not the preferred path for simple
admin operations.

MCP should wrap the same CLI/shared-helper contracts. The output envelope, error
codes, and field names should stay equivalent between:

```bash
nvflare job submit -j ./job --idempotency-key <uuid> --format json
```

and a future MCP call such as:

```text
nvflare.job.submit(job_path="./job", idempotency_key="<uuid>")
```

The MCP server's added value is exposing missing-CLI capabilities as typed tools,
session management, progress events, resource browsing, and policy gates. It
should not introduce a second semantic model for job submission, status, logs,
config, studies, or system operations.

### 1. CLI Is the Baseline Contract

Every agent-capable workflow must be possible through the CLI with:

- `--format json` for exactly one JSON envelope on stdout for single-result
  commands.
- `--format jsonl` for streaming commands, where each line is one complete JSON
  event object.
- `--schema` for command discovery.
- stable error codes and exit codes.
- idempotency metadata in schemas so agents know whether retries are safe.
- no prompts in automation paths; use `--force`, `--confirm-*`, or explicit
  non-interactive alternatives.
- human progress and diagnostics on stderr in JSON mode.

Command `--schema` output is itself a stable JSON contract. It is not prose
help text and not raw argparse output. Minimum shape:

```json
{
  "schema_version": "1",
  "command": "nvflare job submit",
  "description": "Submit a job to the selected FLARE server.",
  "arguments": [
    {"name": "job_id", "type": "string", "required": true}
  ],
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

The field names above are the shared contract for skill linting,
`cli_commands_used` validation, CLI contract tests, and future MCP schema
derivation. A command may add command-specific metadata, but these fields must
remain stable.

The one-envelope rule does not apply to streaming commands such as progress
monitoring. Command schema output must declare `streaming: true|false`,
`output_modes`, and `idempotent: true|false`. For agents that need one final
result, provide a blocking single-result command such as:

```bash
nvflare job wait <job_id> --timeout 3600 --format json
```

For agents that need progress events:

```bash
nvflare job monitor <job_id> --timeout 3600 --format jsonl
```

`job monitor` exits when the job reaches a terminal state: `COMPLETED`,
`FAILED`, or `ABORTED`. It should also accept `--timeout <seconds>`. The final
JSONL event must include `terminal: true`; timeout exits should emit a final
event with `status: "TIMEOUT"`, `terminal: true`, and
`recovery_category: "RETRYABLE"` or a more specific category.

Non-idempotent commands that agents may need to retry must support an
idempotency key. For example:

```bash
nvflare job submit -j ./job --idempotency-key <uuid> --format json
```

If the server accepted the original submit but the client timed out, retrying
with the same key must return the existing accepted job rather than creating a
duplicate. Skills must not retry non-idempotent commands unless the command
supports and uses an idempotency key.

MCP tools must not invent a different contract. They should wrap the same command
or shared Python helper layer and return the same envelope shape.

Log commands must be bounded by default. `nvflare job logs` should support
`--tail`, `--since`, and `--max-bytes`; without an explicit bound it should
return at most 500 lines per site and set `logs_truncated: true` when capped.
Skills should call logs with an explicit bound, such as:

```bash
nvflare job logs <job_id> --site all --tail 200 --format json
```

Job download must return discoverable artifact paths. It should accept
`--output-dir <path>` with a default of `./<job_id>/` and return data like:

```json
{
  "download_path": "./job-abc123",
  "artifacts": {
    "global_model": "job-abc123/server/models/global_model.pt",
    "metrics_summary": "job-abc123/server/metrics_summary.json",
    "client_logs": {
      "site-1": "job-abc123/site-1/log.txt"
    }
  },
  "missing_artifacts": []
}
```

Agents must use `data.artifacts.*` from the JSON response rather than guessing
download paths.

### 2. Skills Are the Primary Agent UX

Agents need procedural knowledge, not just tool schemas. NVFLARE should ship
skills that explain when to use Client API, FedJob, Recipe API, POC, production
startup kits, and distributed provisioning.

Skills should be small, task-specific playbooks with:

- decision tree.
- file patterns to inspect.
- edits to make.
- commands to run.
- validation checklist.
- common failure diagnosis.
- safety boundaries.

Any skill that edits user files must be reversible and idempotent:

- Before editing, copy every file that will be modified into
  `.nvflare_bak/<timestamp>/`.
- The skill output records `backup_path`.
- Re-running a skill must not duplicate `flare.init()`, `flare.receive()`,
  `flare.send()`, imports, or config blocks already present.
- `nvflare agent skills revert --backup <backup_path> --format json` restores
  files from a backup created by a skill.

### 3. Prefer Low-Touch Conversion

For centralized training code, the preferred path is:

1. Keep the user's training logic mostly intact.
2. Add FLARE Client API receive/send boundaries.
3. Define job orchestration with Recipe API or FedJob.
4. Use SimEnv first, then POC, then production.

Agents should only use custom Controller/Executor code when Client API, Recipe,
or FedJob cannot express the workflow.

### 4. Make Environment, Identity, and State Explicit

Agent workflows fail when hidden state is required. FLARE should expose:

- active startup kit and identity.
- available startup kits.
- POC workspace and running services.
- FLARE version and optional dependency availability.
- command schemas.
- recipe catalog and recipe requirements.
- job state, logs, stats, and artifacts.

Agents must not depend on `nvflare config kit use` as their workflow state.
That command mutates `~/.nvflare/config.conf`, which is shared by every process
running under the same user. This is convenient for a human default identity,
but it is unsafe for concurrent agents, notebooks with non-sequential cells, and
shared developer machines.

Stage 1 should add a non-mutating startup-kit override for agent-facing admin
commands. The preferred shape is a per-command kit selector, for example
`--kit-id <id>` or `--startup-kit <path>`, supported by `job`, `system`, and
`study` commands and by `agent doctor --online`. A process-scoped environment
variable can be a secondary convenience for notebook/session workflows. `config
kit use` should remain available for human defaults, but skills should prefer
the scoped selector and should warn before mutating the global active kit.

File locking around config writes may still be useful for the registry, but it
does not solve the race between "check active kit" and "run next command." Agent
workflows need scoped identity at command execution time.

### 5. Preserve Security Boundaries

Agent support must not weaken FLARE's security model:

- never copy private keys into transfer artifacts.
- redact secrets from logs and JSON outputs.
- use dedicated reduced-privilege startup kits for agent workflows in
  production; do not hand an agent a broad admin kit when a member or lead role
  is sufficient.
- require explicit user approval for production submission, shutdown, deletion,
  distributed approval, and key generation flows.
- treat generated job code as user-reviewable source, not hidden agent state.
- keep site-local data paths local; do not assume global data visibility.

Approval checkpoints are skill-enforced workflow boundaries, not
platform-enforced gates. They are reliable only when agents follow the skills;
they can be bypassed when an agent or script calls the CLI directly. In Stage 1,
the mitigation is clear documentation, reduced-privilege agent startup kits, and
explicit production-submit warnings. Production environments that require
bypass prevention need future server-side approval enforcement.

Short-term authorization guidance:

- admin/project-admin identities can manage provisioning and high-impact system
  operations and should not be the default agent identity.
- lead identities can usually submit/manage jobs for authorized studies and are
  preferable for production job workflows when allowed.
- member identities are preferred for read-only or bounded job workflows.
- future work should add per-agent authorization scope, such as
  `agent_role: read_only`, command allowlists, and study-scoped tokens enforced
  below the skill layer.

### Agent Security Boundaries for Code Inspection

Agent-assisted conversion creates a specific privacy tension: federated learning
keeps raw training data at the sites, but an agent may read training code that
references data systems, credentials, service endpoints, and proprietary model
details. Stage 1 must make this boundary explicit.

`agent inspect` rules:

- never include raw file contents in JSON output.
- report structural facts, symbols, imports, call patterns, and sanitized
  evidence only.
- do not follow symlinks during directory traversal by default. At minimum, do
  not follow symlinks that resolve outside the inspected root.
- report skipped symlinks as findings, for example
  `SYMLINK_SKIPPED` with name, sanitized target, and action.
- redact string literal values that match secret, credential, token, password,
  cloud key, connection string, or private endpoint patterns.
- default to redaction in all JSON and skill outputs; support an explicit
  `--redact on|off` mode for local-only debugging.
- findings such as `ABSOLUTE_DATA_PATH` should report the pattern and location,
  not the full sensitive path unless redaction is disabled.

Conversion skills must include an explicit review step before export:
"Review generated code for hardcoded credentials, data paths, and private
endpoints." For highly sensitive workloads, docs should recommend running the
agent locally with a local model or within an approved enterprise environment
rather than sending source code to a cloud-hosted model.

## Agent Conversion Workflow

The main user story is: "Convert my deep learning training code to federated
learning with NVFLARE."

### Phase 1: Inspect

The agent inspects the user's codebase and produces an inventory:

- framework: PyTorch, Lightning, TensorFlow, Hugging Face, sklearn, XGBoost,
  NumPy, or custom.
- training entry point and CLI args.
- model construction and initial weights.
- optimizer and scheduler ownership.
- train/eval loop boundaries.
- checkpoint format.
- metrics/logging framework.
- data path assumptions.
- distributed/multi-GPU assumptions.
- privacy/security requirements.

Output should be a short conversion plan before edits.

### Phase 2: Choose Integration Lane

| Lane | Use When | Agent Action |
| --- | --- | --- |
| Client API + Recipe | Standard algorithm such as FedAvg, FedProx, FedOpt, Scaffold, or Ditto; normal train/eval loop; no custom controller logic | Patch script with `nvflare.client`, create `job.py` using a recipe, run with `SimEnv`, export for CLI submit |
| Client API + FedJob ScriptRunner | The script should run as a standalone client task, or the workflow needs custom job assembly while reusing existing training script behavior | Create `FedJob`, controller, model wrapper, and `ScriptRunner` |
| Lightning integration | User uses PyTorch Lightning Trainer | Apply Lightning patching pattern and recipe/job wrapper |
| Existing job template | User wants minimal config job and already follows template layout | Generate app/config files or adapt template |
| Custom Executor/Controller | Workflow has custom aggregation beyond weighted averaging, server-driven per-round hyperparameter control, non-model tasks, or P2P behavior | Generate explicit FLARE components with tests |
| Production-only package/provision | User already has a job folder and deployment | Use CLI job/system/study commands, avoid code conversion |

Agents should default to the first lane that can satisfy the workflow, but the
decision must be evidence-based. `agent inspect` should include a `recipe_fit`
assessment:

```json
{
  "recipe_fit": {
    "assessment": "compatible",
    "evidence": [
      "standard train/eval loop detected",
      "no custom controller or server-side aggregation logic found"
    ],
    "recommendation": "Use nvflare-generate-job with Recipe API"
  }
}
```

Allowed values are `compatible`, `requires_fedjob`, and `uncertain`. If the
assessment is `uncertain`, the skill should ask the user before editing code.

### Phase 3: Patch Training Script

For Client API conversion, the common patch is:

```python
import nvflare.client as flare

flare.init()
input_model = flare.receive()

# Convert input_model.params into the user's model weights.
# Run the original local training code.
# Convert trained weights and metrics into FLModel.

flare.send(flare.FLModel(params=new_params, metrics=metrics))
```

The agent must preserve the user's original local training path where possible.
If practical, keep a local-only entry point so the user can still run the script
outside FLARE.
Before any skill edits source files, it must create
`.nvflare_bak/<timestamp>/`, copy the original files there, include
`backup_path` in its output, and make the edit idempotent so repeated runs do
not duplicate FLARE imports or Client API calls.

Framework-specific conversion skills must provide exact code patterns, not only
the generic Client API shape:

- PyTorch: use `model.load_state_dict(input_model.params)` and
  `model.state_dict()`; handle optimizer state only when the local training
  loop owns resume semantics; document `DataParallel`/DDP `module.` prefixes.
- PyTorch Lightning: use callback or strategy integration around `Trainer`
  rather than assuming a hand-written loop.
- Hugging Face: use `TrainerCallback` or a custom loop as appropriate; for
  PEFT/LoRA, exchange adapter weights when that is the user's training target.
- TensorFlow/Keras: use `model.get_weights()` and `model.set_weights()` and do
  not recompile after setting received weights unless the original code requires
  it.
- scikit-learn: distinguish estimators that support `partial_fit` from
  coefficient-averaging patterns.

Each released framework skill needs a minimal working before/after example.

### Phase 4: Generate Job Definition

Preferred output is a `job.py` that can:

- run locally with `SimEnv`;
- export a job folder for CLI submission.

Canonical agent templates must come from the installed package, not from a
GitHub checkout:

```text
nvflare/agent/templates/
  hello-pt/
    job.py
    client.py
    model.py
    prepare_data.py
    README.md
```

The source-tree `examples/` directory can remain as developer reference, but
skills for released users should read templates through
`importlib.resources.files("nvflare.agent.templates")`.

The generated job should keep runtime configuration in CLI args or a small YAML
file, not hard-code local paths unless the user already did.

For automation, `job.py` should not use `PocEnv` or `ProdEnv` as the normal
submission path. Those environments are useful for human-driven Python workflows,
but agents should keep job construction separate from runtime submission:

```bash
python job.py --export --export-dir <job_folder>
nvflare job submit -j <job_folder>/<job_name> --format json
```

This separation gives agents a stable artifact to inspect, test, archive, and
submit to either POC or production with the same CLI command. Startup kit
selection, scoped identity, study routing, monitoring, logs, download, and
diagnosis stay in the `nvflare job` and `nvflare config kit` CLI contracts.

The exported folder should include `_export_manifest.json` so the agent can
verify the artifact before submission:

```json
{
  "schema_version": "1",
  "nvflare_version": "2.8.0",
  "export_timestamp": "2026-04-27T10:00:00Z",
  "source_job_py": {
    "path": "../../job.py",
    "sha256": "sha256:abc123"
  },
  "required_files": [
    "meta.json",
    "app/config/config_fed_client.json",
    "app/config/config_fed_server.json"
  ],
  "exported_by": "SimEnv",
  "poc_validated": false,
  "poc_validation": null
}
```

The exported folder should also include `job_fingerprint.json` for
reproducibility:

```json
{
  "schema_version": "1",
  "nvflare_version": "2.8.0",
  "python_version": "3.11.4",
  "recipe": "fedavg",
  "recipe_version": "2.8.0",
  "source_hash": "sha256:abc123",
  "framework_deps": {"torch": "2.2.0"},
  "export_timestamp": "2026-04-27T10:00:00Z"
}
```

`agent inspect` on an exported folder should read these files, classify the
target as `exported_job`, confirm required files exist, and warn when
`source_job_py.sha256` no longer matches the current source `job.py`.
`nvflare job submit` should eventually validate the export manifest before
accepting the folder. Production/study submission should warn when
`poc_validated` is false or missing.

### Phase 5: Validate Locally

Agent validation order:

1. syntax/import checks for modified files.
2. unit-level smoke test of model state conversion.
3. `python job.py` with `SimEnv` for a small number of clients/rounds.
4. `python job.py --export --export-dir <job_folder>`.
5. POC validation with `nvflare poc prepare/start/wait-ready`.

Minimum metrics convention for Stage 1:

- `python job.py` with `SimEnv` should document where metrics are written, even
  if that is TensorBoard, MLflow, W&B, stdout, or a generated JSON summary.
- generated examples should prefer a simple `metrics_summary.json` artifact
  when practical, with round-level metrics such as loss and accuracy. For FL
  correctness checks, include round start and end metrics when available.
- `nvflare job download` examples should show where to find metrics artifacts.

This is a convention, not a new metrics service. It lets agents report whether a
job actually trained, not only whether it exited successfully.
6. `nvflare job submit -j <job_folder>/<job_name> --idempotency-key <uuid>`,
   `job wait` for one final JSON result or `job monitor --format jsonl` for
   progress events, `job logs`, and `job download`.

SimEnv is necessary but not sufficient. It does not validate TLS/authentication,
startup kits, network timeouts, per-site resource limits, long-running client
process behavior, or real site-local data heterogeneity. Agents should treat
POC validation as required before production submission unless the user
explicitly accepts the risk. The `nvflare-local-validation` skill must explain
these limits and mark the export manifest's `poc_validated` field only after a
POC job has run successfully.

Agents should not submit to production until local and POC validation pass and
the user explicitly approves.

`nvflare-local-validation` must include FL-specific semantic checks after
SimEnv, not just exit-code checks. These are heuristic checks, not proof, but
they catch common broken conversions:

1. Verify the generated code follows receive -> apply -> train -> send ordering.
2. Compare round-start metrics across rounds. Round 2 starting loss should
   generally be no worse than round 1 starting loss for a stable toy example.
3. Compare round end to next-round start. A large discontinuity can indicate
   the received global model is ignored or applied at the wrong time.
4. Warn when metrics only show local loss decreasing but no global improvement
   across rounds.

Conversion skills should explicitly verify that the received model is applied
before any optimizer step in a round.

### Phase 6: Diagnose and Iterate

Agents need structured observability:

- job metadata and terminal status.
- server and client logs with bounded size and per-site availability metadata.
- stats and resource state.
- downloaded results.
- TensorBoard/MLflow/W&B pointers where available.
- clear mapping from error to likely code/config source.

The diagnosis path also needs an error-source boundary. Client-side user
training exceptions should be logged with a structured prefix such as
`[USER_CODE_EXCEPTION]`, while FLARE framework errors should use `[FLARE]`.
Diagnosis findings should include `source: "user_code"`,
`"flare_framework"`, or `"unknown"` rather than inferring source from a flat log
line.

This points to several gaps in the current CLI and docs.

## Skill Catalog

NVFLARE should ship Stage 1 skills under an installable skill bundle. The
catalog below also shows later-stage skills so the roadmap stays coherent.
Skill names are examples; exact names can follow the target agent's naming
convention.

Skills should not be one wrapper per CLI command. The CLI command schema is the
tool contract; skills are workflow playbooks that teach agents when to combine
commands, what to inspect, what safety checks to apply, and how to interpret
results.

| Skill | Stage | Purpose | Main CLI/API Surface |
| --- | --- | --- | --- |
| `nvflare-orient` | 1 | Explain FLARE concepts, choose simulator/POC/prod path, run doctor commands | `nvflare agent doctor`, `--schema`, docs/examples |
| `nvflare-convert-client-api` | 1 | Convert centralized script with minimal Client API changes | Client API |
| `nvflare-convert-pytorch` | 1 | PyTorch-specific state_dict, checkpoint, metrics, and recipe patterns | Client API, Recipe API |
| `nvflare-migrate-from-distributed` | 1+ | Migrate DDP/FSDP/Accelerate code toward a single-site training loop before FL conversion | PyTorch distributed APIs, Client API |
| `nvflare-convert-lightning` | 1+ | Lightning Trainer patching and FLARE logging patterns | Client API, Recipe API |
| `nvflare-convert-huggingface` | 1+ | Trainer/model checkpoint and PEFT/LoRA patterns | Client API, Recipe API |
| `nvflare-convert-tensorflow` | 1+ | TF/Keras model weight and recipe patterns | Client API, Recipe API |
| `nvflare-convert-xgboost` | 1+ | XGBoost Booster/DMatrix and tree-model exchange patterns | Client API, XGBoost examples |
| `nvflare-convert-sklearn` | 1+ | scikit-learn estimator, Pipeline, fit/predict, and metric patterns | Client API, sklearn examples |
| `nvflare-convert-numpy` | 1+ | NumPy/custom training loops and explicit parameter exchange patterns | Client API, NumPy recipe examples |
| `nvflare-generate-job` | 1 | Create `job.py`, choose Recipe/FedJob/ScriptRunner, run `SimEnv`, export job folder for CLI submit | Recipe/FedJob APIs, `python job.py --export --export-dir ...` |
| `nvflare-local-validation` | 1 | Run SimEnv, check FL receive/apply/train/send semantics, and interpret simulator outputs before server submission | `SimEnv`, local Python checks, `metrics_summary.json` |
| `nvflare-setup-local` | 1 | Install NVFLARE locally, verify Python/package state, prepare POC, and confirm readiness | `pip install nvflare`, `nvflare agent doctor`, `nvflare poc ...` |
| `nvflare-poc-workflow` | 1 | Prepare/start/stop/clean local POC and understand the POC startup kit identity | `nvflare poc ...`, `nvflare config kit show/list`, scoped kit selector |
| `nvflare-deploy-multimachine` | 1+ | Plan and validate a multi-machine deployment with generated startup kits and explicit operator steps | `nvflare provision`, distributed provisioning docs, `nvflare system ...` |
| `nvflare-deploy-docker` | 1+ | Run FLARE server/client/admin components with container images and mounted startup kits | Docker examples/docs, `nvflare system ...` |
| `nvflare-deploy-k8s` | 1+ | Generate or validate K8s deployment artifacts and check cluster readiness | Helm/K8s deployment docs and generated artifacts |
| `nvflare-deploy-cloud` | 2+ | Map FLARE deployment steps to cloud-provider environments and managed clusters | CSP-specific docs/templates; future CLI/MCP surfaces where available |
| `nvflare-identity-and-config` | 1 | Register, inspect, remove startup kits, explain human defaults, and choose scoped identity for job/system/study operations | `nvflare config kit ...`, scoped kit selector |
| `nvflare-job-lifecycle` | 1 | Submit, list, inspect, monitor, abort, clone, delete, download, and read logs/stats for jobs | `nvflare job ...` |
| `nvflare-study-workflow` | 1 | Register studies, manage sites/users, inspect study state, and handle role/authorization boundaries gracefully | `nvflare study ...` |
| `nvflare-system-operations` | 1 | Inspect and control system state with safe gates for production-impacting actions | `nvflare system ...` |
| `nvflare-federated-statistics` | 1+ | Generate, submit, and interpret federated statistics jobs for data discovery and pre-training validation | federated statistics examples/jobs, `nvflare job ...` |
| `nvflare-diagnose-job` | 1 | Collect meta/logs/stats/results and identify likely failure cause | `nvflare job meta/logs/stats/download`, `nvflare system ...` |
| `nvflare-production-submit` | 1+ | Checklist for production startup kit, fresh doctor snapshot, study preflight, approval, and submit | `nvflare config kit ...`, `nvflare study list`, `nvflare job submit` |
| `nvflare-distributed-provisioning` | 1+ | Async request/status/approve/package workflow with key-handling warnings | `nvflare cert request/request-status/approve`, `nvflare package` |
| `nvflare-privacy-security` | 1+ | Add filters, DP/HE references, policy checks, and risk notes | FLARE privacy/security APIs and config |
| `nvflare-auto-research` | 3 | Run experiment variants, collect metrics, compare, and report | Recipe/FedJob APIs, `SimEnv`, `nvflare job ...` |

Stage `1+` means useful for agent consumers but not required for the first
minimum Stage 1 bundle. `nvflare-auto-research` is listed for continuity with
the readiness roadmap, but it belongs to Stage 3 and is not a Stage 1
deliverable.

To control context overhead, skills should be tiered:

| Tier | Skills | Loaded by Default |
| --- | --- | --- |
| Essential | `nvflare-orient`, `nvflare-setup-local`, `nvflare-job-lifecycle`, `nvflare-poc-workflow`, `nvflare-identity-and-config` | Yes |
| Conversion core | `nvflare-convert-client-api`, `nvflare-generate-job`, `nvflare-local-validation` | When conversion intent is detected |
| Framework-specific | PyTorch, Lightning, TensorFlow/Keras, Hugging Face, XGBoost, scikit-learn, NumPy, distributed migration | When framework evidence selects it |
| Advanced | distributed provisioning, privacy/security, system operations, auto-research, deployment specializations | On explicit request |

Stage 1 should add on-demand skill retrieval:

```bash
nvflare agent skills get --name nvflare-convert-pytorch --format markdown
```

`nvflare-orient` should act as the router: run doctor/inspect, select the needed
tier 2/3 skill, and fetch only that skill instead of requiring every installed
skill to sit in the agent context.

POC job operations should not be modeled as a separate POC job API. After
`nvflare poc prepare/start`, the POC admin startup kit is available through the
startup-kit registry, so the same `nvflare job ...` commands apply. The POC
workflow skill should teach that environment setup and job lifecycle are
separate steps.

`nvflare poc start --format json` must report the actual bound server/admin
addresses, not just the requested defaults. If a default port is unavailable,
the response should include `port_conflict: true`, the default port, the actual
bound address, and a warning. The POC workflow skill must pass the actual
address into readiness checks, for example `nvflare agent doctor --online
--server <address> --format json`, instead of assuming the active kit still
points to the intended POC server.

If POC setup mutates the human default startup kit, it must preserve and restore
the previous default. `nvflare poc prepare --format json` should report the
prior kit id, and `nvflare poc stop --restore-kit --format json` should restore
it by default for agent workflows. Skills should also store `pre_poc_kit_id` in
`.nvflare_workflow_state.json` so they can recover if POC stop did not run
cleanly.

Identity and study selection should also be explicit skill topics. Agents should
use `nvflare config kit show/list` to understand registered startup kits, then
run `job`, `system`, `study`, and `agent doctor --online` with scoped identity
selection where identity matters. Study commands should rely on the selected
startup kit identity and server-side study authorization, not on mutating the
global active kit.
Study skills must treat authorization differences as normal runtime facts. If
the selected identity cannot list or mutate studies, the skill should report the
authorization finding and suggest selecting a different startup kit or asking a
study lead/project admin, instead of treating the whole workflow as a tool
failure.

`nvflare-production-submit` must preflight the target study before submit:

```bash
nvflare study list --format json
```

If the target study is present with `can_submit_job: true`, the skill can
continue. If it is absent or the identity cannot submit, the skill should emit a
structured finding before export/submit reaches the server-side failure path.

Setup and deployment skills are a separate category from job lifecycle skills.
They should help the agent choose and execute the environment path before job
operations:

- local installation and POC for first validation.
- multi-machine deployment with explicit server/client/admin roles.
- Docker deployment with startup-kit mounts and port checks.
- K8s deployment with generated artifacts and readiness checks.
- cloud-provider environments where provider-specific networking, storage,
  identity, and cluster setup matter.

These skills should not hide production approval boundaries. For multi-machine,
Docker, K8s, and CSP deployments, the skill should produce a concrete checklist
of files, hosts, ports, certificates/startup kits, and human handoff points
before running commands that affect remote systems.

Each skill should include:

- "when to use" and "when not to use".
- required local files and commands.
- output artifacts.
- safe edit boundaries.
- validation commands.
- troubleshooting table.

## Agent Items Moved Out of `nvflare_cli.md`

Earlier CLI design drafts included concrete agent commands and skill
installation directly in `nvflare_cli.md`. Those items belong in this
agent-readiness design instead. The CLI design should define the stable command
contracts; this document should define how agents consume those contracts.

| Old Item in `nvflare_cli.md` | Current Decision |
| --- | --- |
| `nvflare agent` as broad "bootstrap/context management" command | Narrow to concrete commands only, such as `nvflare agent doctor` and possibly `nvflare agent inspect`; do not create a broad overlapping agent control plane |
| `nvflare install-skills` | Move under the agent namespace as `nvflare agent skills install` and `nvflare agent skills list`; do not add a broad top-level install-skills command |
| skill compatibility checks | Add `nvflare agent skills validate [--skill <name>] --format json` so installed skills can be checked against the current NVFLARE version and command schemas |
| `agent.poll_interval` / `agent.job_timeout` config keys | Do not keep as core CLI config. Job monitor has explicit CLI flags; agent defaults can live in skills or future agent config if needed |
| `claude.skills_dir` / `openclaw.skills_dir` config keys | Treat as skill-installer target discovery, not core CLI configuration |
| `nvflare/agent/skills/*.md` | Keep and expand under the packaged skill bundle described in this document |

Preferred direction:

1. CLI command schemas are exposed by `--schema`.
2. MCP tools, if implemented, derive their schemas from the same CLI/shared
   helper contract.

## `nvflare agent inspect` Scope

`nvflare agent inspect` is a read-only inspection command for the user's local
work, not a way to inspect the installed NVFLARE package or fetch content from
GitHub.

The command implementation is provided by the installed `nvflare` package. The
target inspected by the command is an explicit local path supplied by the user:

```bash
nvflare agent inspect ./my_training_repo --format json
nvflare agent inspect ./jobs/hello-pt --format json
nvflare agent inspect ./job.py --format json
nvflare agent inspect ./my_training_repo --confirm-framework pytorch --redact on --format json
```

The inspected path may be:

- a normal ML code repository or training script before conversion.
- an NVFLARE job directory before submission.
- a submit-ready exported job folder produced by
  `python job.py --export --export-dir`.
- a generated `job.py` and exported job definition.
- a project-local workspace containing scripts, configs, and README files.

The command should report facts and findings that help an agent plan the next
step:

- framework and entrypoint detection.
- numeric framework confidence with supporting and contradicting evidence.
- whether the code already uses FLARE Client API, regardless of framework.
- whether the code appears to be plain training code, partially converted Client
  API code, complete Client API code, controller/workflow code, or an existing
  FLARE job source or exported job artifact.
- likely train/evaluate/data-loading functions.
- missing or suspicious FLARE job files.
- whether `job.py` supports `--export --export-dir`.
- whether `job.py` imports or constructs `SimEnv` for local validation.
- `recipe_fit` with `compatible`, `requires_fedjob`, or `uncertain`.
- multi-GPU or distributed-training patterns, such as
  `DistributedDataParallel`, `DataParallel`, FSDP, `torch.distributed`,
  `accelerate.Accelerator`, and gradient accumulation boundaries.
- high-severity `ABSOLUTE_DATA_PATH` findings for hardcoded local data paths,
  including Unix absolute paths, Windows absolute paths, `Path.home()`, and
  `os.path.expanduser("~")` patterns.
- dynamic framework-resolution findings, such as Hydra/OmegaConf
  instantiation, dynamic imports, `getattr` dispatch, external training
  functions, or `torch.compile` wrappers that make static detection incomplete.
- suggested next commands, such as export, submit, monitor, logs, or diagnose.

An exported submit-ready job folder should be classified distinctly from source
job code. Its `conversion_state` should be `exported_job`, and the primary next
step should be `nvflare job submit -j <folder> --format json`, not generation of
a new `job.py`.

Inspect output should use `kind` for the target type, `entrypoints` for likely
training entry files, `recommended_skills` for skill names, and
`recommended_next_steps` for human/agent-readable actions. `recommended_next_steps`
is the superset; it may include skill usage, export commands, submit commands,
or diagnosis commands.

Framework detection should return a ranked list, not a single winner. Each
framework entry should include numeric `confidence`, `supporting_evidence`, and
`contradicting_evidence`. Mixed-framework projects are common; the agent should
ask the user when confidence is low or evidence conflicts.

When distributed or multi-GPU training is detected, inspect output should include
`training_mode`:

```json
{
  "training_mode": {
    "multi_gpu": true,
    "distributed": true,
    "wrapper": "DDP",
    "conversion_path": "nvflare-migrate-from-distributed",
    "conversion_difficulty": "hard",
    "evidence": ["torch.nn.parallel.DistributedDataParallel"],
    "special_concerns": ["module-prefix in state_dict", "remove all_reduce calls"],
    "fl_impact": "Distributed and federated training are architecturally different; remove collective communication before adding FLARE Client API."
  }
}
```

If dynamic patterns reduce confidence, inspect should return a warning finding
instead of guessing:

```json
{
  "code": "DYNAMIC_FRAMEWORK_RESOLUTION",
  "severity": "warning",
  "message": "Code uses dynamic framework resolution; static detection may be incomplete.",
  "recommendation": "Re-run inspect with --confirm-framework pytorch|tensorflow|huggingface."
}
```

If symlinks are encountered, inspect should skip them by default and report:

```json
{
  "code": "SYMLINK_SKIPPED",
  "severity": "info",
  "name": "data",
  "target": "<outside-target-redacted>",
  "action": "skipped"
}
```

The command should not:

- require a GitHub checkout of NVFLARE.
- scan arbitrary home-directory content unless explicitly pointed there.
- read private key contents or include secrets in JSON output.
- modify user code, generate files, submit jobs, or start/stop systems.
- replace full static analysis or framework-specific conversion skills.

This keeps the ownership clear:

- packaged skills and command implementation come from `pip install nvflare`.
- `agent inspect` reads the user's local target path.
- deeper conversion remains skill-guided agent work using the inspection result,
  examples, recipes, and CLI schemas.

## Agent Environment Snapshot

Earlier discussions also included an agent command that connects to the active
FLARE environment and tells the agent "what is true right now." This is a
different scope from `agent inspect`.

Recommended split:

| Command | Scope | Connects to Server? | Purpose |
| --- | --- | --- | --- |
| `nvflare agent inspect <path>` | User's local code or job directory | No | Static inspection before conversion, export, or submit |
| `nvflare agent doctor --format json` | Local NVFLARE installation and active startup kit | Optional | Readiness check with actionable errors and hints |
| `nvflare agent doctor --online --format json` | Active FLARE environment | Yes | Stage 1 online readiness snapshot with latest server status, server/client versions, resources, connected sites, and authorization checks |
| scoped startup-kit selection | Per-command `--kit-id <id>` or `--startup-kit <path>` on `job`/`system`/`study`/online doctor | Yes | Avoid races caused by global `config kit use` in concurrent agent and notebook workflows |

`nvflare agent doctor --online --format json` is a Stage 1 requirement. It
should wrap existing CLI/API operations, not define a separate status protocol.
It should run the local doctor checks first, then connect with the active startup
kit identity and collect a bounded environment snapshot.

The online snapshot is point-in-time data. Output must include a timestamp and
recommended TTL, and production-submit skills must re-run online doctor when the
snapshot is stale before submitting:

```json
{
  "snapshot_timestamp": "2026-04-27T10:00:00Z",
  "snapshot_ttl_seconds": 60
}
```

Concrete checks:

| Check | Source | Required? | Failure Handling |
| --- | --- | --- | --- |
| Active startup kit | `nvflare config kit show --format json` | Yes | Return `STARTUP_KIT_MISSING` or stale-path finding |
| Server connection/auth | shared session creation | Yes | Return structured connection/auth finding |
| Server status | `nvflare system status --format json` | Yes | Return finding; continue only if session remains usable |
| Server/client versions | `nvflare system version --format json` | Yes | Return finding if server version is unavailable; warn if client versions are unavailable or inconsistent |
| Startup kit validity | active startup kit metadata and certificate dates | Yes | Return `STARTUP_KIT_EXPIRED` or expiration warning before long-running work |
| Resource snapshot | `nvflare system resources --format json` | No | Include warning if unauthorized or unsupported |
| Job snapshot | `nvflare job list --format json` | No | Include warning if unauthorized or unavailable |
| Study snapshot | `nvflare study list --format json` | No | Include authorized studies with role/capability details when available; otherwise report authorization findings |

The output should be one JSON envelope:

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "mode": "online",
    "snapshot_timestamp": "2026-04-27T10:00:00Z",
    "snapshot_ttl_seconds": 60,
    "active_startup_kit": {
      "id": "lead@nvidia.com",
      "path": "/secure/startup_kits/example_project/lead@nvidia.com",
      "identity": "lead@nvidia.com",
      "cert_role": "lead",
      "org": "nvidia",
      "project": "example_project",
      "startup_kit_expiration": {
        "expires_at": "2026-06-27T00:00:00Z",
        "days_remaining": 61,
        "status": "ok",
        "renewal_command": "nvflare cert renew --kit-id lead@nvidia.com --format json"
      }
    },
    "server": {
      "reachable": true,
      "status": "running",
      "version": "2.x"
    },
    "clients": [
      {"name": "site-1", "connected": true, "version": "2.x"},
      {"name": "site-2", "connected": true, "version": "2.x"}
    ],
    "version_skew": [],
    "resources": {
      "available": true,
      "summary": {}
    },
    "jobs": {
      "available": true,
      "running": 1,
      "recent": []
    },
    "studies": {
      "available": true,
      "items": [
        {
          "name": "research_study",
          "role": "lead",
          "can_submit_job": true
        },
        {
          "name": "pilot_study",
          "role": "member",
          "can_submit_job": false,
          "reason": "authorization_required"
        }
      ]
    },
    "capabilities": {
      "can_submit_job": true,
      "can_view_system": true,
      "can_manage_studies": false
    },
    "findings": [],
    "next_steps": [
      "nvflare job submit -j <job_folder> --idempotency-key <uuid> --format json",
      "nvflare job list --format json"
    ]
  }
}
```

Partial failures should not automatically make the whole command fail. If the
active startup kit is valid and the server is reachable, unauthorized optional
checks should appear as findings with `severity`, `code`, `message`, and `hint`.
The command should exit nonzero only when the agent cannot establish the base
environment, such as missing startup kit, invalid startup kit, server
unreachable, authentication failure, or malformed server response.

In the concrete check table above, this means a failed check with
`Required? = Yes` can make the command fail, while a failed check with
`Required? = No` should be reported as a finding when the base session remains
usable.

The command must not:

- submit, abort, delete, download, restart, or shut down anything.
- modify `~/.nvflare/config.conf`.
- switch startup kit identity.
- expose private key contents or tokens.
- scan local project code; that belongs to `agent inspect <path>`.

The online snapshot should remain part of `agent doctor --online` for Stage 1.
Do not add a separate `agent context` or `agent describe` command unless future
usage shows the snapshot has grown beyond readiness checking.

Long-running monitoring and diagnosis skills should periodically run a
lightweight online doctor check. If credentials expire, the server restarts with
different configuration, or POC is stopped, the skill should surface the
environment or credential issue early instead of reporting it as a job-code
failure.

Job recovery after session loss depends on idempotency state. The agent must
generate the idempotency key and write it to `.nvflare_workflow_state.json`
before calling `job submit`. `job list` and `job meta` should include the
idempotency key, and `job list --idempotency-key <uuid> --format json` should
allow recovery when the submit response was lost.

When certificate expiration is warning or expired, the finding should include
the renewal command when supported:

```bash
nvflare cert renew --kit-id <id> [--extend-days 90] --format json
```

`nvflare-distributed-provisioning` should treat certificate expiration
monitoring and renewal as part of the lifecycle, not only as initial setup.

Distributed provisioning approval is asynchronous. The site-side skill must not
assume the lead approves immediately. After request creation, write workflow
state:

```json
{
  "provisioning_step": "awaiting_approval",
  "request_dir": "./hospital-1",
  "request_id": "<uuid>",
  "submitted_at": "2026-04-27T10:00:00Z",
  "expires_at": "2026-04-28T10:00:00Z"
}
```

Stage 1+ should add a status command:

```bash
nvflare cert request-status --request-dir <path> --format json
```

It returns `pending`, `approved`, `rejected`, or `expired`, plus `request_id`,
`expires_at`, and `time_remaining_hours`. The site-side skill should poll or
resume from workflow state and should not run `cert approve`; approval is the
lead's action. If less than one hour remains, emit
`CERT_REQUEST_EXPIRING_SOON` and tell the user to contact the lead or regenerate
the request if it expires.

## Jupyter Notebook Path

Many data scientists will use NVFLARE from notebooks while an agent edits or
drives cells. Stage 1 must document the notebook-safe path because notebooks
make hidden global state and blocking monitors more dangerous.

Notebook guidance:

- avoid `nvflare config kit use` inside notebooks for agent workflows; prefer
  per-command `--kit-id <id>` or a process-scoped startup-kit override once
  implemented.
- verify the notebook kernel and shell command use the same NVFLARE
  installation with `nvflare --version` before running workflow commands.
- use polling commands such as `nvflare job meta <job_id> --format json` or
  short bounded monitor intervals instead of blocking the kernel indefinitely
  with a long `job monitor`.
- after kernel restart, run `nvflare agent doctor --online --format json`
  before re-running `poc prepare` or switching identity; POC processes may still
  be running from the previous session.
- keep workflow state in `.nvflare_workflow_state.json` so a restarted notebook
  or a different agent can resume without rediscovering every artifact.

A Stage 1 example notebook should demonstrate safe POC validation, scoped
startup-kit selection, export, submit, bounded monitoring, logs, and download.

## Required CLI Tools and Gaps

### Existing or In-Progress CLI Surface

| Area | Command Surface | Agent Value |
| --- | --- | --- |
| Discovery | `--schema`, `--format json`, `nvflare recipe list` | Tool discovery and planning |
| Local setup | `nvflare poc prepare/start/stop/clean` | Local runtime for validation |
| Install/setup/deploy | `pip install nvflare`, `nvflare agent doctor`, `nvflare provision`, Docker/K8s deployment artifacts | Environment bring-up before job operations |
| Job lifecycle | `nvflare job submit/list/meta/monitor/download/abort/delete/clone` | End-to-end execution |
| Observability | `nvflare job logs`, `job stats`, `system status/resources/version` | Diagnosis |
| Identity/config | `nvflare config kit`, `cert`, `package`, `provision` | Environment setup |
| Studies | `nvflare study ...` | Multi-study workflows |

Assuming PR #4479 merges, the core admin CLI baseline for agent consumption is
considered closed for the `job`, `system`, `study`, `poc`, and `config kit`
surfaces. Remaining gaps below are agent-readiness additions on top of that
baseline, not missing replacements for those CLI commands.

### Gaps to Close

| Gap | Current Status | Target Status | Why It Matters | Proposed Surface |
| --- | --- | --- | --- | --- |
| Environment doctor | Not implemented | Ready | Agents need one command to learn whether FLARE is usable | `nvflare agent doctor --format json` |
| Online environment snapshot | Not implemented | Ready | Agents need one bounded server/client status snapshot | `nvflare agent doctor --online --format json` |
| POC readiness wait | Not implemented | Ready | Agents need deterministic readiness after background POC start | `nvflare poc wait-ready --timeout <seconds> --format json`, or equivalent blocking `poc start --format json` behavior |
| POC kit restore | Not implemented | Ready | POC workflows must not leave users on a stopped POC admin kit | `poc prepare` records prior kit; `nvflare poc stop --restore-kit --format json`; workflow state records `pre_poc_kit_id` |
| Recipe details | Partial: `recipe list` | Ready | `recipe list` is not enough for generation | `nvflare recipe show <name> --format json` |
| Recipe filtering | Not implemented | Ready | Agents need queryable recipe selection, not keyword search | `nvflare recipe list --filter framework=pytorch --filter privacy=differential_privacy --format json` |
| Study preflight | Partial | Ready | Production submit should fail before submit if study/role is wrong | `nvflare study list --format json` with `can_submit_job` |
| Job scaffold | Convention exists in source-tree examples | Ready | Agents need a known starting point for `job.py`, client script, support files, and README without cloning GitHub | Packaged `nvflare/agent/templates` recipe job templates following the common `README.md`, `job.py`, `client.py`, `model.py`, `prepare_data.py`, `download_data.py` structure; optional future `nvflare recipe scaffold` can instantiate the same templates |
| Conversion lint | Not implemented | Ready | Agents need feedback before running expensive jobs | `nvflare agent inspect <path> --format json` |
| Inspect symlink safety | Not implemented | Ready | Static inspection must not traverse outside target root | skip symlinks by default and report `SYMLINK_SKIPPED` |
| Export convention | Design convention | Ready | Agents need a uniform way to export generated jobs | standard `python job.py --export --export-dir <job_folder>` convention |
| Job diagnosis | Not implemented | Partial | Agents need one structured collection step | `nvflare job diagnose <job_id> --format json` |
| Partial log visibility | Not implemented | Ready | Multi-org production may not expose every site log | `job logs --format json` includes per-site availability and reason |
| Bounded logs | Not implemented | Ready | Unbounded multi-site logs can exceed agent context | `job logs --tail`, `--since`, `--max-bytes`, default 500 lines/site, `logs_truncated` |
| Job download artifact contract | Partial | Ready | Agents need returned paths to trained model, metrics, and logs | `job download --output-dir <path>` returns `download_path`, `artifacts`, and `missing_artifacts` |
| Metrics retrieval | Not implemented | Partial | Agents need at least a documented metrics artifact convention for conversion validation | `metrics_summary.json` convention first; future `nvflare job metrics <job_id> --format json` |
| FL semantic validation | Not implemented | Ready | Exit code 0 does not prove protocol correctness | local-validation skill checks receive/apply/train/send ordering and round-start/end metrics |
| Export manifest/fingerprint | Not implemented | Ready | Agents need to know an exported job folder is complete, fresh, and reproducible | `_export_manifest.json`, `job_fingerprint.json`, and `agent inspect` validation |
| Scoped identity selection | Not implemented | Ready | Agents and notebooks must avoid racing on global active-kit state | per-command `--kit-id <id>` or `--startup-kit <path>` on admin commands and online doctor; process-scoped environment override as secondary path |
| Skill runtime validation | Not implemented | Ready | Installed skills can drift from the current NVFLARE command surface | `nvflare agent skills validate [--skill <name>] --format json`; doctor summarizes compatible/stale/broken skills |
| Skill edit backup/revert | Not implemented | Ready | Conversion skills modify user files and need a recovery path | `.nvflare_bak/<timestamp>/`, `backup_path`, idempotent edits, and `nvflare agent skills revert --backup <path>` |
| Workflow continuity | Not implemented | Ready for orient | Agents may be interrupted mid-workflow | `.nvflare_workflow_state.json` owned by skills; `nvflare-orient` must read it before starting a fresh plan |
| Example index | Partial: examples exist as files/docs | Partial | Agents need framework-specific examples without web search | `nvflare examples list/show` or packaged docs resource |
| Skill installer/listing | Stub exists | Ready | Released users should install packaged skills without cloning GitHub and verify versions/status later | `nvflare agent skills install`, `nvflare agent skills list` |
| Skill on-demand loading | Not implemented | Ready | Agents should not load every skill into context | `nvflare agent skills get --name <skill> --format markdown` |
| Workflow planning | Not implemented | Ready | Agents need dependency-checked skill routing | `nvflare agent workflow plan --goal "<goal>" --format json` |
| Certificate renewal | Not implemented | Partial | Doctor can warn about expiration but needs an actionable path | `nvflare cert renew --kit-id <id> [--extend-days 90] --format json` |
| Async distributed approval | Not implemented | Partial | Site and lead actions are separated by hours/days | `nvflare cert request-status --request-dir <path> --format json` and workflow state |

## MCP Server

### Do We Need MCP?

Not as the first requirement. The CLI is the required baseline because it works
in shells, CI, notebooks, and all coding agents. MCP should be lowest priority
for workflows already covered cleanly by CLI. MCP becomes valuable when the
alternative is asking the agent to write Python glue against lower-level APIs,
or when a workflow needs stateful/streaming behavior that repeated CLI calls do
poorly.

MCP is useful for:

- typed access to capabilities that do not yet have a stable CLI.
- reducing agent-generated Python glue for operational workflows.
- long-running job monitor operations with progress events.
- structured resources such as recipe docs, examples, schemas, and active config.
- safer file/resource access with explicit tool boundaries.
- local policy gates for production submission, shutdown, approval, and deletes.
- multi-step sessions that keep connection context without repeating shell args.

MCP is not required to make a single `job submit`, `job list`, `job logs`, or
`system status` operation agent-usable. It becomes important for Stage 3
auto-research, where the agent may coordinate multiple runs over hours, recover
from tool-call failures, and stream progress without repeatedly reconnecting.

| Dimension | CLI | MCP |
| --- | --- | --- |
| Session lifetime | Stateless command execution; each operation resolves active kit and connects as needed | MCP server can hold or pool sessions and reconnect internally |
| Multi-agent coordination | Each agent/subagent shells out independently | Multiple agents/subagents can share the MCP server policy and connection layer |
| Experiment continuity | Agent re-discovers job state through CLI calls | MCP server can expose durable experiment context and subscriptions |
| Metric/log streaming | Poll with repeated CLI calls | Stream or subscribe through a long-lived local tool context |
| Best fit | Short operations, CI, reproducible shell workflows, all covered admin commands | Missing-CLI capabilities, long-running research loops, typed resources, policy-gated operations |

MCP is not a substitute for:

- CLI contract hardening.
- Python APIs.
- user review of generated code.
- production authorization.

### When to Use CLI vs MCP

| Situation | Prefer CLI | Prefer MCP |
| --- | --- | --- |
| One-shot job submit/list/log command | Yes | No |
| CI/CD workflow | Yes | No |
| Agent does not support MCP | Yes | No |
| Functionality already has a stable CLI | Yes | No |
| Agent would otherwise write Python admin glue | Possible | Yes |
| Long-running job monitor with streaming updates | Possible | Yes |
| Structured docs/examples/resource browsing | Possible | Yes |
| Production-destructive actions needing policy gates | Possible | Yes |
| Debugging CLI itself | Yes | No |

Rule: implement a workflow in CLI first. Add MCP only when the CLI contract is
stable and a richer agent transport provides real value.

### MCP Architecture

The MCP server should be a local process, for example `nvflare mcp serve` or a
separate `nvflare-mcp` entry point.

It should not call hidden, divergent logic. Instead:

```text
Agent
  -> MCP tool
    -> shared NVFLARE command/service helper
      -> Session / Recipe / FedJob / CLI helper
        -> FLARE runtime or local files
```

The CLI and MCP tools should share helper modules wherever possible. If a
workflow still exists only as a CLI handler with output side effects, refactor
the core logic into a pure helper first.

### Candidate MCP Tools

| MCP Tool | Backing Surface |
| --- | --- |
| `nvflare.version` | `nvflare --version`, package metadata |
| `nvflare.commands.schema` | `--schema` |
| `nvflare.agent.doctor` | new doctor helper |
| `nvflare.recipes.list` | `nvflare recipe list --format json` |
| `nvflare.recipes.show` | proposed recipe detail helper |
| `nvflare.job.submit` | `nvflare job submit` |
| `nvflare.job.monitor` | `nvflare job monitor --format jsonl` / Session API |
| `nvflare.job.wait` | `nvflare job wait --format json` |
| `nvflare.job.logs` | `nvflare job logs` |
| `nvflare.job.diagnose` | proposed diagnosis helper |
| `nvflare.poc.prepare/start/stop/clean` | `nvflare poc` |
| `nvflare.system.status/resources/version` | `nvflare system` |
| `nvflare.config.kit.*` | `nvflare config kit` |
| `nvflare.cert.request/approve` | `nvflare cert` |
| `nvflare.package.signed_zip` | `nvflare package` |

### Candidate MCP Resources

| Resource | Purpose |
| --- | --- |
| `nvflare://docs/index` | Agent-readable docs index |
| `nvflare://examples/index` | Framework/example catalog |
| `nvflare://recipes/index` | Recipe metadata |
| `nvflare://schemas/<command>` | Command schemas |
| `nvflare://config/startup-kits` | Active kit and registered identities |
| `nvflare://poc/status` | POC workspace and process state |
| `nvflare://jobs/<job_id>/manifest` | Job meta, status, artifacts |

### MCP Contract Tests

If MCP is implemented, CLI/MCP equivalence tests should become a release gate
for mirrored operations. A test should call the CLI and MCP path for the same
operation and assert the shared envelope fields, status/code semantics, and key
data fields are equivalent. This is how "same contract" stays enforceable
rather than aspirational.

### MCP Security

- Bind to local stdio by default, not a network port.
- Do not expose private key contents through resources.
- Redact tokens, cert private paths when not needed, and environment secrets.
- Destructive tools require explicit confirmation inputs.
- Production submit/delete/shutdown/approve actions require an explicit
  `approved=true` parameter or equivalent host-agent confirmation.
- All tool calls should return structured audit metadata.

## Auto-Research With FLARE

Agentic research means an agent can propose experiment variants, run them,
collect evidence, compare results, and produce a reproducible summary.

### Research Loop

1. User states objective and constraints.
2. Agent chooses base recipe/example.
3. Agent generates experiment plan with variables and budget.
4. Agent runs local smoke test.
5. Agent runs simulation/POC experiments.
6. Agent collects metrics, logs, artifacts, and environment metadata.
7. Agent compares results and updates the plan.
8. Agent writes a report with commands and artifact references.

### Required Capabilities

| Capability | Current State | Gap |
| --- | --- | --- |
| Recipe catalog | Exists through recipes and `recipe list` | Need richer recipe metadata and examples |
| Local simulation | Exists through SimEnv/FedJob | Need consistent `job.py` run/export/report convention |
| POC execution | CLI exists | Need doctor and diagnosis workflow |
| Metrics collection | Distributed across logs/TB/MLflow/W&B/results | Need `job metrics` or artifact manifest |
| Experiment lineage | User-managed | Need standard experiment manifest |
| Sweeps | User scripts | Need lightweight plan/run/compare convention |
| Failure diagnosis | Logs and meta | Need one diagnosis aggregation command |
| Report generation | Agent-specific | Need stable data sources and artifact paths |

Recipe metadata must be queryable, not just discoverable by name. `recipe show`
and filtered `recipe list` should expose structured properties:

```json
{
  "name": "fedavg",
  "algorithm": "federated_averaging",
  "aggregation": "weighted_average",
  "client_requirements": {
    "needs_partial_fit": false,
    "state_exchange": "full_model",
    "min_clients": 2
  },
  "framework_support": ["pytorch", "tensorflow", "huggingface", "sklearn"],
  "heterogeneity_support": "non_iid",
  "privacy_compatible": ["differential_privacy"],
  "tags": ["standard", "baseline", "centralized_aggregation"]
}
```

The `nvflare-generate-job` skill should use filters such as:

```bash
nvflare recipe list --filter framework=huggingface --filter privacy=differential_privacy --format json
```

The agent should not choose algorithms by recipe-name keyword matching.

### Proposed Research Artifacts

Each agent-run experiment should create:

```text
experiments/<name>/
  plan.yaml
  runs/<run_id>/
    command_log.jsonl
    environment.json
    job_manifest.json
    metrics.json
    diagnosis.json
    artifacts/
  report.md
```

This can start as a skill convention before becoming a formal CLI.

### Proposed Future CLI

These are future-facing and should not block core agent readiness:

```bash
nvflare experiment init <name>
nvflare experiment run --recipe fedavg --param num_rounds=5 --param lr=0.01
nvflare experiment list
nvflare experiment compare <run_id...>
nvflare experiment report <experiment>
```

The first implementation can be a thin local artifact manager over Recipe API,
SimEnv, POC, and `nvflare job`.

## Gap Analysis

| Area | Current Status | Target Status | Missing for Agent Readiness |
| --- | --- | --- | --- |
| CLI JSON/JSONL | Closed by PR #4479 for single-result JSON; streaming JSONL still needed | Partial | Add `--format jsonl` for streaming commands, `job wait` for one-envelope final result, and terminal/timeout events for monitor |
| CLI schema | Closed by PR #4479 after merge for core admin CLI | Ready | Define concrete `--schema` JSON shape with flags, output modes, streaming, idempotency, mutating, and examples |
| Error handling | Closed by PR #4479 after merge for core admin CLI | Ready | Keep stable taxonomy for diagnosis and retry |
| Startup kit selection | Registry exists; scoped per-command selection still needed | Partial | Add non-mutating `--kit-id`/`--startup-kit` path and document human default vs agent workflow behavior |
| Recipe discovery | `recipe list` and common `job.py` convention | Partial | `recipe show`, filtered `recipe list`, packaged `nvflare/agent/templates` recipe job templates, structured task metadata |
| Client API docs | Good human docs | Ready | Agent skill with code-edit patterns |
| FedJob docs | Good human docs | Ready | Agent skill with templates and validation commands |
| Job logs | Core CLI closed by PR #4479 after merge | Partial | Bounded `--tail`/`--since`/`--max-bytes`, truncation metadata, diagnosis aggregation, and metrics extraction |
| Job download | Core CLI closed by PR #4479 after merge | Partial | `--output-dir`, returned artifact paths, and `missing_artifacts` list |
| POC | Core CLI closed by PR #4479 after merge | Partial | Doctor command, wait-ready, process status model, bound-address reporting, port-conflict warnings, explicit `agent doctor --server`, and prior-kit restore |
| Setup/deployment | Local install and POC paths exist; multi-machine/Docker/K8s/cloud docs vary by environment | Partial | Agent skills that choose deployment mode, verify prerequisites, and expose handoff/checkpoint steps |
| Distributed provisioning | CLI exists/in progress | Partial | Agent skill, explicit approval gates, async request status, and request expiration metadata |
| Skill install | Stub exists | Not ready | Real installer, packaged skill files, validation, on-demand skill loading |
| MCP | Not implemented | Lowest priority unless it avoids Python glue or enables stateful streaming | Derive tools/resources from CLI/shared helpers where CLI is insufficient |
| Auto-research | Possible manually | Future | Metrics/artifact manifest, compare/report conventions |

## Implementation Plan

### Phase 0: Contract Hardening

- Treat PR #4479 as closing the core admin CLI baseline after merge.
- Keep JSON/schema/exit-code coverage for all public CLI commands and extend the
  same contract to any new agent-facing commands.
- Add CLI contract tests that assert:
  - `--schema` succeeds without required args and returns the normalized schema
    JSON shape.
  - single-result `--format json` emits one envelope.
  - streaming commands use `--format jsonl`.
  - schemas declare arguments, flags, `streaming`, `output_modes`,
    `idempotent`, `idempotency_key_supported`, and `mutating`.
  - invalid args exit 4 with structured error.
  - non-interactive prompts fail unless `--force` or explicit confirmation is
    provided.
- Keep `nvflare_cli.md` as the command-specific implementation plan.

### Phase 1: Agent Skills

- Implement `nvflare agent skills install/list/validate/get`.
- Define and lint the skill frontmatter schema: inputs, outputs, approvals,
  CLI command references, and recovery categories.
- Add packaged skills under an NVFLARE-owned directory, for example:

```text
nvflare/agent/skills/
  nvflare-orient/SKILL.md
  nvflare-convert-client-api/SKILL.md
  nvflare-generate-job/SKILL.md
  nvflare-convert-pytorch/SKILL.md
  nvflare-migrate-from-distributed/SKILL.md
  nvflare-convert-lightning/SKILL.md
  nvflare-convert-huggingface/SKILL.md
  nvflare-convert-tensorflow/SKILL.md
  nvflare-convert-xgboost/SKILL.md
  nvflare-convert-sklearn/SKILL.md
  nvflare-convert-numpy/SKILL.md
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
  nvflare-diagnose-job/SKILL.md
  nvflare-federated-statistics/SKILL.md
  nvflare-privacy-security/SKILL.md
```

- Support user-level and project-local installs.
- Support skill tiers and `nvflare agent skills get --name <skill>` for
  on-demand loading.
- Keep `nvflare-auto-research` out of the Stage 1 bundle; it is a Stage 3
  skill.
- Add docs showing how agents should use skills with NVFLARE.

### Phase 2: Agent Doctor and Scaffolds

- Add `nvflare agent doctor --format json`.
- Add `nvflare poc wait-ready --timeout <seconds> --format json`, or make
  `nvflare poc start --format json` block until the server can accept
  connections.
- Add `nvflare poc stop --restore-kit --format json` and preserve
  `pre_poc_kit_id` in workflow state.
- Add recipe detail and template support:
  - `nvflare recipe show <name> --format json`
  - `nvflare recipe list --filter key=value --format json`
  - packaged recipe job templates under `nvflare/agent/templates` with the
    common layout convention: `README.md`, `job.py`, `client.py`, `model.py`,
    `prepare_data.py`, and `download_data.py` where applicable; a future
    `nvflare recipe scaffold <name> -o <dir>` can instantiate the same templates
    if needed.
- Add a documented `python job.py --export --export-dir <job_folder>`
  convention for generated examples and skills. Generated `job.py` should use
  `SimEnv` for local execution and export artifacts for POC/prod submission
  through `nvflare job submit`; it should not use `PocEnv`/`ProdEnv` for agent
  automation.

### Phase 3: Diagnosis and Metrics

- Add `nvflare job diagnose <job_id> --format json`.
- Add a packaged failure-pattern catalog for the diagnosis skill before relying
  on raw log interpretation.
- Add a structured log source boundary: `[USER_CODE_EXCEPTION]` for exceptions
  caught around user training code and `[FLARE]` for framework/runtime errors.
- Add `nvflare job metrics <job_id> --format json` or define a standard result
  artifact discovery contract.
- Make bounded job logs, stats, and download artifact paths sufficient for
  agents to produce a failure report.

The first diagnosis catalog should cover common, high-value patterns:

| Pattern | Evidence | Default Recovery |
| --- | --- | --- |
| `CUDA_OOM` | `CUDA out of memory` in client logs | `FIXABLE_BY_CODE`: reduce batch/model size or adjust device use |
| `AUTH_FAILURE` | authentication or certificate rejection | `FIXABLE_BY_CONFIG`: verify active startup kit and certificate validity |
| `ROUND_TIMEOUT` | round timeout or client non-response | `ENVIRONMENT_FAILURE`: inspect client connectivity/resources |
| `IMPORT_ERROR` | `ModuleNotFoundError` or import failure | `FIXABLE_BY_CODE`: update dependencies or job package contents |
| `ABSOLUTE_DATA_PATH` | local absolute path in code or logs | `FIXABLE_BY_CODE`: replace with site-local config/data path |
| `STARTUP_KIT_EXPIRED` | certificate validity failure | `FIXABLE_BY_CONFIG`: renew/repackage startup kit |

### Phase 4: MCP Server

- Keep MCP low priority while CLI covers the workflow.
- Implement MCP only after CLI contracts and shared helpers stabilize, and only
  for workflows where CLI is missing, agent-generated Python glue would be
  fragile, or long-lived streaming/session state is materially useful.
- Derive MCP tool schemas from CLI schemas or the same explicit command registry.
- Expose resources for docs, examples, recipes, config, and job manifests.
- Add policy gates for destructive/production actions.

### Phase 5: Auto-Research

- Start with skill conventions and artifact layouts.
- Add experiment local artifact manager only if repeated workflows justify it.
- Integrate metrics from TensorBoard/MLflow/W&B where available without making
  those dependencies mandatory.

## Success Criteria

Agent readiness should be validated with scenario tests:

1. A coding agent converts a simple PyTorch training script to Client API,
   generates `job.py`, runs SimEnv, and reports metrics.
2. A coding agent prepares POC, submits the exported job, monitors it, fetches
   logs/results, and diagnoses a deliberate client error.
3. A coding agent lists recipes, chooses an appropriate recipe for a framework,
   uses the common recipe `job.py` convention, and exports it.
4. A coding agent runs a small experiment sweep and writes a report with run
   lineage and artifact references.
5. A coding agent performs distributed provisioning request/package steps
   without exposing a private key.
6. MCP and CLI return equivalent envelopes for the same operation.

## Resolved Planning Decisions

1. Stage 1 adds an `agent` namespace with concrete commands:
   `nvflare agent skills install`, `nvflare agent skills list`,
   `nvflare agent inspect`, and `nvflare agent doctor`.
2. `agent doctor --online` ships in Stage 1 as a read-only environment snapshot.
3. Skills and examples guide coding agents; FLARE does not implement a broad
   automatic code rewriter in Stage 1.
4. MCP is lowest priority while CLI covers the workflow. Add MCP only when it
   avoids fragile agent-written Python glue for functionality that lacks CLI
   coverage, or when long-lived session/streaming state is materially useful.
5. Stage 1 metrics work stays limited to existing job logs, stats, downloads,
   and artifacts. A cross-tracker metrics artifact contract belongs to later
   auto-research design.
