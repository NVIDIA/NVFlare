# NVFLARE Agent Consumer Implementation Plan

## Document Control

| Field | Value |
| --- | --- |
| Created date | 2026-04-27 |
| Last updated | 2026-04-27 |
| Parent design | [nvflare_agent_consumer_readiness.md](nvflare_agent_consumer_readiness.md) |
| Stage 1 design | [nvflare_agent_foundation.md](nvflare_agent_foundation.md) |
| Related CLI design | [nvflare_cli.md](nvflare_cli.md) |
| Implementation scope | Stage 1 - agent-ready foundation |

## Goal

Implement the Stage 1 agent-consumer foundation so a coding agent can inspect
local ML code, install NVFLARE workflow skills, verify local and online FLARE
readiness, generate/export jobs through `job.py`, and operate POC or production
through the normal CLI.

The implementation must build on existing NVFLARE CLI, Client API, Recipe/FedJob,
SimEnv, session, and config-kit work. It must not create a parallel agent-only
control plane.

## Non-Goals

- MCP server implementation. MCP remains lower priority until a workflow lacks a
  CLI surface or needs long-lived state that CLI cannot reasonably provide.
- Auto-research registry, sweep runner, report generator, or experiment planner.
- A general automatic code rewriter. Coding agents edit user code; NVFLARE
  provides inspection, examples, templates, and deterministic skills.
- Replacing `nvflare job`, `nvflare system`, `nvflare study`, `nvflare poc`, or
  `nvflare config kit` with agent-specific wrappers.
- Running or importing user training code during inspection.

## Assumptions and Dependencies

1. PR #4479 or equivalent CLI work provides the shared command substrate:
   `job`, `system`, `study`, `poc`, and `config kit` support startup-kit
   resolution, JSON output, schemas, and structured errors.
2. New agent commands use the existing CLI output/error/schema helpers. Do not
   introduce a new envelope or error taxonomy.
3. Online checks use the same startup-kit/session resolution as normal CLI
   commands, including scoped kit overrides. Do not duplicate startup-kit
   parsing or session creation.
4. Skills and templates are packaged in the installed `nvflare` wheel so users do
   not need a GitHub checkout.
5. `agent inspect` is static only. It may parse source files with `ast`, but it
   must not import user modules or optional ML frameworks.

## Command Surface

Add an `agent` namespace:

```bash
nvflare agent skills install (--agent claude|codex|openclaw | --target <dir>) [--dry-run] [--force] [--format json] [--schema]
nvflare agent skills install --local --skill <name> <path> [--dry-run] [--force] [--format json] [--schema]
nvflare agent skills list [--agent claude|codex|openclaw | --target <dir>] [--filter capability=<name>] [--format json] [--schema]
nvflare agent skills validate [--agent claude|codex|openclaw | --target <dir>] [--skill <name>] [--format json] [--schema]
nvflare agent skills get --name <skill> [--format markdown|json] [--schema]
nvflare agent skills report-bug --skill <name> [--format json] [--schema]
nvflare agent skills revert --backup <path> [--format json] [--schema]
nvflare agent workflow plan --goal <goal> [--format json] [--schema]
nvflare agent doctor [--online] [--kit-id <id>] [--startup-kit <path>] [--server <address>] [--format json] [--schema]
nvflare agent inspect <path> [--confirm-framework <name>] [--redact on|off] [--format json] [--schema]
```

Agent-facing online commands must support non-mutating startup-kit selection.
Add a shared scoped selector, for example `--kit-id <id>` or
`--startup-kit <path>`, to `job`, `system`, `study`, and
`agent doctor --online`. This selector overrides the global active kit for that
command only. `nvflare config kit use` remains a human default selector, not an
agent workflow primitive. File locking for config writes is still useful, but it
does not remove the check-then-run race for agents.

`agent doctor --online` should also accept `--server <address>` for POC and
diagnosis workflows that need to verify the actual server address returned by
`poc start --format json`.

Implementation files:

```text
nvflare/tool/agent/__init__.py
nvflare/tool/agent/agent_cli.py
nvflare/tool/agent/skills.py
nvflare/tool/agent/doctor.py
nvflare/tool/agent/inspect.py
nvflare/agent/skills/
nvflare/agent/templates/
```

`nvflare/cli.py` should register `CMD_AGENT = "agent"` and dispatch to
`handle_agent_cmd`. Keep command parsing consistent with other top-level
commands.

## Gap-Driven CLI Work

The readiness design's gap table maps several missing capabilities to concrete
CLI surfaces. The implementation plan should track those explicitly so the
agent work does not stop at skill files.

### Required in the First Foundation Implementation

| Gap | CLI Work | Notes |
| --- | --- | --- |
| Skill installer/listing | `nvflare agent skills install`, `nvflare agent skills list` | New `agent` namespace commands; use packaged resources and skill schema linting |
| Skill runtime validation | `nvflare agent skills validate` | Validate installed skills against current NVFLARE version and `--schema` output; local doctor summarizes result |
| Skill on-demand loading | `nvflare agent skills get` | Return one skill on demand so agents do not load the full bundle |
| Skill capability filtering | `nvflare agent skills list --filter capability=cli_only` | Avoid routing constrained agents to skills requiring Python execution or file editing |
| Skill maintenance | `nvflare agent skills report-bug`; local override install | Give users a supported path for broken skills and custom fixes |
| Skill edit revert | `nvflare agent skills revert --backup <path>` | Restore files from `.nvflare_bak/<timestamp>/` after a failed or unwanted conversion |
| Workflow planning | `nvflare agent workflow plan` | Build dependency-checked skill route from goal, skill `depends_on`, `produces`, and current workflow state |
| Conversion lint | `nvflare agent inspect <path>` | Static local inspection, exported-job detection, absolute-path findings |
| Environment doctor | `nvflare agent doctor` | Local install/config/skill/startup-kit checks |
| Online environment snapshot | `nvflare agent doctor --online` | Read-only server/client/status/version/study capability snapshot with `snapshot_timestamp` and `snapshot_ttl_seconds` |
| Scoped identity selection | `--kit-id <id>` or `--startup-kit <path>` on online admin commands | Avoid global active-kit races in concurrent agents, notebooks, and shared machines |
| POC readiness wait | `nvflare poc wait-ready --timeout <seconds>` or blocking `nvflare poc start --format json` | Must provide a deterministic readiness point before online doctor or job submit |
| POC port conflict reporting | `poc prepare/start --format json` reports actual bound addresses and conflicts | Prevent agents from checking or submitting to the wrong server after port fallback |
| POC kit restore | `nvflare poc stop --restore-kit --format json` | Restore prior human default kit after POC workflows; `poc prepare` reports prior kit id |
| Recipe details/filtering | `nvflare recipe show <name> --format json`; `nvflare recipe list --filter key=value --format json` | Return queryable recipe metadata, framework/privacy support, requirements, parameters, and example/template references |
| Study preflight | `nvflare study list --format json` before production submit | Verify target study exists and selected identity has `can_submit_job: true` |
| Streaming output contract | `--format jsonl` for streaming commands; `nvflare job wait` for one final result | Avoid conflicting with the one-envelope JSON contract |
| Idempotency contract | `--schema` includes `idempotent`; `job submit` supports `--idempotency-key` | Prevent duplicate jobs during agent retries |
| Job recovery | `job list --idempotency-key <uuid>` and `job meta` include idempotency key | Recover job id after session loss |
| Bounded/partial log visibility | `job logs --tail/--since/--max-bytes --format json` includes per-site availability, truncation, and reason | Avoid overconfident diagnosis and oversized agent context in multi-org production |
| Download artifact contract | `job download --output-dir <path> --format json` returns artifact paths | Let agents locate trained models, metrics, and logs without guessing paths |
| Distributed approval status | `nvflare cert request-status --request-dir <path> --format json` | Model the site/lead approval gap in distributed provisioning |
| Export manifest/fingerprint | extend generated `job.py` export and `nvflare job submit -j` validation path | Export writes `_export_manifest.json` and `job_fingerprint.json`; submit eventually validates manifest before accepting folder |

### Required Skill/Template Work, Not New CLI Commands

| Gap | Implementation Surface | Notes |
| --- | --- | --- |
| Job scaffold | packaged templates under `nvflare/agent/templates/` | Do not add `nvflare recipe scaffold` in the first PR unless templates need a materialization CLI |
| Export convention | generated `job.py` convention | Standardize `python job.py --export --export-dir <job_folder>` |
| Metrics retrieval, minimum Stage 1 | examples/templates and downloaded artifacts | Document SimEnv metrics locations and generate `metrics_summary.json` where practical |
| FL semantic validation | `nvflare-local-validation` skill and generated example metrics | Verify receive/apply/train/send ordering and round-start/end metric continuity, not only exit code |
| Reduced-privilege agent identity guidance | skills and docs | Recommend dedicated member/lead startup kits for agents instead of shared admin kits |
| Workflow continuity | skill-owned `.nvflare_workflow_state.json` | Required for `nvflare-orient` resume/restart behavior; not a new CLI command |

### Follow-On CLI Candidates

These are useful but should not block the first `agent` command PR:

| Gap | Candidate CLI | Promotion Rule |
| --- | --- | --- |
| Job diagnosis | `nvflare job diagnose <job_id> --format json` | Promote after the diagnosis skill's failure-pattern catalog stabilizes |
| Metrics retrieval | `nvflare job metrics <job_id> --format json` | Promote after metrics artifact conventions stabilize across examples/recipes |
| Example index | `nvflare examples list/show --format json` | Promote when packaged examples/docs have a stable resource index |
| Recipe scaffold | `nvflare recipe scaffold <name> -o <dir>` | Promote only if packaged templates need a first-class materialization command |
| Live job inspect | `nvflare agent inspect job:<job_id> --format json` | Future operational extension; keep local file inspection as Stage 1 scope |
| Certificate renewal | `nvflare cert renew --kit-id <id> [--extend-days 90] --format json` | Promote with distributed-provisioning lifecycle work so doctor expiration warnings are actionable |
| Cert request status | `nvflare cert request-status --request-dir <path> --format json` | Support asynchronous distributed provisioning approval and workflow resume |

## Shared Contracts

### CLI Output and Retry Contract

Single-result commands use `--format json` and emit exactly one JSON envelope on
stdout. Streaming commands use `--format jsonl`, where each line is one complete
JSON event. Command schemas must declare:

- `streaming`: true or false.
- `output_modes`: supported formats, such as `["json"]` or `["json", "jsonl"]`.
- `idempotent`: true or false.
- `idempotency_key_supported`: true or false.

`--schema` output must be a concrete JSON structure used by skill linting,
contract tests, and future MCP derivation:

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

The schema serializer should normalize argparse and custom command metadata into
this shape. Consumers must not scrape `--help`.

`nvflare job monitor` is streaming and should support
`--timeout <seconds> --format jsonl`. It exits when the job reaches
`COMPLETED`, `FAILED`, or `ABORTED`. The final JSONL event must include
`terminal: true`; timeout emits a final event with `status: "TIMEOUT"` and
`terminal: true`.

`nvflare job wait <job_id> --timeout <seconds> --format json` is the
single-result alternative for agents that need one final envelope.

Non-idempotent operations must not be retried by skills unless protected by an
idempotency key. `nvflare job submit -j` should support:

```bash
nvflare job submit -j <job_folder>/<job_name> --idempotency-key <uuid> --format json
```

The server should deduplicate accepted submissions by key within a documented
time window and return the existing job when the key is reused.

`nvflare job logs` must be bounded:

- support `--tail <lines>`, `--since <timestamp>`, and `--max-bytes <bytes>`.
- default to at most 500 lines per site when no explicit bound is provided.
- include `logs_truncated: true` when output is capped.
- include per-site availability and reason when logs are unavailable.

`nvflare job download` must accept `--output-dir <path>` and return artifact
locations:

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

Skills must use returned artifact fields instead of guessing directory layouts.

### Skill Schema

Before writing skill content, define a lintable skill schema:

- `skill_version`, `min_flare_version`, `max_flare_version`.
- `maintainer` and `status`: active, experimental, or deprecated.
- `tier`: essential, conversion_core, framework_specific, or advanced.
- `agent_requirements`: python execution, file editing, shell execution, and
  network access requirements.
- `inputs`: named inputs with type, required flag, default, and description.
- `outputs`: named outputs produced by the skill.
- `depends_on`: prior skill outputs and required values.
- `produces`: fields this skill writes into workflow state.
- `cli_commands_used`: command names that must exist in `--schema`.
- `json_fields_used`: response fields the skill expects to extract, such as
  `data.job_id`.
- `approval_checkpoints`: id, risk level, reversibility, confirm phrase.
- `recovery_categories`: error recovery categories the skill knows how to
  handle.
- optional `dry_run_plan`: planned commands, mutating steps, files to change,
  approvals, and estimated duration.

Add a skill linter that validates `cli_commands_used` against installed command
schemas and fails tests when a packaged skill references a missing command or
flag.

At runtime, `nvflare agent skills validate` performs the same class of checks
against the user's installed NVFLARE version and installed skills. Missing
commands in a future skill should be reported as `command_not_found` or
`pending` when the skill's `min_flare_version` is newer than the installed
runtime, not as an opaque failure.

### Recovery Taxonomy

Use these categories in skill contracts, findings, and command data where
available:

| Category | Default Behavior |
| --- | --- |
| `RETRYABLE` | retry with bounded backoff |
| `FIXABLE_BY_CONFIG` | adjust identity/config/study and re-check |
| `FIXABLE_BY_CODE` | re-inspect, patch/regenerate, validate again |
| `REQUIRES_USER_APPROVAL` | emit approval checkpoint and wait |
| `ENVIRONMENT_FAILURE` | stop workflow and report remediation |
| `UNKNOWN` | collect diagnosis data and ask for human review |

The JSON error envelope must include `recovery_category` for every public error
code:

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

Add a central error-code to recovery-category mapping and contract tests that
fail when a public error code has no mapping.

### Approval Checkpoints

High-risk actions must use structured approval checkpoints with an exact confirm
phrase. The first implementation only needs skill-side checkpoints, but the
schema should be compatible with future server-side audit.

Stage 1 approval checkpoints are not platform-enforced gates. They can be
bypassed by direct CLI calls, so production skills must state this limitation and
recommend dedicated reduced-privilege startup kits. Server-side approval
enforcement is future platform work.

### Workflow State

Skills may write `.nvflare_workflow_state.json` in the project workspace. This
is not required for core CLI correctness, but it is required for
`nvflare-orient` to be useful after an interrupted workflow. The file records
the active workflow, current step, artifacts, pending approvals, last command,
approved checkpoints, denied checkpoints, `backup_path`, `pre_poc_kit_id`, and
next suggested command so another agent can resume without rediscovering
everything.
`nvflare-orient` must check this file before starting fresh and ask whether to
resume or restart when it exists.

Before `job submit`, the skill must generate and store `idempotency_key` in this
state file. If the session ends before the submit response is received, the next
session retries with the same key or searches with
`nvflare job list --idempotency-key <uuid> --format json`.

Distributed provisioning state should include `provisioning_step`,
`request_dir`, `request_id`, `submitted_at`, and `expires_at`; after request
creation, the site-side state is `awaiting_approval`. On resume, skills must
check `expires_at` and emit `CERT_REQUEST_EXPIRING_SOON` when less than one hour
remains.

## Workstream 1: Agent CLI Skeleton

### Implementation

1. Add `nvflare.tool.agent.agent_cli`.
2. Define subparsers for `skills install`, `skills list`, `skills validate`,
   `skills get`, `skills report-bug`, `skills revert`, `workflow plan`,
   `doctor`, and `inspect`.
3. Add `--schema` to every concrete command.
4. Use existing `output_ok`, `output_error`, `output_error_message`,
   `print_human`, and schema helpers.
5. Return one JSON envelope on stdout in `--format json` mode for single-result
   commands.
6. Keep progress and non-JSON diagnostics on stderr where the existing CLI
   contract requires it.
7. Add shared scoped startup-kit resolution for `--kit-id <id>` and
   `--startup-kit <path>` without mutating `~/.nvflare/config.conf`.

### Output Shape

Agent commands should use command-specific data inside the shared envelope:

```json
{
  "schema_version": "1",
  "status": "ok",
  "code": "OK",
  "data": {
    "command": "nvflare agent doctor",
    "findings": [],
    "next_steps": []
  }
}
```

### Tests

- `nvflare agent` with no subcommand prints help and exits with `INVALID_ARGS`.
- Every new command returns schema.
- JSON mode produces exactly one JSON envelope for single-result commands.
- Streaming commands use JSONL and declare that in schema.
- Invalid arguments use the common invalid-argument path.

## Workstream 2: Packaged Skills Installer

### Implementation

Replace the existing `nvflare/tool/install_skills.py` stub with the agent skills
installer, or keep it as a compatibility wrapper that delegates to
`nvflare.tool.agent.skills`.

Package skills under:

```text
nvflare/agent/skills/<skill-name>/SKILL.md
```

Use `importlib.resources.files("nvflare.agent.skills")` to read bundled skills
from the installed package.

Supported install targets:

- `--target <dir>` for explicit project-local, user-local, or custom installs.
- `--agent claude`, `--agent codex`, and `--agent openclaw` as named shortcuts
  that resolve to the currently supported location for that agent.
- `--local --skill <name> <path>` for installing a user-maintained override of
  one skill without modifying the package-bundled copy.

Do not silently write to hardcoded agent-specific paths when neither `--agent`
nor `--target` is provided. The no-target case should fail with a structured
hint showing the available `--agent` shortcuts and the explicit `--target`
form. Keeping path lookup behind `--agent` isolates future agent directory
changes to one resolver.

Installer behavior:

1. Read skill metadata from `SKILL.md` frontmatter.
2. Skip skills whose `min_flare_version` is higher than the installed NVFLARE
   version.
3. Back up an existing NVFLARE-managed skill directory to `.bak/<timestamp>/`
   before overwrite.
4. Never modify user-customized skills outside the NVFLARE-managed subdirectory.
5. Support `--dry-run` with the same validation and output shape but no writes.
6. Return `installed`, `skipped`, `backed_up`, and `warnings` arrays.
7. Install or expose tier metadata so agents can load only essential skills by
   default and fetch conversion/framework skills on demand.

`nvflare agent skills list` should report both packaged and installed skills:

- packaged skill name, version, min/max FLARE version.
- installed path when found.
- installed version when found.
- status: `packaged`, `installed`, `stale`, `incompatible`, or `missing`.
- maintainer, skill lifecycle status, tier, and declared agent capabilities.
- recommended install/update command.

The `--filter capability=<name>` option should filter by declared
`agent_requirements`, for example `capability=cli_only` excludes skills that
require Python execution or file editing.

`nvflare agent skills validate` should:

- read skill frontmatter and installed skill versions.
- compare `min_flare_version` and `max_flare_version` with the installed
  NVFLARE version.
- validate every `cli_commands_used` entry against command `--schema` when the
  command exists in the installed runtime.
- validate expected JSON field paths where the skill declares them.
- return per-skill status: `compatible`, `version_mismatch`,
  `command_not_found`, `schema_mismatch`, or `pending`.
- be summarized by `nvflare agent doctor`.

`nvflare agent skills get --name <skill> --format markdown` should return one
packaged or installed skill body for on-demand agent loading. It should also
support `--format json` to return metadata plus body text.

`nvflare agent skills report-bug --skill <name>` should return a structured
GitHub issue URL or project-specific reporting instruction that includes the
skill name, skill version, NVFLARE version, and validation status.

`nvflare agent skills revert --backup <path>` should restore files from a
skill-created `.nvflare_bak/<timestamp>/` directory. It must validate that the
backup metadata was created by an NVFLARE skill, report restored files, and
refuse to restore paths outside the original project root.

### Initial Skill Set

Ship the minimum useful Stage 1 bundle first:

- `nvflare-orient`
- `nvflare-convert-client-api`
- `nvflare-convert-pytorch`
- `nvflare-generate-job`
- `nvflare-local-validation`
- `nvflare-setup-local`
- `nvflare-poc-workflow`
- `nvflare-identity-and-config`
- `nvflare-job-lifecycle`
- `nvflare-diagnose-job`

Additional Stage 1+ skills can follow after the first bundle:

- `nvflare-production-submit`
- `nvflare-convert-lightning`
- `nvflare-migrate-from-distributed`
- `nvflare-convert-tensorflow`
- `nvflare-convert-huggingface`
- `nvflare-convert-xgboost`
- `nvflare-convert-sklearn`
- `nvflare-convert-numpy`
- `nvflare-federated-statistics`
- `nvflare-distributed-provisioning`
- `nvflare-privacy-security`

Advanced deployment skills are intentionally separated from this follow-on list
because they are more complex than local setup and POC. Multi-machine, Docker,
K8s, and cloud deployments require additional environment discovery, network
validation, artifact distribution, credential handling, and human handoff
boundaries. Treat them as a separate implementation track after the first Stage
1 foundation lands.

### Tests

- Install from package resources into a temp target.
- Dry run reports planned writes without creating files.
- Existing managed install is backed up before overwrite.
- Version-incompatible skill is skipped with a hint.
- Permission failure returns a structured error for explicit invocation.

## Workstream 3: Static Agent Inspect

### Purpose

`nvflare agent inspect <path>` tells the agent what it is looking at and whether
the code appears already converted to FLARE. It does not mutate files and does
not execute user code. It must redact potentially sensitive literals by default.

### Implementation

Add `nvflare/tool/agent/inspect.py` with pure helpers:

```python
inspect_path(path: str) -> dict
inspect_python_file(path: Path) -> dict
inspect_job_layout(path: Path) -> dict
detect_frameworks(files: list[Path]) -> list[dict]
detect_dynamic_patterns(files: list[Path]) -> list[dict]
detect_secret_like_literals(files: list[Path]) -> list[dict]
detect_flare_client_api(files: list[Path]) -> dict
detect_job_export_support(files: list[Path]) -> dict
```

Traversal rules:

- accept a file or directory.
- skip `.git`, `.venv`, `venv`, `__pycache__`, `.mypy_cache`, `.pytest_cache`,
  `build`, `dist`, and hidden directories unless the target itself is hidden.
- cap file count and individual file size to avoid scanning large datasets.
- default caps: inspect at most 5,000 files, skip individual files larger than
  1 MiB, and include at most 20 evidence strings per category in the result.
- parse Python files with `ast` where possible; fall back to text evidence when
  syntax is invalid.
- do not follow symlinks by default. If symlink support is later enabled, never
  follow symlinks resolving outside the inspected root.
- report skipped symlinks as `SYMLINK_SKIPPED` findings with sanitized target
  and action `skipped`.
- never return raw file contents in JSON. Evidence should be structural,
  redacted, and capped.
- default `--redact on`; `--redact off` is explicit local-debug behavior.

Detect framework evidence for:

- PyTorch: `torch`, `torch.nn.Module`, `DataLoader`, `optimizer.step`.
- PyTorch Lightning: `lightning`, `pytorch_lightning`, `LightningModule`,
  `Trainer`.
- TensorFlow/Keras: `tensorflow`, `tf.keras`, `keras.Model`, `model.fit`.
- Hugging Face: `transformers`, `Trainer`, `AutoModel`, `AutoTokenizer`,
  `datasets`.
- XGBoost: `xgboost`, `xgb.train`, `XGBClassifier`, `DMatrix`.
- scikit-learn: `sklearn`, estimator imports, `fit`, `predict`, `Pipeline`.
- NumPy/custom: `numpy` and custom training loops only when no stronger
  PyTorch/Lightning/TensorFlow/Hugging Face/XGBoost/sklearn evidence is found.

Framework detection should evaluate stronger framework evidence first. NumPy is
common inside other ML stacks, so NumPy/custom is a fall-through classification,
not a high-confidence primary framework when PyTorch, TensorFlow, Hugging Face,
XGBoost, or scikit-learn evidence is also present.
Return frameworks as a ranked list with numeric `confidence` from `0.0` to
`1.0`, `supporting_evidence`, and `contradicting_evidence`.

Detect high-risk code patterns:

- Unix absolute paths.
- Windows absolute paths.
- `Path.home()`.
- `os.path.expanduser("~")`.
- secret-like literals and credential-adjacent names such as passwords, tokens,
  cloud keys, API keys, connection strings, and private endpoints.

Report these as `ABSOLUTE_DATA_PATH` findings because they often work in SimEnv
but fail on remote sites. With redaction enabled, report file/line and pattern
type, not the full literal value.

Detect dynamic patterns that make static classification incomplete:

- Hydra/OmegaConf instantiation.
- dynamic imports through `importlib`.
- `getattr` dispatch on framework or Client API objects.
- `torch.compile` wrappers.
- external training functions imported from unknown packages.

Report these as `DYNAMIC_FRAMEWORK_RESOLUTION` findings and recommend
`--confirm-framework <name>` when confidence is below threshold.

Detect multi-GPU and large-model patterns:

- `torch.nn.DataParallel`.
- `torch.nn.parallel.DistributedDataParallel`.
- `torch.distributed.init_process_group`.
- `FullyShardedDataParallel` / `FSDP`.
- `accelerate.Accelerator`.
- gradient accumulation variables or delayed `optimizer.step()` patterns.

Report these in `training_mode` with evidence and `fl_impact`, such as
unwrapping `module.` prefixes, gathering full state dicts for FSDP, or sending
after the optimizer update boundary. For DDP/FSDP/Accelerate migration, set
`training_mode.distributed = true`,
`training_mode.conversion_path = "nvflare-migrate-from-distributed"`, and
`training_mode.conversion_difficulty = "hard"`.

Detect FLARE Client API conversion evidence:

- `import nvflare.client as flare` or equivalent import.
- `flare.init()`.
- `flare.receive()`.
- `flare.send(...)`.
- `FLModel`.
- optional `flare.is_running()` loop.

Classify conversion state:

- `not_converted`
- `partial_client_api`
- `client_api_converted`
- `flare_job`
- `exported_job`
- `unknown`

Detect job/export readiness:

- `job.py` exists.
- `job.py` imports Recipe/FedJob/SimEnv.
- `job.py` appears to support `--export --export-dir`.
- existing variants such as `--export_job` or `--export_config` are recognized
  for compatibility but reported as non-standard.

Detect recipe fit:

- `compatible` when a standard recipe appears sufficient.
- `requires_fedjob` when custom aggregation, server-side per-round decisions,
  standalone ScriptRunner behavior, or unusual orchestration is detected.
- `uncertain` when evidence is mixed.

Classify target kind:

- `training_repo`
- `training_script`
- `job_source`
- `exported_job`
- `workspace`
- `unknown`

### Output Data

```json
{
  "path": "./my_training_repo",
  "kind": "training_repo",
  "redaction": "on",
  "frameworks": [
    {
      "name": "pytorch",
      "confidence": 0.85,
      "supporting_evidence": [
        {"file": "train.py", "line": 3, "pattern": "import torch"},
        {"file": "model.py", "line": 1, "pattern": "torch.nn.Module"}
      ],
      "contradicting_evidence": []
    }
  ],
  "entrypoints": ["train.py"],
  "conversion_state": "not_converted",
  "recipe_fit": {
    "assessment": "compatible",
    "evidence": ["standard train/eval loop detected"],
    "recommendation": "Use Recipe API"
  },
  "training_mode": {
    "multi_gpu": false,
    "distributed": false,
    "framework": "single_process",
    "conversion_path": null,
    "conversion_difficulty": null,
    "evidence": [],
    "special_concerns": [],
    "fl_impact": null
  },
  "flare_client_api": {
    "detected": false,
    "missing": ["flare.init", "flare.receive", "flare.send"]
  },
  "job_layout": {
    "has_job_py": false,
    "standard_files": ["README.md", "model.py", "train.py"],
    "missing_recommended_files": ["job.py", "client.py"]
  },
  "recommended_skills": ["nvflare-convert-pytorch", "nvflare-generate-job"],
  "recommended_next_steps": [
    "Use nvflare-convert-pytorch skill",
    "Generate job.py with SimEnv and export support"
  ]
}
```

### Tests

Use small fixture directories for:

- plain PyTorch training script.
- PyTorch script partially converted to Client API.
- fully converted Client API script.
- Lightning, TensorFlow/Keras, Hugging Face, XGBoost, scikit-learn, NumPy.
- existing FLARE job folder with exported config.
- submit-ready exported job folder with no `job.py`.
- `job.py` using `--export --export-dir`.
- repo with DDP/FSDP/Accelerate evidence.
- repo with hardcoded absolute data paths.
- repo with Hydra/OmegaConf or dynamic import evidence.
- repo with secret-like literals to verify default redaction.
- large/skipped files and hidden directories.
- symlinks inside the target and symlinks resolving outside the target root;
  both should be skipped by default and reported without reading the target.

## Workstream 4: Agent Doctor

### Local Doctor

`nvflare agent doctor --format json` performs local checks only:

1. NVFLARE import/version.
2. Python version.
3. command schema availability for `agent`, `job`, `poc`, `system`,
   `study`, `recipe`, and `config kit`.
4. active startup kit registration and stale-path detection.
5. POC workspace configuration and basic path existence.
6. packaged skill availability.
7. installed skill locations.
8. `skills validate` compatibility summary.
9. optional framework dependency presence for the installed extras.

No server connection is attempted unless `--online` is provided.

### Online Doctor

`nvflare agent doctor --online --format json` runs local doctor first, then uses
the selected startup kit to connect to the FLARE server. The selected kit comes
from `--kit-id`, `--startup-kit`, a process-scoped override, or finally the
human default active kit.
When `--server <address>` is supplied, doctor verifies that address with the
selected identity instead of relying only on the address embedded in the active
kit; this is primarily for POC port-conflict handling.

Online checks:

1. startup-kit identity, role, org, and project metadata when derivable.
2. server connection and authentication.
3. server status.
4. server version.
5. connected client list.
6. client-side versions when the system command supports it.
7. resource summary.
8. job list summary.
9. study list summary when authorized.
10. authorization findings for read-only commands that are not permitted.
11. startup kit certificate expiration and warning window.
12. authorized study roles and submit capability when available.
13. renewal command such as `nvflare cert renew --kit-id <id> --format json`
    when expiration is near or expired and renewal is supported.
14. snapshot timestamp and recommended TTL for using this readiness result in
    follow-on production decisions.

The command must remain read-only. It must not submit, abort, delete, download,
shutdown, restart, change identity selection, or modify configuration.

### Helper Strategy

Refactor boundary:

- Use startup-kit/session helpers from the config-kit CLI work directly for base
  identity and connection checks. The helper must accept a scoped kit override
  and must not write the global config.
- For optional read-only snapshots that already have stable `--format json`
  commands, call the existing CLI contract rather than copying internal logic:
  `job list`, `system status`, `system version`, `system resources`, and
  `study list`.
- Refactor a command into a shared pure helper only when startup-kit/session
  resolution must happen at the call site or when subprocess overhead becomes a
  measured problem.
- Do not copy parser, startup-kit resolution, or authorization logic into
  `agent doctor`.

### Output Data

```json
{
  "schema_version": "1",
  "status": "ok",
  "code": "OK",
  "data": {
    "mode": "online",
    "snapshot_timestamp": "2026-04-27T10:00:00Z",
    "snapshot_ttl_seconds": 60,
    "nvflare_version": "2.8.0",
    "selected_startup_kit": {
      "id": "admin@nvidia.com",
      "path": "/tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com",
      "status": "ok"
    },
    "startup_kit_expiration": {
      "expires_at": "2026-06-27T00:00:00Z",
      "days_remaining": 61,
      "status": "ok",
      "renewal_command": "nvflare cert renew --kit-id admin@nvidia.com --format json"
    },
    "server": {"reachable": true, "status": "running", "version": "2.8.0"},
    "clients": [{"name": "site-1", "connected": true, "version": "2.8.0"}],
    "jobs": {"available": true, "running": 0, "recent": []},
    "studies": {
      "available": true,
      "items": [
        {"name": "research_study", "role": "lead", "can_submit_job": true}
      ]
    },
    "findings": [],
    "next_steps": []
  }
}
```

### Tests

- no config file.
- stale active startup kit.
- valid local POC kit.
- skills packaged but not installed.
- server unreachable.
- authentication failure.
- reachable POC server with server/client version reporting.
- explicit `--server <address>` override verifies the requested server address.
- unauthorized study list does not fail the whole doctor command; it is reported
  as an authorization finding.
- expired or soon-to-expire startup kit returns `STARTUP_KIT_EXPIRED` or a
  warning finding.
- multi-study role/capability data is included when available.
- snapshot timestamp and TTL are present in online output.

## Workstream 5: Job Template and Example Conventions

Stage 1 templates should be packaged in the installed wheel under
`nvflare/agent/templates/` and loaded through `importlib.resources`. The
source-tree `examples/hello-world` examples can mirror these templates for
developers, but released skills must not depend on a GitHub checkout.

Recommended layout:

```text
README.md
job.py
client.py
model.py
prepare_data.py
download_data.py
requirements.txt
```

Not every job needs every file. Use these filenames when the concept exists.

`job.py` convention:

```bash
python job.py
python job.py --export --export-dir <job_folder>
nvflare job submit -j <job_folder>/<job_name> --format json
```

Rules:

1. `python job.py` uses `SimEnv` for local validation.
2. `python job.py --export --export-dir <job_folder>` writes a submit-ready job
   folder.
3. Export should be atomic: write to a temporary directory and move into place
   only after the manifest and required files are complete.
4. Export should write `_export_manifest.json` and `job_fingerprint.json`.
5. `_export_manifest.json` should include `poc_validated` and
   `poc_validation` fields. The default is false/null. POC validation should
   set these only after a successful POC run of the exported job.
6. `agent inspect` should classify a valid exported folder as `exported_job` and
   warn when source hashes or required files do not match.
7. POC and production submission use `nvflare job submit -j`, not `PocEnv` or
   `ProdEnv` in generated automation flows.
8. `nvflare job submit` should eventually validate `_export_manifest.json`
   before accepting an exported folder and warn on production/study submission
   when `poc_validated` is false or missing.
9. POC workflows should record `pre_poc_kit_id` before prepare and call
   `nvflare poc stop --restore-kit --format json` after validation.
10. First concrete PyTorch conversion template should use packaged
   `nvflare/agent/templates/hello-pt`.
11. Generated examples should make the FL ordering constraint visible:
    receive global model, apply received weights, train locally, then send the
    updated model and metrics.
12. When practical, generated jobs should write `metrics_summary.json` with
    round-start and round-end metrics so the local-validation skill can detect
    obvious FL semantic errors, such as ignoring the received global model.

`nvflare-local-validation` must document SimEnv limitations: no TLS/auth, no
startup-kit validation, no network latency or timeout coverage, no real
site-local data separation unless the job explicitly models it, and no guarantee
that per-site resource limits match production. POC validation is required
before production submission unless the user explicitly accepts the risk.
The skill should treat process exit code as necessary but insufficient:
successful local validation also checks exported artifacts, receive/apply/train
ordering in the converted code, and metric continuity between FL rounds when the
example provides the required metrics.

POC commands should make port binding explicit. `poc prepare --format json`
should preflight default port availability when possible. `poc start --format
json` should return actual bound server/admin addresses, `port_conflict`,
`default_port`, and warnings. POC skills must use these returned addresses for
online doctor and submit flows instead of assuming defaults.

Implementation can add packaged templates under `nvflare/agent/templates/`.
Do not add `nvflare recipe scaffold` in the first cut unless the skill bundle
needs a CLI to materialize templates. If a scaffold command is added later, it
must use the same templates.

## Workstream 6: Skill Content

Skills should encode deterministic workflows, not generic prose.

Each skill should include:

- required inputs.
- files to inspect.
- commands to run.
- expected JSON fields to read.
- stop conditions.
- human approval boundaries.
- troubleshooting branches.

Skills that edit user files must:

- copy every file to be modified into `.nvflare_bak/<timestamp>/` before the
  first write.
- include `backup_path` in skill output and workflow state.
- be idempotent: do not duplicate imports, `flare.init()`, `flare.receive()`,
  `flare.send()`, or config blocks when re-run.
- document `nvflare agent skills revert --backup <backup_path> --format json`
  as the recovery command.

Initial end-to-end skills:

1. `nvflare-orient`: first check `.nvflare_workflow_state.json`. If present,
   summarize current step, artifacts, and pending approvals, then ask whether to
   resume or restart. If absent, run doctor, inspect schemas, determine whether
   the user is first-time setup, submitting an existing job, diagnosing a
   failure, or converting code. It should produce a short explicit plan before
   invoking mutating commands.
2. `nvflare-convert-client-api`: convert a centralized training loop to Client
   API using static evidence from `agent inspect`.
3. `nvflare-convert-pytorch`: specialize the Client API skill using packaged
   `hello-pt` as the first concrete pattern. Include `state_dict`/`load_state_dict`,
   optimizer-state guidance, DDP/DataParallel `module.` prefix handling, FSDP
   full-state-dict guidance, and gradient accumulation boundaries.
4. `nvflare-migrate-from-distributed`: guide DDP/FSDP/Accelerate users to
   remove collective communication, unwrap the model/training loop, verify a
   single-site training loop, then add FLARE Client API.
5. `nvflare-generate-job`: create `job.py` using the standard SimEnv/export
   convention.
6. `nvflare-local-validation`: run `python job.py`, check output/artifacts, and
   run FL semantic checks from available metrics. It must verify the generated
   code follows receive/apply/train/send ordering and warn when round-start/end
   metrics suggest the received global model is ignored.
7. `nvflare-setup-local`: install NVFLARE locally, run doctor checks, prepare
   POC, and verify readiness.
8. `nvflare-poc-workflow`: prepare/start POC, export job, submit, wait or
   monitor with JSONL, logs, download, stop, and restore prior kit.
9. `nvflare-identity-and-config`: show/list startup kits, explain human default
   selection, and prefer scoped `--kit-id`/`--startup-kit` selectors for agent
   workflows.
10. `nvflare-job-lifecycle`: submit/list/wait/monitor/meta/logs/download/abort.
11. `nvflare-diagnose-job`: collect job meta, logs, stats, system status, and
   produce evidence-backed findings.

Stage 1+ production and distributed skills:

- `nvflare-production-submit`: run a fresh online doctor snapshot, validate the
  target study through `nvflare study list --format json`, require
  `can_submit_job: true`, generate and persist a submit idempotency key before
  calling `job submit`, and use a structured approval checkpoint before any
  production mutation.
- `nvflare-distributed-provisioning`: model site and lead responsibilities as
  separate asynchronous steps. Site-side request creation writes workflow state
  with `provisioning_step: "awaiting_approval"`, `request_dir`, `request_id`,
  `submitted_at`, and `expires_at`; resume/poll uses
  `nvflare cert request-status --request-dir <path> --format json`; approval is
  explicitly the lead action. Request-status output should include `status`,
  `expires_at`, and `time_remaining_hours`; emit
  `CERT_REQUEST_EXPIRING_SOON` when less than one hour remains.

`nvflare agent workflow plan` should use skill `depends_on`/`produces`
metadata, current inspect output, workflow state, and approval checkpoint status
to produce a valid skill route. It should fail with unresolved questions rather
than route through a skill whose dependencies are not satisfied.

Diagnosis skill requirements:

- collect raw data through existing CLI commands.
- detect per-site log availability from `job logs --format json` and emit
  `PARTIAL_LOG_VISIBILITY` when logs are missing or permission-denied.
- request bounded logs, for example `--tail 200`, and treat
  `logs_truncated: true` as a finding when more context may be needed.
- use log source markers: `[USER_CODE_EXCEPTION]` means user training code;
  `[FLARE]` means framework/runtime code; no marker means `source: "unknown"`.
- avoid confident root-cause claims when required site logs are unavailable.
- match a packaged failure-pattern catalog before asking an LLM to interpret
  logs.
- return structured findings with `pattern`, `evidence`, `severity`,
  `recovery_category`, and `hint`.

Skill dry-run requirements:

- every mutating workflow skill should support a dry-run plan in its contract.
- dry-run output lists commands, files to create/change, approval checkpoints,
  mutating steps, and estimated duration.

Metrics convention:

- examples should document where SimEnv writes metrics.
- generated jobs should produce `metrics_summary.json` when practical.
- `metrics_summary.json` should include enough round-level data to compare
  round-start and round-end loss/accuracy for the FL semantic checks.
- job download docs should point agents to metrics artifacts so completion can
  be evaluated by training outcome, not only process status.

Recipe/FedJob decision content:

- Recipe is preferred for supported standard algorithms and normal model
  receive/send loops.
- FedJob/ScriptRunner is preferred when the existing script should be launched
  as a client task or job assembly needs direct control.
- Custom Executor/Controller is used when aggregation, per-round server logic,
  non-model tasks, or orchestration cannot be expressed by Recipe or
  ScriptRunner.
- Skills should branch on `recipe_fit.assessment` and ask the user when it is
  `uncertain`.

Recipe metadata schema for `recipe show` and filtered `recipe list`:

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

The first filter keys should include `framework`, `privacy`, `algorithm`,
`aggregation`, and `state_exchange`.

Framework-specific conversion content must include exact state extraction and
application code, optimizer-state guidance, metric collection, ordering
constraints, and at least one before/after example before the skill is released
for that framework. PyTorch ships first with packaged `hello-pt`; Lightning, Hugging
Face, TensorFlow/Keras, XGBoost, scikit-learn, and NumPy/custom can follow as
their examples are ready.

Notebook guidance:

- avoid `config kit use`; use scoped identity selectors.
- poll short-lived status commands instead of blocking notebook kernels with
  long monitors.
- after kernel restart, run `agent doctor --online` before preparing POC again.
- provide a Stage 1 notebook example when docs/examples are updated.

Advanced deployment skill track, separate from the initial foundation
implementation:

- `nvflare-deploy-multimachine`: provision or distributed-provision startup
  kits, identify server/client/admin hosts, verify ports and startup-kit
  placement, and run read-only system checks after deployment.
- `nvflare-deploy-docker`: document container image, volume mounts for startup
  kits/workspaces, port mappings, and container health checks.
- `nvflare-deploy-k8s`: validate generated Helm/K8s artifacts, namespaces,
  secrets/config maps, service exposure, and pod readiness checks.
- `nvflare-deploy-cloud`: guide CSP-specific networking, storage, identity,
  cluster, and firewall/security-group setup; keep provider-specific mutation
  steps behind explicit human approval.

Do not implement these as part of the first `agent` command PR. Each advanced
deployment skill should get a short design note or implementation checklist
before coding:

| Skill | Required Pre-Implementation Decisions |
| --- | --- |
| `nvflare-deploy-multimachine` | Provisioned vs distributed provisioning flow, host inventory format, startup-kit transfer assumptions, remote command boundary |
| `nvflare-deploy-docker` | Supported image source, compose vs plain Docker command pattern, volume mount layout, port mapping and health-check contract |
| `nvflare-deploy-k8s` | Helm artifact source, namespace/secrets strategy, service exposure model, pod readiness checks, cluster permission assumptions |
| `nvflare-deploy-cloud` | Supported CSPs, managed K8s vs VM path, networking/security-group assumptions, storage/identity integration, approval gates |

The first implementation may include placeholder skill catalog entries marked
`deferred`, but it should not ship incomplete deployment automation that appears
production-ready.

## Workstream 7: Documentation

Add or update:

- `docs/user_guide/nvflare_cli/agent_cli.rst`.
- agent skill installation section in the user guide.
- setup/deployment skill section covering local setup in the first pass, with
  multi-machine, Docker, K8s, and cloud-provider skills documented as a
  separate advanced deployment track.
- `README.md` snippets for the first skill bundle.
- packaged `nvflare/agent/templates/hello-pt` notes showing the Stage 1 export
  convention.

Docs must show the full POC path:

```bash
nvflare agent doctor --format json
nvflare agent inspect ./my_training_repo --format json
python job.py
python job.py --export --export-dir ./jobs
nvflare poc prepare --format json
nvflare poc start --format json
nvflare poc wait-ready --timeout 60 --format json
nvflare config kit show --format json
nvflare job submit -j ./jobs/<job_name> --idempotency-key <uuid> --format json
nvflare job wait <job_id> --timeout 3600 --format json
nvflare job logs --site all <job_id> --tail 200 --format json
nvflare job download <job_id> --output-dir ./<job_id> --format json
nvflare poc stop --restore-kit --format json
```

## Workstream 8: Tests and Validation

### Unit Tests

Add:

```text
tests/unit_test/tool/agent/agent_cli_test.py
tests/unit_test/tool/agent/agent_skills_test.py
tests/unit_test/tool/agent/agent_inspect_test.py
tests/unit_test/tool/agent/agent_doctor_test.py
```

Coverage:

- parser/dispatch/schema for all new commands.
- concrete `--schema` output shape for flags, arguments, output modes,
  streaming, idempotency, mutating, and examples.
- JSON envelope for success and error cases.
- static inspect classifications and framework evidence.
- symlink traversal safety for `agent inspect`.
- absolute path anti-pattern findings.
- exported job manifest/fingerprint validation.
- skill frontmatter schema linting against CLI schemas.
- skill `agent_requirements`, capability filters, maintainer metadata, and
  local override install behavior.
- skill-created backup metadata and `skills revert` restore behavior.
- recovery category and approval checkpoint schema validation.
- skill install dry-run, backup, overwrite, version filtering, and permission
  failures.
- doctor local findings without requiring a server.
- online doctor error paths through mocked session helpers.
- online doctor snapshot timestamp/TTL output.
- production-submit preflight behavior for study capability results.
- distributed provisioning workflow state and request-status handling.
- cert request expiration and `CERT_REQUEST_EXPIRING_SOON` handling.
- partial log visibility findings.
- bounded log handling and `logs_truncated` findings.
- user-code vs FLARE-framework log source markers.
- job id recovery through idempotency key lookup.
- job download artifact path contract.
- FL semantic local-validation findings from `metrics_summary.json`.

### Integration Tests

Keep integration tests small:

1. POC prepare/start.
2. POC wait-ready or blocking start readiness.
3. `agent doctor --online`.
4. export a tiny `hello-pt` style job and validate `_export_manifest.json`.
5. submit with an idempotency key, wait, monitor with JSONL, logs, download.

Run heavier end-to-end tests in CI only when the relevant POC integration suite
is enabled.

### Static Safety Tests

Add tests proving that `agent inspect` does not import user code:

- fixture module raises at import time.
- inspection succeeds without triggering import side effects.
- optional framework packages are not required for detection.

### Contract Tests

Add a release-blocking contract suite for agent-facing commands and packaged
skills:

- every public command referenced by a skill has working `--schema`.
- every public command schema uses the normalized JSON shape.
- every referenced flag exists in that command schema.
- every single-result `--format json` path emits the shared envelope.
- every streaming command declares JSONL support and emits one event per line.
- every command schema declares `idempotent`; non-idempotent retryable commands
  declare whether an idempotency key is supported.
- every public error envelope has `recovery_category`.
- packaged skills pass frontmatter schema validation.
- packaged skills declare `agent_requirements`, maintainer, lifecycle status,
  and ownership metadata.
- `nvflare agent skills validate` detects compatible, stale, missing-command,
  and version-mismatch skill states.
- scoped startup-kit selection does not mutate `~/.nvflare/config.conf`.
- `job logs --format json` includes per-site availability fields.
- `job logs` supports bounded retrieval and truncation metadata.
- `job download --format json` returns `download_path`, `artifacts`, and
  `missing_artifacts`.
- `agent doctor --online` includes snapshot timestamp and TTL.
- `job list` and `job meta` expose the idempotency key when present.
- future MCP mirrors, if added, return equivalent status/code and key data
  fields to the CLI path.

## Implementation Sequence

1. Add the `agent` parser skeleton and command dispatch.
2. Add concrete `--schema` serializer and JSON-output tests for command
   skeletons.
3. Add CLI contract hardening for JSONL streaming, `job wait`, schema
   idempotency metadata, bounded logs, download artifact paths, and
   `job submit --idempotency-key`.
4. Add scoped startup-kit resolution helpers for non-mutating `--kit-id` and
   `--startup-kit` selection.
5. Implement static `agent inspect`, including exported job detection,
   redaction, dynamic-pattern findings, `recipe_fit`, `training_mode`, and
   absolute path findings.
6. Implement local `agent doctor`.
7. Define the skill schema and write `nvflare-orient` plus
   `nvflare-convert-client-api` skill content.
8. Implement packaged skills installer, `skills list`, `skills validate`,
   `skills get`, `skills report-bug`, local override installs, workflow
   planner, and package-data wiring.
9. Add `poc wait-ready` or equivalent blocking `poc start --format json`
   behavior, bound-address/port-conflict reporting, plus
   `poc stop --restore-kit`.
10. Add `recipe show <name> --format json` and filtered `recipe list`.
11. Implement online `agent doctor` using shared session/read helpers and
    scoped identity selection, including snapshot timestamp/TTL and study
    capability summary.
12. Add first template/example convention updates, including export manifest,
    `poc_validated`, job fingerprint, metrics summary conventions, and FL
    semantic local-validation checks.
13. Add framework-specific conversion skill content, PyTorch first, plus
    distributed migration skill content.
14. Add user guide and notebook updates.
15. Add POC end-to-end smoke validation.
16. Add follow-on CLI candidates only after their backing skill/template
    contracts stabilize.
17. Add production-submit and distributed-provisioning skill content after
    study preflight, partial log visibility, and request-status contracts are
    available.

Mapping to the Stage 1 foundation phases:

| Foundation Phase | Implementation Workstreams |
| --- | --- |
| Phase 1: Agent CLI Skeleton | Gap-driven CLI work; Workstream 1 |
| Phase 2: Agent Inspect | Gap-driven CLI work; Workstream 3 |
| Phase 3: Agent Doctor | Gap-driven CLI work; Workstream 4 |
| Phase 4: Minimum Skill Bundle and Installer | Workstreams 2 and 6 |
| Phase 5: Skill-Guided Conversion Examples | Workstreams 5, 6, and 7 |
| Phase 6: Diagnosis Workflow | Follow-on CLI candidates; Workstreams 6, 7, and 8 |

## Quality Gates

Before merge:

- `python -m pytest tests/unit_test/tool/agent`
- targeted tests for `nvflare poc wait-ready` or blocking `poc start`.
- targeted tests for `poc prepare/start --format json` bound-address and
  port-conflict reporting.
- targeted tests for `nvflare poc stop --restore-kit`.
- targeted tests for `nvflare recipe show <name> --format json` and filtered
  `recipe list`.
- targeted tests for `job wait --format json`, `job monitor --format jsonl`,
  and `job submit --idempotency-key`.
- targeted tests for `job list --idempotency-key` and `job meta`
  idempotency-key reporting.
- targeted tests for `job logs --format json` partial availability fields.
- targeted tests for bounded logs, `logs_truncated`, and `job download`
  artifact paths.
- targeted tests for `study list --format json` submit capability fields.
- targeted tests for `nvflare cert request-status --format json` when that
  command is implemented, including `expires_at` and `time_remaining_hours`.
- targeted tests for scoped startup-kit selection on affected commands.
- targeted tests for `nvflare agent skills validate`, capability filters,
  local override install, `skills report-bug`, and `skills revert`.
- targeted tests for `agent inspect` symlink skipping.
- targeted tests for online doctor timestamp/TTL.
- targeted tests for online doctor `--server <address>` override.
- targeted tests for local-validation semantic checks over
  `metrics_summary.json`.
- targeted tests for `nvflare job submit -j` manifest validation when that
  validation is implemented.
- contract tests for packaged skills and CLI schema references.
- targeted tests for any shared CLI helpers touched.
- `python -m pytest tests/unit_test/tool/cli_invalid_args_json_test.py` if
  parser behavior changes.
- `git diff --check`.
- manual smoke:

```bash
nvflare agent skills install --agent codex --dry-run --format json
nvflare agent skills list --agent codex --format json
nvflare agent skills validate --agent codex --format json
nvflare agent skills get --name nvflare-orient --format markdown
nvflare agent workflow plan --goal "convert this PyTorch repo and run POC" --format json
nvflare agent inspect nvflare/agent/templates/hello-pt --format json
nvflare agent doctor --format json
```

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Agent commands duplicate job/system/study logic | Use shared startup-kit/session helpers and call existing stable `--format json` CLI contracts for optional snapshots; refactor only when there is a clear shared-helper boundary |
| Skill installer fails from a wheel because package data is missing | Add package-data tests that import resources from the built package layout |
| `agent inspect` executes user code accidentally | Restrict implementation to file reads, text scanning, and `ast`; add import-side-effect fixture tests |
| `agent inspect` leaks sensitive literals | Redact by default, never return raw file contents, and add secret-like fixture tests |
| Skills overload agent context | Tier skills and use `nvflare agent skills get` for on-demand loading |
| Agents duplicate non-idempotent operations | Require schema idempotency metadata and idempotency keys for retryable submit paths |
| Agent loses job id after a crash | Persist the idempotency key before submit and expose it in `job list`/`job meta` |
| Streaming output breaks JSON parsers | Keep one-envelope JSON for single-result commands and JSONL for streaming commands |
| Online doctor snapshot goes stale | Include timestamp/TTL and require production skills to re-check before submit when stale |
| Online doctor mutates production state | Maintain a read-only allowlist; add tests that mock mutating APIs and assert they are never called |
| Agent uses overpowered production credentials | Recommend dedicated reduced-privilege startup kits and add future per-agent scope to authorization roadmap |
| Diagnosis overclaims with partial production logs | Include per-site log availability and emit `PARTIAL_LOG_VISIBILITY` |
| Framework detection becomes a dependency trap | Static detection only; optional imports are not required |
| `agent inspect` reads outside the target root | Do not follow symlinks by default and report skipped links |
| MCP work pulls scope away from Stage 1 | Keep MCP out of this implementation plan except for future shared-contract compatibility |

## Acceptance Criteria

Stage 1 implementation is complete when:

1. A user with only `pip install nvflare` can run
   `nvflare agent skills install --agent codex` or
   `nvflare agent skills install --target <dir>`, and can access packaged
   templates without cloning GitHub.
2. `nvflare agent inspect <path> --format json` identifies framework evidence
   and FLARE conversion state without importing user code, including absolute
   data path findings and symlink skip findings.
3. `nvflare agent doctor --format json` reports local readiness and actionable
   next steps.
4. `nvflare agent doctor --online --format json` reports server/client status
   and versions through the selected startup kit without mutating the system.
5. Skills guide an agent through `python job.py`, export, POC readiness wait,
   submit, wait or JSONL monitor, logs with availability awareness, and
   download.
6. The initial PyTorch path is demonstrated with the packaged `hello-pt`
   template.
7. Exported job folders include `_export_manifest.json` and
   `job_fingerprint.json`.
8. Packaged skills pass schema linting and declare recovery categories and
   approval checkpoints, capability requirements, and maintainers.
9. Diagnosis skill returns pattern-based findings for common failures.
10. All new commands support `--schema` and the shared JSON/error contract.
11. Online doctor snapshots include timestamp/TTL and production submit
    workflows re-check stale snapshots before mutation.
12. Job submission recovery is possible through persisted idempotency keys.
13. Conversion skills create backups before editing and can revert through
    `nvflare agent skills revert`.
