# NVFLARE Agent Integration Design

## Document Control

| Field | Value |
| --- | --- |
| Created date | 2026-05-25 |
| Current design date | 2026-05-26 |
| Status | Ready for Implementation |
| Supersedes | `nvflare_agent_consumer_readiness.md`, `nvflare_agent_foundation.md`, `nvflare_agent_foundation_implementation_plan.md` for current agent-integration decisions |
| Current owner | NVFLARE product/docs maintainers |
| Review scope | Runtime integration, packaged install, CLI contracts, generated-code policy, initial skill scope, safety boundaries, and links to authoring/evaluation specs |

## Table of Contents

- [Document Control](#document-control)
- [Related Specifications](#related-specifications)
- [Source Inputs](#source-inputs)
- [Problem Statement](#problem-statement)
- [Goals](#goals)
- [Non-Goals](#non-goals)
- [Key Decisions](#key-decisions)
- [Agent Readiness Scope](#agent-readiness-scope)
- [Product Skill Catalog](#product-skill-catalog)
- [Initial Seed Bundle](#initial-seed-bundle)
- [Skill File Requirements](#skill-file-requirements)
- [Skill Evaluation](#skill-evaluation)
- [Publication Handoff Boundary](#publication-handoff-boundary)
- [Command Surfaces](#command-surfaces)
- [Example and Scaffold Conventions](#example-and-scaffold-conventions)
- [Notebook-Safe Path](#notebook-safe-path)
- [Production Safety](#production-safety)
- [Auto-Research Evaluation](#auto-research-evaluation)
- [Workstreams and Deliverables](#workstreams-and-deliverables)
- [Publication Handoff Artifacts](#publication-handoff-artifacts)
- [Success Criteria](#success-criteria)
- [Open Questions and Deferred Decisions](#open-questions-and-deferred-decisions)

## Related Specifications

- [Agent Skill Authoring](agent_skill_authoring.md) owns skill-writing rules,
  helper scripts, minimal metadata, examples, eval input files, and maintenance
  policy.
- [Agent Skill Evaluation](agent_skill_evaluation.md) owns measurement,
  the initial admission gate, runtime skill-performance checks, Auto-FL research
  evaluation conventions, and publication handoff artifacts.
- Deferred roadmap notes own future mechanisms such as durable workflow state,
  receipts, provenance, transcript replay, workspace cleanup, full lifecycle
  commands, compatibility shims, PR-bot automation, and the large policy
  catalog.

## Source Inputs

This document consolidates current decisions and reviews these historical inputs
for still-relevant gaps:

- NVIDIA Agent Skills publishing guidance for externally published product skills.
- `nvflare_agent_consumer_readiness.md`.
- `nvflare_agent_foundation.md`.
- `nvflare_agent_foundation_implementation_plan.md`.

The company-level publishing guidance is the governing source for how NVIDIA
skills are validated, signed, licensed, and published externally. This design
set is the current NVFLARE source for FLARE-specific agent workflows, commands,
examples, and product behavior; the historical docs are used only to recover
still-relevant gaps.

## Problem Statement

NVFLARE is increasingly used through coding agents such as Codex, Claude Code,
Cursor, Windsurf, and similar developer-side automation tools. In this design,
the agent is not a FLARE runtime participant. The agent is a developer tool that
can inspect local training code, modify user files, generate a FLARE job, run
local validation, submit through the CLI, inspect logs, and help the user
iterate.

The current CLI work makes FLARE more scriptable, but agent readiness requires
more than command availability. Agents need:

- discoverable product-specific skills;
- stable CLI JSON, schema, and error contracts;
- deterministic local inspection and readiness checks;
- examples and scaffolds that can be copied safely;
- diagnosis workflows that do not depend on scraping human-only output;
- a publication handoff model that keeps FLARE skills aligned with FLARE
  releases while leaving external catalog registration and sync to the
  NVIDIA skills publication process.

## Goals

- Define how NVFLARE should be made ready for external coding agents.
- Separate company-level skill publishing from FLARE product-level skill
  ownership.
- Define the command, skill, example, and validation surfaces for agent
  readiness.
- Preserve the CLI as the baseline execution contract.
- Use Auto-FL research as a separate benchmark and research consumer for the
  FLARE skills, measuring whether skills improve accuracy, cost, efficiency,
  and reproducibility compared with direct agent use without skills.
- Keep Auto-FL orchestration, retry, routing, and persistence separate from
  FLARE reasoning skills.
- Keep the first implementation intentionally small. Deferred mechanics should
  stay outside this design set until a concrete failure mode or product
  requirement justifies promotion.
- Keep security boundaries explicit for production systems, startup kits,
  private keys, and site-local data.

## Non-Goals

- Make FLARE server or client processes an agentic runtime.
- Replace Client API, Recipe API, FedJob, SimEnv, or Session APIs.
- Build a fully automatic converter for arbitrary ML code without user review.
- Create a separate FLARE-only public skill marketplace that competes with
  github.com/NVIDIA/skills.
- Require agents to clone the NVFLARE GitHub repository to use released FLARE.
- Let agents bypass authorization, handle private keys unsafely, or approve
  production-impacting actions automatically.

## Key Decisions

### 1. FLARE Needs Product-Owned Skills, But Not a Separate Public Catalog

NVIDIA's public skills catalog and private NVCARPS pipeline are the external
publication and trust path. FLARE still needs a product-owned way to author,
validate, version, package, and prepare FLARE-specific skills for that
publication handoff.

The split is:

| Layer | Owner | Purpose |
| --- | --- | --- |
| FLARE product source | NVFLARE repo | Source of truth for FLARE-specific skills, examples, tests, release compatibility, and product workflow ownership |
| NVIDIA public skills catalog | `github.com/NVIDIA/skills`, public Apache-2.0 repo | Public catalog registration, external discovery, signed skill sync target, and user installation |
| NVIDIA private validation and signing pipeline | NVCARPS/NV-BASE CI/backend | Internal NVIDIA validation, scanning, signing, skill-card generation, and trust metadata |
| Optional wheel/package distribution | NVFLARE package | Convenience path for users who install `nvflare` and want matching skills without separately browsing the catalog |
| Internal-only contributor skills | NVFLARE repo, not catalog synced | Contributor workflows such as CI debug, code review, or repo maintenance that should not publish externally |

This means FLARE should prepare skills from the product repo for the public
NVIDIA skills catalog and rely on the private NVIDIA pipeline for validation
and signing when external publication is requested. The product repo is the
source of truth; the public catalog is the distribution layer; the private
pipeline is the trust and release-gate layer. Catalog registration, sync, and
public installer metadata are separate `github.com/NVIDIA/skills` integration
work, not part of the native NVFLARE installer design.

### 2. The Canonical Skill Source Should Be Repo-Root `skills/`

For external product skills, use the company-standard layout:

```text
skills/
  nvflare-orient/
    SKILL.md
    references/
    evals/
      evals.json
      files/
  nvflare-convert-pytorch/
    SKILL.md
    references/
    scripts/
    evals/
      evals.json
      files/
```

Rules:

- The directory name must match the `name` field in `SKILL.md` frontmatter.
- Skill names should be lowercase, hyphenated, product-scoped, and CLI-safe.
- FLARE product skills should use the `nvflare-` prefix to avoid collisions.
- Do not use `.claude/skills`, `.codex/skills`, or `.cursor/skills` as the
  primary source.
- Agents that expect a repo-local `.agents/skills` discovery path should resolve
  it through a symlink to `../skills`; do not maintain a parallel skill tree.
- Internal-only contributor skills should live outside the catalog-sync path,
  for example `.agents/contributor-skills/`.

Released customer-facing skills should be bundled inside the Python wheel so a
user who installs NVFLARE can install matching skills without cloning the
NVFLARE source repo. The bundled copy should be generated from the same
repo-root `skills/` source. There should not be a second hand-maintained skill
tree under `nvflare/agent/skills/`.

The source and release packages intentionally have different file sets. The
repo/source skill should keep `evals/` as development, benchmark, and
release-evidence metadata. A release/company skill package may filter those
analysis-only files out. Released runtime instructions must not depend on
filtered files, and install/list tests must prove a release-filtered skill
still installs and functions without `evals/`.

The wheel should include a minimal released-skill manifest with skill names,
versions, hashes, and FLARE-version compatibility. Native install-all uses this
manifest rather than scanning arbitrary package files. Rich catalog versioning,
changelogs, compatibility shims, and migration metadata are deferred roadmap
items.

Skill bundling is the default wheel behavior. Benchmark and CI flows that need a
no-skills control package may build the wheel with
`NVFLARE_PACKAGE_AGENT_SKILLS=0`. A disabled-skill wheel should add a wheel
build tag containing `no_skills`, such as `1no_skills`, so the control wheel is
easy to identify and does not overwrite the default wheel in the same output
directory. The wheel should still include an empty released-skill manifest so
`nvflare agent skills list/install` can report that no bundled skills are
available instead of failing on a missing package resource.

### 3. Support Packaged, Repo-Local, and Catalog Skill Use

Installing NVFLARE-owned skills should not require users to install Node.js,
`npm`, `npx`, or an external skills CLI. The primary package UX should stay in
the Python/NVFLARE toolchain:

```bash
python3 -m pip install nvflare
nvflare agent skills install --agent codex
nvflare agent skills install --agent claude
```

For installing only one skill:

```bash
nvflare agent skills install --agent codex --skill <nvflare-skill-name>
```

This command is intentionally narrower than a general skills marketplace. It
should copy NVFLARE-owned skills from the installed package by default, or from
repo-root `skills/` when running from an editable/source checkout. It should
support a small explicit set of agent targets, starting with `codex` and
`claude`, plus clear dry-run output and safe overwrite behavior. When `--skill`
is omitted, it should install all compatible released NVFLARE skills bundled
with the package or present in the selected source. It should not download
third-party skills, validate catalog signatures, or own the public NVIDIA
skills publication flow.

`--dry-run --format json` should use the same envelope as a real install with
`data.applied: false`. The `data` object should include source type, skill
versions, files to copy, target paths, conflicts found, backups planned,
deprecated skills skipped, and version deltas. A real install should report the
same plan with `data.applied: true` after changes are written.

Published skills should still appear in the public NVIDIA skills catalog after
catalog sync and NVCARPS validation. If users already have a public skills
installer, that installer may provide an alternate catalog install path, but
its exact syntax is owned by that ecosystem and is not part of the NVFLARE
contract. It is not a dependency for installing skills shipped with NVFLARE.
Integration details with `github.com/NVIDIA/skills`, including any stable
metadata exposed by future catalog installers, are separate publication work and
are not part of the native NVFLARE installer contract.
External catalog sync should follow the FLARE release that contains the matching
skills; users should not rely on catalog HEAD being newer than their installed
NVFLARE package.

FLARE should also support pre-publication and developer use directly from the
NVFLARE product repo without adding a Node dependency. For branch-specific
testing, users should install NVFLARE from that branch or work from an editable
clone, then use the same native command:

```bash
python3 -m pip install -e .
nvflare agent skills install --agent codex --skill <nvflare-skill-name>
```

If the public skills ecosystem later supports direct repo sources, it may
provide an additional path for release-candidate testing or for users who
intentionally want skills matching a specific NVFLARE branch or commit. Those
paths are not required for NVFLARE package usage, and they are not the external
trust path; catalog publication with NVCARPS validation, skill card, and
signature remains required for official release.

For a local NVFLARE clone, agents should discover skills from the checked-out
repo when working in that repo:

```text
NVFlare/
  skills/
    nvflare-orient/
      SKILL.md
  .agents/
    skills -> ../skills
```

`nvflare agent skills list` should report which FLARE skills are present and
compatible with the installed package. If the public `skills` installer
later gains first-class local-source support, `nvflare agent skills install`
may optionally delegate to it only when that dependency is already available or
explicitly requested. The default NVFLARE path should remain native Python
package behavior.

The implemented native skill-management and evaluation surface through
Milestone 7 is intentionally small:

- `nvflare agent skills install --agent codex|claude [--skill <name>]
  [--dry-run] [--format json]` copies NVFLARE-owned skills from the installed
  package or editable checkout into the selected agent's skill directory.
- `nvflare agent skills list --agent codex|claude --format json` reports the
  installed managed skills for that target and the released skills available
  from the current NVFLARE package or editable checkout.
  Its `conflicts` field is limited to target directories whose names overlap
  with NVFLARE-source skills and are not managed by NVFLARE, or managed
  NVFLARE installs with local edit/version conflicts. Existing third-party
  agent skills with unrelated names are ignored by default.
- Public runtime skill-evaluation, skill-performance, and skill-benchmarking
  subcommands are not part of this design. Benchmark evidence and report
  generation remain internal benchmark-tool concerns, not `nvflare agent` CLI
  contracts.
- `nvflare agent skills install --target <dir>` may be supported for explicit
  custom or project-local installation when named agent shortcuts are not
  enough.
- `nvflare agent skills install --local --skill <name> <path>` may be supported
  as a documented user-side workaround for a critical skill issue before an
  official NVFLARE patch release.

Full lifecycle commands such as catalog browsing, explain/audit, validate,
skills doctor, compatibility reports, changelog, uninstall, revert, report-bug,
feedback, transcript replay, and workspace cleanup are deferred to
the future roadmap. They should not be required for the first implementation.

Runtime skill-performance evidence is produced by benchmark harnesses, research
workflows, or reviewer processes that run normal agent tasks and record bounded
evidence. Installed skills do not activate a separate evaluator mode and do not
create evaluator-only artifacts during normal use.

Installer authority rules:

- The native `nvflare agent skills install` command is authoritative for
  wheel-bundled NVFLARE skills. External catalog installers are alternate
  ecosystem paths, not dependencies of the native path.
- Native installs are pinned to the installed NVFLARE package or the explicit
  editable/local source selected by the user. They do not follow public catalog
  HEAD automatically.
- Named targets initially resolve as follows: `codex` resolves to
  `$CODEX_HOME/skills` or `~/.codex/skills` when `CODEX_HOME` is unset; `claude`
  resolves to `~/.claude/skills`. `--target <dir>` overrides named resolution.
- If an external catalog installer uses the same directories, native conflict
  detection applies. If it uses a different directory, `nvflare agent skills
  list` may report it only when that location is discoverable; the native
  installer still manages only its selected `--agent` or `--target` directory.
  The native installer should not rely on external catalog metadata to classify
  unmanaged directories.
- Every native install writes a management manifest, for example
  `.nvflare_skill_install.json`, with skill name, skill version, NVFLARE
  version, source type (`wheel`, `editable`, or `local_override`), source hash,
  installed paths, and `managed_by: nvflare`.
- Reinstalling the same managed version is idempotent. Reinstalling a newer
  managed version backs up the old managed copy, then replaces it.
- If the target skill directory already exists without the NVFLARE management
  manifest, the native installer treats it as external or user-managed and
  skips it with a structured conflict finding. It does not overwrite catalog,
  `npx`, or hand-edited installs by default.
- If a managed skill's content hash differs from the manifest, reinstall skips
  by default and reports `local_modifications_detected`; the user must pass
  `--overwrite-managed` to replace it after backup.
- Adopting an external install into NVFLARE management should require an
  explicit `--adopt-external` flag; the default is to skip external
  directories.
- For wheel-bundled initial skills, `min_flare_version` is primarily lint and
  documentation metadata because the skills ship with the NVFLARE version that
  installed them. Local or editable overrides may warn when their declared
  bounds do not match the running NVFLARE package. `skills install` and
  `skills list` should surface non-blocking compatibility warnings for local,
  editable, or externally managed skill content whose declared bounds do not
  match the running NVFLARE package.
- Deprecated-skill lifecycle behavior is deferred with the broader lifecycle
  command set.

### 4. CLI Is the Execution Contract

Agents should drive FLARE administrative workflows through the CLI:

- `nvflare config` for startup kit registration and active identity selection;
- `nvflare poc` for local proof-of-concept systems;
- `nvflare job` for submit, list, wait, monitor, logs, stats, download, abort,
  clone, and delete;
- `nvflare system` for status, resources, version, restart, and shutdown;
- `nvflare study` for multi-study workflows;
- `nvflare recipe` for recipe discovery.

Agent-used commands must support:

- `--format json` with exactly one JSON envelope on stdout;
- `--format jsonl` for streaming commands, where each line is one complete JSON
  event envelope;
- `--schema` without required operational arguments;
- stable error codes and exit codes;
- schema metadata for `streaming`, `output_modes`, `idempotent`, and
  `idempotency_key_supported` when applicable;
- no required prompts on automation paths;
- human progress and diagnostics on stderr in JSON mode.

The JSON envelope should use a stable top-level shape:

```json
{
  "schema_version": "1",
  "status": "error",
  "code": "JOB_FOLDER_NOT_FOUND",
  "message": "Job folder './jobs/hello-pt' not found.",
  "hint": "Run python job.py --export --export-dir first.",
  "recovery_category": "FIXABLE_BY_CODE",
  "suggested_skill": "nvflare-diagnose-job",
  "data": null
}
```

Successful commands should use the same top-level fields with `status: "ok"`
and command-specific content under `data`.
Per-command `data` schemas are defined by each command's `--schema` output.

### 4.1 Error Recovery Contract

Multi-step agent workflows need stable recovery categories so agents know when
to retry, edit code, switch scoped identity, ask for approval, or stop. JSON
error envelopes for agent-used commands should include `recovery_category` and,
when useful, `suggested_skill` in addition to `code`, `message`, and `hint`.

| Recovery Category | Meaning | Default Agent Action |
| --- | --- | --- |
| `RETRYABLE` | Transient timeout, dropped connection, or temporary server unavailability | Retry with bounded backoff, then re-run `nvflare agent doctor --online --format json` |
| `FIXABLE_BY_CONFIG` | Wrong identity, stale startup kit, wrong study, missing local config | Adjust scoped identity/config or ask the user, then re-check |
| `FIXABLE_BY_CODE` | Bad generated code, missing import, wrong data path, invalid job folder | Re-inspect, patch or regenerate affected files, and validate again |
| `REQUIRES_USER_APPROVAL` | Production action, irreversible action, key generation, distributed approval | Surface an approval checkpoint and wait |
| `ENVIRONMENT_FAILURE` | POC partially started, port conflict, GPU unavailable, disk/resource failure | Stop and report environment remediation steps |
| `UNKNOWN` | Error does not match a known category | Collect diagnosis data and ask for human review |

Contract tests should verify that every public error code used by agent
workflows maps to a recovery category.

### 5. Skill Authoring Rules Live in the Authoring Guide

Detailed guidance for skill playbooks, granularity, naming, public frontmatter,
product sidecar metadata, helper scripts, `SKILL.md` structure, examples,
eval input files, and maintenance policy lives in
[Agent Skill Authoring](agent_skill_authoring.md). This document keeps the
runtime, package, command, catalog, and safety integration decisions.

### 6. How Skills Are Triggered and Used

Skills are loaded and selected by the coding agent, not by the FLARE runtime.
There is no expectation that a user runs a command such as `nvflare skill run`.
The user either asks a normal task question or explicitly asks the agent to use
a named skill. The agent then uses the installed skill catalog to decide which
skill content to load.

The intended flow is:

1. The user makes FLARE skills available through the NVIDIA catalog, direct
   NVFLARE repo install, or repo-local discovery from a cloned checkout.
2. The agent indexes the skill `name` and `description` frontmatter. It should
   not load every full `SKILL.md` and reference file at startup.
3. The user asks a task, such as "convert this PyTorch training script to
   FLARE", "run this exported job", or "diagnose why this job failed".
4. The agent selects a lead skill from the frontmatter match. Explicit user
   naming wins, for example "use `nvflare-diagnose-job`".
5. If the task is ambiguous, the agent uses `nvflare-orient` as the routing
   skill. `nvflare-orient` should call deterministic discovery commands such as
   `nvflare agent inspect <path> --format json` and, when environment readiness
   matters, `nvflare agent doctor --format json`.
6. The command outputs provide evidence, such as detected framework,
   conversion state, exported job status, active startup kit, server
   reachability, or authorization findings.
7. The agent loads the lead skill and only the references needed for the
   detected workflow. It may also load supporting skills when they represent a
   distinct sub-workflow, such as local validation or job lifecycle.
8. The skill tells the agent what files are safe to edit, which CLI commands or
   helper scripts to run, which artifacts to produce, and where destructive
   actions require explicit confirmation.
9. The agent executes the workflow through CLI commands, scripts, and scoped
   file edits. The skill should not rely on hidden actions outside the trace.
10. The workflow produces command logs, generated artifacts, validation output,
    and a final user-facing answer. The initial eval gate can check these visible
    outputs against the skill's mandatory and prohibited behavior IDs.

The initial handoff contract is deliberately lightweight. The active agent session
keeps conversational context, and each skill should make artifacts visible in
normal command output and the final user-facing summary. Skills that edit files
must report what they changed, which validation commands ran, and where any
generated job or support bundle was written.

Durable cross-session workflow state, `_receipt.json`, `_provenance.json`,
transcript replay, and workspace cleanup are deferred roadmap items. They
should be added only if real multi-skill or support workflows need a stable
machine-readable handoff record.

Trigger sources should be explicit:

| Trigger Source | Example | Expected Result |
| --- | --- | --- |
| Direct user naming | "Use `nvflare-diagnose-job` on job abc" | Load that skill unless it is clearly unsafe or irrelevant |
| Prompt semantic match | "Convert this PyTorch script to federated learning" | Load `nvflare-convert-pytorch` after confirming the target path |
| Routing skill decision | "Help me use FLARE on this repo" | Load `nvflare-orient`, run inspect/doctor as needed, then route to a specific workflow skill |
| `agent inspect` evidence | framework=`tensorflow`, conversion_state=`not_converted` | Use `nvflare-convert-tensorflow`, not PyTorch or generic conversion guidance |
| `agent inspect` evidence | framework=`pytorch_lightning`, conversion_state=`not_converted` | Use `nvflare-convert-lightning`, not plain PyTorch Client API conversion |
| `agent inspect` evidence | conversion_state=`exported_job` | Use `nvflare-job-lifecycle` for validation, submit, monitor, logs/stats, and download |
| `agent inspect` evidence | conversion_state=`exported_job`, framework evidence also present | Exported-job lifecycle takes priority over framework conversion; do not reconvert unless the user explicitly asks to edit source |
| `agent doctor` evidence | no active startup kit | Use setup/config guidance before job submission or remote operations |
| Negative trigger evidence | task is Kubernetes deployment | Do not trigger `nvflare-convert-pytorch` even if PyTorch appears in unrelated docs |

Skill use should have one lead skill and a small number of supporting skills.
For example:

- "Convert this PyTorch training script to FLARE and run locally" should use
  `nvflare-convert-pytorch` as the lead skill. The conversion skill owns its
  first local validation pass.
- "Submit this exported job to my remote FLARE system" should use
  `nvflare-job-lifecycle` as the lead skill, with target identity/config
  checks before submit.
- "Why did this job fail?" should use `nvflare-diagnose-job` as the lead skill
  and should collect evidence through `nvflare job meta`, `nvflare job logs`,
  `nvflare job stats`, `nvflare system status`, and `nvflare system resources`.

`nvflare-orient`, `nvflare agent inspect`, and `nvflare agent doctor` have
different roles in this trigger path:

- `nvflare-orient` is a routing skill. It interprets user intent, checks
  current context, calls inspect/doctor when needed, and routes to the right
  workflow. It is not the long-form conceptual explanation source for all FLARE
  concepts; conceptual background should live in docs and concise references
  loaded only when needed.
- `nvflare agent inspect` is a deterministic CLI command. It classifies local
  code, scripts, job source, or exported job folders.
- `nvflare agent doctor` is a deterministic CLI command. It checks local and
  optionally online environment readiness.

Skill descriptions and negative examples must be precise enough that adjacent
skills do not steal each other's prompts. Trigger evaluation should include
customer-like prompts, explicit skill-name prompts, ambiguous prompts that
should route through `nvflare-orient`, and negative prompts that should trigger
another FLARE skill or no FLARE skill.

## Agent Readiness Scope

Agent readiness includes:

- FLARE-specific published skills under repo-root `skills/`.
- Stable CLI contracts for the agent-used command set.
- `nvflare agent inspect <path>` for read-only static inspection of local user
  code, job source, and exported job folders.
- `nvflare agent doctor` for local readiness checks.
- `nvflare agent doctor --online` for a bounded read-only environment snapshot.
- Standard job generation and export conventions.
- Product examples and scaffolds that agents can safely copy.
- Diagnosis skills using job metadata, logs, stats, downloads, and system state.
- Evaluation loops, including Auto-FL research benchmarks, that measure whether
  skills improve real agent performance.

This design does not require:

- a generic automatic converter for all ML code;
- a production experiment registry;
- agent-managed production authorization.

## Product Skill Catalog

The FLARE skill catalog should be workflow-oriented and grouped by customer
intent. This table is a roadmap view, not the initial implementation scope. The
`Tier` column is the release-grouping source for this design: `seed` means the
first implementation seed, `m8` means the Milestone 8 base conversion and
customer lifecycle expansion, `next` means follow-on customer-journey skills,
and `later` means later catalog expansion. The `Category` column is also the
initial source for same-category trigger-overlap lint unless a later metadata
schema adds category directly to skills. Any change to category names or
skill-category assignment must update the evaluation lint expectations in the
same PR.
Framework conversion scope, repo evidence, and tier are canonical in
[Agent Skill Authoring](agent_skill_authoring.md#skill-granularity-and-naming).
This catalog table is a product overview and should not redefine conversion
scope.

| Category | Skill | Tier | Purpose |
| --- | --- | --- | --- |
| Orientation | `nvflare-orient` | seed | Route ambiguous requests using inspect/doctor evidence and choose the next workflow skill |
| Conversion | `nvflare-convert-pytorch` | seed | PyTorch-specific state dict, checkpoint, metrics, and recipe patterns |
| Conversion | `nvflare-convert-lightning` | m8 | PyTorch Lightning Trainer, LightningModule, logging, checkpoint, and multi-GPU patterns |
| Conversion | `nvflare-convert-huggingface` | next | Hugging Face Trainer, datasets, tokenizers, PEFT/LoRA, LLM, and VLM workflows |
| Conversion | `nvflare-convert-tensorflow` | next | TensorFlow/Keras model weight, metric, and recipe patterns |
| Conversion | `nvflare-convert-xgboost` | next | XGBoost horizontal and vertical/federated tree workflows |
| Conversion | `nvflare-convert-sklearn` | next | scikit-learn estimators and pipelines, including logistic regression, SVM, and k-means |
| Conversion | `nvflare-convert-survival-analysis` | next | Kaplan-Meier and time-to-event analytics workflows |
| Conversion | `nvflare-convert-monai` | later | MONAI medical imaging training workflows |
| Conversion | `nvflare-convert-jax` | later | JAX training workflows |
| Conversion | `nvflare-integrate-flower` | later | Interoperate with existing Flower applications |
| Conversion | `nvflare-convert-gnn` | later | Graph neural network workflows |
| Conversion | `nvflare-convert-numpy` | later | NumPy or custom training loops when no stronger framework-specific path applies |
| Job creation | `nvflare-site-specific-training` | next | Configure jobs where sites use different training scripts, data loaders, training parameters, or app folders while preserving a common FLARE task contract |
| Local runtime | `nvflare-poc-workflow` | m8 | Bridge from SimEnv validation to local server/client processes; prepare, start, verify, stop, clean, and recover local POC systems |
| Operations | `nvflare-job-lifecycle` | m8 | Validate, submit, monitor, inspect, download, collect job outputs, and check target identity/config context |
| Observability | `nvflare-experiment-tracking` | next | Add, validate, and interpret TensorBoard or MLflow experiment tracking for FLARE jobs without owning an external tracking service |
| Operations | `nvflare-system-operations` | later | Inspect and manage FLARE system state with explicit safety gates |
| Operations | `nvflare-study-workflow` | later | Work with study-scoped job routing, authorization, and data isolation |
| Deployment | `nvflare-deploy-multimachine` | later | Plan and validate multi-machine server/client/admin deployments |
| Deployment | `nvflare-deploy-docker` | later | Run FLARE components with containers and mounted startup kits |
| Deployment | `nvflare-deploy-k8s` | later | Validate Kubernetes deployment artifacts and readiness |
| Deployment | `nvflare-deploy-cloud` | later | Map FLARE deployment guidance to provider-specific cloud environments |
| Data discovery | `nvflare-federated-statistics` | later | Generate, run, and interpret federated statistics jobs before training |
| Data engineering | `nvflare-collaborative-etl` | next | Orchestrate federated/collaborative ETL, preprocessing, feature validation, and data-quality tasks without centralizing site-local data |
| Privacy/security | `nvflare-privacy-security` | later | Choose the right PET, policy, and security approach for the workflow |
| Privacy/security | `nvflare-add-differential-privacy` | later | Add DP to supported model training and report privacy budget |
| Privacy/security | `nvflare-add-homomorphic-encryption` | later | Add HE or secure aggregation to supported FLARE workflows |
| Privacy/security | `nvflare-run-private-set-intersection` | next | Run PSI (private set intersection) to privately match records before vertical FL or split learning |
| Privacy/security | `nvflare-privacy-policy-filters` | later | Configure or validate site/job privacy filters and policy scopes |
| Troubleshooting | `nvflare-diagnose-job` | seed | Collect evidence and map common failures to recovery categories |
| Troubleshooting | `nvflare-extract-local-debug-script` | later | Extract a non-federated local training/debug script from a FLARE job when diagnosing conversion or convergence issues |
| Provisioning | `nvflare-distributed-provisioning` | later | Guide request, approval, package, and startup-kit handling workflows |

Skill naming convention:

- Use `nvflare-convert-<framework>` when the workflow is primarily adapting a
  framework family, such as PyTorch, TensorFlow, XGBoost, sklearn, or JAX.
- Use `nvflare-convert-<domain-workflow>` when the workflow is described by the
  customer's domain task and has distinct data/evaluation semantics, such as
  survival analysis.
- Use `nvflare-integrate-<external-system>` when the task is interoperability
  with another FL framework or tool, such as Flower.
- Use `nvflare-<workflow>` for operational, validation, diagnosis, PET,
  security, or provisioning workflows.

Local NVFLARE installation, Python environment repair, and first-time machine
setup are not product skills in this design. A general-purpose agent can handle
those ordinary setup tasks from product documentation, and the workflow cannot
depend on NVFLARE already being installed. `nvflare-poc-workflow` starts after
the local CLI is available and owns POC prepare, start, verify, stop, and
cleanup.

`nvflare-poc-workflow` also owns local orphan-process recovery when a POC
workspace was overwritten or deleted and can no longer track the server/client
processes it started. The skill should try normal POC stop/status commands
first, then use bounded process discovery only when needed, report exact PID and
command-line evidence, and ask for explicit user confirmation before killing
any process.
Before starting or cleaning a POC workspace, it should also detect existing
local POC systems, active jobs, occupied ports, and workspace/process evidence.
It must not overwrite an active workspace or clean up a workspace with running
jobs without an explicit user choice. Starting a second local POC in a different
workspace is allowed only after reporting the currently running system and
confirming that the user intends parallel local systems.
This detailed POC cleanup/orphan-process guidance should live in a dedicated
POC reference file and be loaded only for stop, cleanup, overwrite, recovery, or
conflicting-workspace cases. Conversion skills should keep only a short handoff
note and must not load POC cleanup guidance during normal conversion or SimEnv
validation. This handoff is shared by all framework conversion skills and all
exported FLARE jobs; it is not PyTorch- or Lightning-specific.

`nvflare-site-specific-training` is the follow-on skill for heterogeneous site
workflows. It should guide agents when sites need different training scripts,
data loaders, hyperparameters, local preprocessing, or site-specific app/config
folders while still sharing a compatible task contract and aggregation path.
This is different from a normal framework-conversion skill, which assumes one
training pattern can be adapted for all sites.

`nvflare-collaborative-etl` is the follow-on skill for collaborative computing
tasks that are not model training: ETL, preprocessing, feature validation,
data-quality checks, local artifact generation, and handoff into later FLARE
training or statistics workflows. It should keep raw data local and make clear
which outputs are safe to aggregate or share.

`nvflare-experiment-tracking` is the follow-on skill for tracking and
observability. It should guide agents to use FLARE-supported tracking paths,
such as Recipe API `add_experiment_tracking`, `nvflare.client.tracking`
writers, TensorBoard analytics receivers, MLflow writers, and generated
tracking instructions. It should not become a production experiment registry or
provision/manage external TensorBoard or MLflow services; those remain user
environment concerns.

Auto-FL research is a separate research project, not part of the product
readiness surface. FLARE skills should be usable by that project as an
experimental treatment: compare Auto-FL runs with no skills, with relevant
skills available, and with specific skills forced when appropriate. Use the
results to measure whether skills improve research-task accuracy, cost,
efficiency, instruction adherence, reproducibility, and diagnosis quality.

### Auto-FL Orchestration Boundary

Auto-FL can benefit from a Codex workflow-engine-style layer, including
JavaScript-generated or JavaScript-executed workflows, but that layer should own
control-plane behavior rather than FLARE reasoning. The workflow layer should
handle:

- task polling, queueing, routing, and concurrency;
- retry/backoff policy and failed-run recovery;
- workspace naming, run IDs, manifests, and persistence;
- model/agent selection and skill availability per run;
- hooks around setup, validation, artifact collection, and cleanup;
- cost, token, runtime, and outcome ledgers.

FLARE skills should remain the reasoning layer. They should define when to use
FLARE, which integration path to choose, what safety boundaries apply, what CLI
commands and helper scripts to run, and how to interpret validation or failure
evidence. Deterministic product behavior should stay in the `nvflare` CLI or
skill-local helper scripts.

This keeps orchestration reusable across research tasks while keeping
FLARE-specific decisions versioned with the skills and package. The
Codex workflow engine should be an Auto-FL research dependency,
not a dependency for the customer path of installing NVFLARE or NVFLARE skills.

### Privacy-Enhancing Technology Skills

PET skills are deferred from the initial seed. They remain important catalog candidates. PSI
should be an explicit follow-on skill because NVFLARE already has PSI support
and examples, but DP, HE, PSI, and privacy policy filters require stronger
evidence contracts, approval guidance, dependency checks, and validation
artifacts than the first seed skills.

When PET skills are promoted, their design should define:

- the PET decision boundaries across DP, HE, PSI, privacy filters, secure
  provisioning, authorization, and communication security;
- the evidence required before the skill can claim a privacy/security workflow
  was configured or verified;
- role boundaries for researcher, project admin, org admin, and site operator;
- whether a PET-specific evidence artifact such as `_pet_report.json` is
  required.

The PET evidence artifact is deferred with the other durable workflow artifacts.
It should land when PET skills land, not in the initial seed scope.

## Initial Seed Bundle

The first implementation should ship a small seed bundle rather than the whole
catalog. The seed should prove the install path, trigger quality, inspect/doctor
handoff, conversion guidance, and diagnosis guidance before expanding into the
full customer journey.

Recommended initial seed skills:

| Category | Skill | Why It Is Useful to Customers |
| --- | --- | --- |
| Orientation | `nvflare-orient` | Routes ambiguous requests to setup, conversion, validation, POC, remote job, recipe-based, or diagnosis workflows without loading all FLARE docs |
| Conversion | `nvflare-convert-pytorch` | Covers the first high-value path for turning existing PyTorch training code into a FLARE federated workflow |
| Troubleshooting | `nvflare-diagnose-job` | Turns job/system evidence into likely failure causes and recovery categories |

For a general customer request such as "convert my training code to FLARE",
the agent should first use `nvflare-orient` and `nvflare agent inspect` to
identify the framework and conversion state. If PyTorch is detected,
`nvflare-convert-pytorch` owns the conversion workflow. A broad conversion skill
should not be added until it has a clear fallback scope that does not overlap
with framework-specific skills.

Publication sequencing should be driven by evidence, not by whether a skill is
considered core. The follow-on customer-journey bundle can add setup, job
generation, site-specific training, collaborative ETL, local validation, POC,
identity/config, job lifecycle, and experiment tracking skills after the seed
proves the authoring, install, and eval contracts.
Framework, PET, deployment, and advanced operations skills should follow the
tiering in the product catalog.

Helper-script authoring and promotion rules live in
[Agent Skill Authoring](agent_skill_authoring.md#initial-customer-helper-scripts).

## Skill File Requirements

Skill file structure, `SKILL.md` size limits, licensing, example/eval-input
relationships, and maintenance policy live in
[Agent Skill Authoring](agent_skill_authoring.md#skill-file-requirements).

## Skill Evaluation

The initial evaluation gate lives in
[Agent Skill Evaluation](agent_skill_evaluation.md). It separates normal
engineering tests from runtime agent-performance measurements. The required initial
runtime measurements are trigger correctness, negative trigger correctness,
mandatory-instruction coverage, prohibited-action avoidance, and task success
for the skill's guide-compatible eval cases.

Large policy catalogs, paired live-agent harnesses, separate instruction
monitor services, transcript replay, cost accounting, and PR-bot automation are
deferred roadmap items.

## Publication Handoff Boundary

Publication integration with `github.com/NVIDIA/skills` is separate work. This
design owns the FLARE side of the handoff: skill source, wheel packaging,
install behavior, command contracts, and evaluation evidence. The concrete
handoff checklist lives in
[Agent Skill Publication Handoff Checklist](agent_publication_handoff_checklist.md).
Catalog registration, sync, public installer metadata, and public scoreboard
mechanics belong to the NVIDIA skills publication process.

The FLARE release process must still check that skill command examples match
the installed CLI for that release before handoff. Skill drift is a product
responsibility, not something the external publication process can fully
detect.
When external catalog publication is requested, catalog sync should happen as
part of the FLARE release handoff rather than as an independent skill-catalog
cadence.

## Command Surfaces

This section is the canonical contract for agent-facing `nvflare agent`
commands. Earlier sections may mention these commands as examples or trigger
sources, but command names, safety boundaries, and output expectations should
be updated here first.

### `nvflare agent inspect`

`nvflare agent inspect <path> --format json` is a read-only static inspection
command. It inspects the user-supplied local path, not the installed NVFLARE
package and not GitHub content.

It should classify:

- training repository;
- single training script;
- FLARE job source;
- exported submit-ready FLARE job;
- mixed workspace;
- unknown target.

It should report:

- ranked framework evidence with numeric confidence, supporting evidence, and
  contradicting evidence;
  `frameworks[0]` is the framework selected for routing, not a strict
  max-confidence guarantee when framework-family rules apply, such as
  PyTorch Lightning taking priority over plain PyTorch evidence;
- likely entry points;
- whether FLARE integration usage is present;
- whether the code is `not_converted`, `partial_client_api`,
  `client_api_converted`, `flare_job`, `exported_job`, or `unknown`;
- whether `job.py` exists;
- whether SimEnv is used;
- whether export support exists;
- recipe fit as `compatible`, `requires_fedjob`, or `uncertain`;
- distributed or multi-GPU patterns such as DDP, DataParallel, FSDP,
  `torch.distributed`, `accelerate.Accelerator`, and gradient accumulation
  boundaries;
- dynamic framework-resolution findings such as Hydra/OmegaConf, dynamic
  imports, `getattr` dispatch, external training functions, and `torch.compile`;
- hardcoded absolute data path findings;
- skipped symlink findings;
- skill-selection evidence for the agent, such as detected framework,
  conversion state, exported-job state, and safety findings;
- recommended next commands.

Inspection must be static only. It may parse files and use AST-based checks, but
it must not import or execute user modules. It must not scan arbitrary
home-directory content unless explicitly pointed there, read private key
contents, modify code, generate files, submit jobs, or start/stop systems.

Inspection output must avoid leaking user source or secrets:

- never include raw file contents in JSON output.
- report structural facts, symbols, imports, call patterns, capped evidence,
  and sanitized snippets only.
- skip `.git`, virtual environments, caches, build outputs, and hidden
  directories unless the target itself is hidden.
- cap traversal and evidence volume, for example file count, individual file
  size, and evidence strings per category.
- do not follow symlinks by default; report `SYMLINK_SKIPPED` with sanitized
  target and action.
- redact secret-like literals, credentials, tokens, connection strings, cloud
  keys, private endpoints, and sensitive absolute paths by default.
- support explicit `--redact on|off`; redaction remains on by default and
  `--redact off` is for local debugging only.
- report `ABSOLUTE_DATA_PATH` with file/line and pattern type rather than full
  sensitive values when redaction is on.

### `nvflare agent doctor`

`nvflare agent doctor --format json` is a local readiness command. It should
check:

- NVFLARE import and version;
- command registry and schemas;
- active startup kit config;
- stale or invalid startup kit paths;
- optional dependency summary;
- packaged or installed skill availability;
- POC workspace state.

`nvflare agent doctor --online --format json` adds a bounded read-only server
snapshot through the active startup kit or per-command startup-kit selector. It
should include:

- active startup kit identity;
- server connection and authentication status;
- server status;
- server and client versions;
- connected clients;
- resource summary when authorized;
- job summary when authorized;
- study summary and submit capability when authorized;
- startup-kit certificate expiration and renewal hint when supported;
- packaged skill availability summary;
- snapshot timestamp and recommended TTL for using the readiness result in
  follow-on decisions;
- findings and next steps.

It must not submit, abort, delete, download, restart, shut down, switch
identity, modify config, or read private key contents.

### Native Skill Commands

`nvflare agent skills install --agent codex|claude [--skill <name>]
[--dry-run] [--format json]` is the initial management command. It copies
NVFLARE-owned skills from the installed package or editable checkout to the
selected agent target. It does not download third-party skills or depend on
Node.js, `npm`, `npx`, or an external skills CLI.

`nvflare agent skills list --agent codex|claude --format json` reports managed
installed skills for that target and the compatible released skills available
from the current NVFLARE package or editable checkout. It is a simple install
aid, not a public catalog browser. Its `conflicts` field must only describe
name-overlap conflicts with NVFLARE-source skills or managed NVFLARE installs;
unrelated third-party skills already present in the agent target directory are
ignored by default.

Installed skills collect normal task evidence for user-facing results. Runtime
skill-performance evidence is recorded by benchmark harnesses, research
workflows, or reviewer processes outside the normal skill path. This is an
agent/harness convention, not a public `nvflare agent` CLI contract.
environment switch.

Full discovery, audit, validation, compatibility, transcript, feedback,
approval-list, revert, uninstall, and cleanup commands are deferred to
the future roadmap.

### Generated Code API Selection

When an agent generates Python, it should choose the highest-level FLARE API
that can express the user's intent. The persona and task should drive the API
choice:

| User intent | Preferred generated code | Avoid |
| --- | --- | --- |
| Applied data scientist adapting a known workflow to a domain | Job Recipe API in `job.py`, plus minimal Client API changes in the training script when needed | Custom controllers/executors for standard FedAvg, FedOpt, Scaffold, XGBoost, sklearn, statistics, or similar recipe-covered workflows |
| Researcher trying variants of an existing algorithm | Recipe API first; FedJob API when the recipe needs unsupported composition or custom components | Rewriting FLARE internals or hand-editing generated config JSON when Python APIs can express the job |
| Researcher inventing a new algorithm or communication pattern | FedJob API for job assembly; ModelController API for server-side algorithm logic; Client API and `FLModel` for client training/evaluation exchange | Dropping to low-level Controller/Executor APIs before ModelController/FedJob options are exhausted |
| Platform/admin/operator task | `nvflare` CLI with `--format json` where available | Generated Python that wraps provision, config, package, submit, monitor, system, or study operations |

The default path for applied users should be Recipe API plus Client API because
that keeps the generated code close to the user's ML code and hides controller,
executor, and config wiring. The agent should use framework-specific recipes
when available, such as PyTorch/TensorFlow FedAvg recipes, XGBoost recipes,
sklearn recipes, federated statistics recipes, PSI recipes, or other product
recipes that match the detected workflow.

The default path for researchers should still start as high-level as possible.
If the research question is "try a different aggregation rule," prefer a recipe
extension point or FedJob component composition before writing a new workflow.
If the research question is "invent a new server-client algorithm," generate a
ModelController-based workflow and wire it with FedJob. Drop to lower-level
Controller or Executor classes only when the algorithm requires FLARE behavior
that ModelController, Client API, and FedJob cannot represent.

Generated Python should not become operational glue for actions already owned
by the CLI. Running POC, submitting a job, monitoring logs/stats, registering
startup kits, packaging, provisioning, and system operations should stay as
traceable `nvflare` commands. If generated Python is only wrapping a CLI call or
scraping human output, add or use JSON CLI output, a tested helper script, or a
missing product command instead.

The Python escape hatch is for job logic, local algorithm research, notebooks,
and library-first workflows where generated code is the artifact the user wants
to inspect and keep. It is not for hiding administrative side effects. Skills
may offer a Python-only path when the host agent cannot run shell commands, but
that path must state which CLI validation or submission steps the user still
needs to run outside the agent.

### Job Generation and Export

Agent-generated `job.py` should follow a standard convention:

```bash
python job.py
python job.py --export --export-dir <exported_job_root>
nvflare job submit -j <exported_job_root> --format json
```

`exported_job_root` is the submit-ready exported FLARE job directory. The
generated `job.py` must not require the agent to append a second job-name
component before submit. `python job.py` should use SimEnv for local
validation. POC and remote job submission should go through the CLI, not
through generated Python admin glue.

Export should be atomic: write to a temporary directory and move into place only
after required files are complete. As a follow-on core CLI enhancement, exported
job folders should eventually include:

- `_export_manifest.json` with required files, source path, source hash,
  timestamp, NVFLARE version, exporter, `poc_validated`, `poc_validation`, and
  a nested `fingerprint` section containing FLARE, Python, recipe, framework
  dependency, and source-hash metadata. Use one manifest file unless a separate
  consumer explicitly needs a standalone `job_fingerprint.json`.

`nvflare agent inspect` should classify current exported job folders as
`exported_job` even when these future manifest files are absent. When the
manifest/fingerprint files exist, inspect and submit preflight should use them
for stronger completeness, freshness, and validation checks.

Generated examples should make the FL ordering constraint visible: receive the
global model, apply received weights, train locally, then send the updated model
and metrics. When practical, generated jobs should write `metrics_summary.json`
with enough round-start and round-end metrics for conversion and job-lifecycle
skills to
detect semantic mistakes such as ignoring the received global model.

Conversion skills and `nvflare-job-lifecycle` must document SimEnv limitations:
no TLS/auth, no startup-kit validation, no network latency or timeout coverage,
no real site-local data separation unless the job explicitly models it, and no
guarantee that per-site resource limits match a remote deployment. POC
validation is a recommended local system-level check before remote submission.
POC workflows should record the previous kit identity before prepare and
restore it after validation when a supported restore path exists. A future CLI
may expose this as a direct stop-and-restore option, but Milestone 8 skills
must not require an unimplemented restore command.

### Diagnosis

The initial diagnosis workflow can be skill-driven. `nvflare-diagnose-job`
should collect:

```bash
nvflare job meta <job_id> --format json
nvflare job logs <job_id> --site all --tail 200 --format json
nvflare job stats <job_id> --format json
nvflare system status --format json
nvflare system resources --format json
```

Diagnosis must keep evidence bounded and source-aware:

- request bounded logs with `--tail`, `--since`, or `--max-bytes` where
  available.
- treat `logs_truncated: true` as a finding when more context may be needed.
- report `PARTIAL_LOG_VISIBILITY` when per-site logs are unavailable or
  permission-denied.
- use log source markers when available: `[USER_CODE_EXCEPTION]` for user
  training code, `[FLARE]` for framework/runtime code, and `unknown` when no
  marker exists.
- match a packaged failure-pattern catalog before asking an LLM to interpret
  raw logs.
- avoid confident root-cause claims when required site evidence is missing.

The first `nvflare-diagnose-job` release should include at least 12 packaged
failure patterns across user-code exceptions, missing imports, data-path
failures, CUDA/resource exhaustion, auth/startup-kit issues, connection
timeouts, partial log visibility, export/package mistakes, and site
authorization failures. Auto-FL and customer support feedback should grow this
catalog over time.

The skill should map common findings to recovery categories:

| Pattern | Evidence | Recovery Category |
| --- | --- | --- |
| `CUDA_OOM` | CUDA out of memory in logs | `FIXABLE_BY_CODE` |
| `AUTH_FAILURE` | certificate or authentication rejection | `FIXABLE_BY_CONFIG` |
| `ROUND_TIMEOUT` | round timeout or no client response | `ENVIRONMENT_FAILURE` |
| `IMPORT_ERROR` | `ModuleNotFoundError` or import failure | `FIXABLE_BY_CODE` |
| `ABSOLUTE_DATA_PATH` | local absolute path used remotely | `FIXABLE_BY_CODE` |
| `STARTUP_KIT_EXPIRED` | certificate validity failure | `FIXABLE_BY_CONFIG` |

A later `nvflare job diagnose <job_id> --format json` command can formalize the
same evidence collection once the patterns are stable.

## Example and Scaffold Conventions

Example layout, README expectations, eval-input source-of-truth rules, and
recipe-to-skill scaffold guidance live in
[Agent Skill Authoring](agent_skill_authoring.md#example-and-scaffold-conventions).

## Notebook-Safe Path

Many applied users will run NVFLARE from notebooks while an agent edits files or
drives shell commands. Notebook workflows need extra guardrails because kernels
hide global state and long blocking commands can make recovery harder.

Notebook guidance:

- avoid changing the global active startup kit inside notebooks; prefer
  per-command `--kit-id` or `--startup-kit` selectors.
- verify that the notebook kernel and shell commands use the same NVFLARE
  installation with `nvflare --version`.
- use polling commands such as `nvflare job meta <job_id> --format json` or
  bounded monitor intervals instead of blocking the kernel indefinitely.
- after kernel restart, run `nvflare agent doctor --online --format json`
  before preparing POC again, switching identity, or submitting jobs.
- after kernel restart, rediscover artifact state with bounded inspect/doctor
  commands instead of relying on hidden notebook state.
- provide a notebook example that demonstrates safe POC validation, scoped
  startup-kit selection, export, submit, bounded monitoring, logs, download, and
  cleanup.

Notebook guidance is not a separate public skill in the initial catalog. It
belongs in `nvflare-job-lifecycle`, conversion skills, and concise notebook
references. A future
`nvflare-notebook-workflow` skill should be added only if notebook use develops
distinct triggers, artifacts, safety boundaries, and eval inputs/assertions that would
overlap poorly with those existing skills.

## Production Safety

Agent workflows must preserve FLARE's security boundaries:

- use `nvflare config add`, `nvflare config use`, `nvflare config inspect`, and
  per-command `--kit-id` or `--startup-kit` selectors for startup kit identity;
- do not copy private keys into generated artifacts;
- redact secrets from logs and JSON outputs;
- treat generated job code as user-reviewable source;
- keep site-local data paths local and never assume globally visible data.

## Auto-Research Evaluation

Auto-FL and auto-research evaluation conventions live in
[Agent Skill Evaluation](agent_skill_evaluation.md#auto-fl-research-evaluation).
This integration doc only defines the boundary: Auto-FL can consume FLARE
skills, but orchestration, retry, routing, and persistence remain outside the
customer skill install path.

## Workstreams and Deliverables

The implementation work spans the three documents:

| Workstream | Owning Spec | Scope |
| --- | --- | --- |
| Runtime commands | This document | `nvflare agent inspect`, `doctor`, native skill install/list, CLI JSON/jsonl contracts, and generated-code API policy |
| Skill source and authoring | [Agent Skill Authoring](agent_skill_authoring.md) | repo-root `skills/`, `SKILL.md`, `references/`, helper scripts, examples, eval input files, minimal metadata, and maintenance policy |
| Skill evaluation and admission | [Agent Skill Evaluation](agent_skill_evaluation.md) | guide-compatible trigger/task evals, initial admission gate, engineering-vs-runtime test split, and Auto-FL research evaluation conventions |
| Publication handoff artifacts | [Agent Skill Evaluation](agent_skill_evaluation.md#publication-handoff-boundary) | FLARE skill content, benchmark evidence, and handoff artifacts consumed by the NVIDIA skills publication process |
| Diagnosis productization | This document plus [Agent Skill Authoring](agent_skill_authoring.md#initial-customer-helper-scripts) | `nvflare-diagnose-job`, bounded evidence collection, failure-pattern catalog, helper scripts, and later promotion to `nvflare job diagnose` |
| Deferred roadmap | Future roadmap notes | receipts, provenance, workflow state, transcript replay, workspace cleanup, full lifecycle commands, compatibility shims, PR bot, and large policy catalog |

## Publication Handoff Artifacts

Skill content, initial evaluation evidence, benchmark summaries, and publication
handoff artifact requirements live in
[Agent Skill Evaluation](agent_skill_evaluation.md#publication-handoff-boundary).
Public scoreboards and external skill-card mechanics belong to the
company-wide NVIDIA skills publication process, not FLARE agent integration.

## Success Criteria

Agent readiness is successful when:

- A user can install NVFLARE and obtain FLARE-specific skills from the wheel
  without needing a separate NVFLARE source checkout.
- A developer can test FLARE skills from the public NVFLARE GitHub repo or a
  local NVFLARE clone before the skills are synced into the public catalog.
- FLARE skills pass the initial evaluation gate before external publication handoff.
- FLARE handoff artifacts are ready for the NVIDIA skills publication process
  when external catalog publication is requested.
- An agent can inspect a simple PyTorch training repo, select the right FLARE
  skill, apply the appropriate FLARE integration pattern, generate `job.py`, run
  SimEnv, export a job folder, and submit through `nvflare job submit`.
- An agent can prepare POC, wait for readiness, submit an exported job, monitor
  it, collect logs and results, and diagnose a deliberate failure.
- Job submission reports the active startup kit identity and target context
  before submission.
- Skill examples remain aligned with the installed CLI for the corresponding
  FLARE release.

## Open Questions and Deferred Decisions

No known blocker remains for the first implementation pass. Deferred roadmap
items are tracked separately, including durable handoff artifacts, transcript
replay, workspace cleanup, full lifecycle commands, compatibility shims, PR-bot
automation, and the large policy catalog.

Open non-blocking decisions:

- which additional agent targets beyond `codex` and `claude` should be added
  after their skill locations, capability model, and trace capture are known;
- how many diagnosis patterns are required for each later release after Auto-FL
  and customer feedback produce real failure data.
