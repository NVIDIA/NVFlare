# FLARE Agent Skill Architecture

This document describes the implemented `nvflare agent` skill system: the
agent-facing CLI, packaged skills, and install/list paths.
The benchmark harness architecture is follow-up work outside this PR's source
set.

The implementation has two distinct scopes. **Dev-time and build-time**
components author, validate, and package skills inside the repository. **Runtime**
components are the commands and files an end user's agent uses. The skill install
CLI is the bridge: it takes validated skill content from the repo or wheel bundle
and installs it into an agent-specific skill directory.

## What This Is (Core Model)

An **agent skill** is a procedural playbook — a Markdown `SKILL.md` plus
supporting `references/` — that guides a general coding agent (Codex, Claude,
etc.) through an NVFLARE workflow such as converting training code into a
federated job. Skills are guidance, not a runtime library.

The design rests on one principle: **skills own the procedure; the NVFLARE
product owns the truth.**

- The **skill** tells the agent *how to operate*: what to inspect, which CLI
  commands to run, what to extract, which files to edit, when to stop, and what
  evidence to report.
- The **product API** owns *what is true and executable*: recipe schema and
  defaults, argument validation, job construction, simulation, and export. The
  agent discovers this at runtime through commands like
  `nvflare recipe show --format json` and lets recipe constructors validate.

A skill must never re-encode product behavior in prose — no hand-authored recipe
schemas, no reimplemented validation, no hidden fact-to-parameter tables. That
"shadow API" would drift from the product and mislead the agent. When guidance
would also be needed by a human writing the job by hand, it belongs in the
product, not the skill.

NVFLARE does **not** run its own agent runtime. It authors, lints, packages,
installs, and measures skill files; the agent host (Codex/Claude) loads and
executes them. The packaged skills today are:

- `nvflare-orient` — read-only router that inspects a project and recommends the
  next skill or workflow;
- `nvflare-convert-pytorch` and `nvflare-convert-lightning` — convert existing
  PyTorch / PyTorch Lightning training code into a federated NVFLARE job via the
  Client API, with local validation and export;
- `nvflare-diagnose-job` — read-only diagnosis of a failed job from bounded
  evidence.

The rest of this document details how those skills are authored, validated,
packaged, installed, and consumed.

## Responsibility Layers (LLM / Skill / Product API)

The core architecture is a separation of responsibilities across three layers.
This is *who owns what at runtime*, and is distinct from the delivery/lifecycle
layers (dev-time, build-time, install, runtime) in [System Layers](#system-layers)
below.

```mermaid
flowchart TB
    subgraph LLM["LLM / Agent — interpretation"]
        direction TB
        L["Read request; statically inspect source; extract facts; edit code; invoke product commands; report evidence"]
    end
    subgraph Skill["Skill — procedure and constraints"]
        direction TB
        S["Envelope; fields to extract; commands to call; validation ladder; stop/approval conditions; reporting shape"]
    end
    subgraph Product["Product API — authoritative behavior"]
        direction TB
        P["Recipe catalog, schema, defaults; argument validation and invariants; job construction, simulation, export; CLI contracts"]
    end

    LLM -->|follows the procedure in| Skill
    LLM -->|discovers, validates, executes via| Product
    Product -->|schema, validation errors, results| LLM
    Skill -.->|defers schema/validation to, never re-encodes| Product

    style LLM fill:#f8fbff,stroke:#4f7fb8,stroke-width:2px
    style Skill fill:#fafbfc,stroke:#334155,stroke-width:3px
    style Product fill:#f4fbf8,stroke:#2f7d68,stroke-width:2px
```

| Layer | Owns | Must not |
| --- | --- | --- |
| LLM / Agent | Natural-language interpretation, source inspection, code edits, running product commands, reporting | Invent product capability that does not exist |
| Skill | Procedure: envelope, fields to extract, commands to call, validation ladder, stop/approval conditions, reporting shape | Own schema, defaults, or validation; re-encode product behavior as a "shadow API" |
| Product API | Recipe schema/defaults, validation and invariants, job construction, simulation, export, CLI contracts | — |

The load-bearing rule is the arrow from Skill to Product: the skill *defers*
schema and validation to the product and never re-encodes them. The engineering
lints (below) enforce this boundary at admission time.

## High-Level System View

```mermaid
flowchart TB
    subgraph DevBuild["Dev-time"]
        direction TB
        Authoring["Skill authoring source: skills, references, eval contracts"] --> Lint["Engineering lint tool: repo-local CLI plus pytest checks"]
    end

    subgraph InstallBridge["Install-time bridge"]
        direction TB
        Install["Skill install: npx skills add ./skills -a claude-code -a codex"]
    end

    subgraph Runtime["Runtime"]
        direction TB
        AgentRuntime["Agent runtime: Codex or Claude loads installed SKILL.md"] --> AgentCLI["Agent-facing NVFLARE CLI: info, inspect"]
        AgentRuntime --> NVFLAREWork["NVFLARE workflows: recipes, job.py, simulator, job CLI"]
    end

    Authoring ==> Install
    Install ==> AgentRuntime

    style DevBuild fill:#f8fbff,stroke:#4f7fb8,stroke-width:2px
    style InstallBridge fill:#fafbfc,stroke:#334155,stroke-width:3px
    style Runtime fill:#f4fbf8,stroke:#2f7d68,stroke-width:2px
```

## System Layers

| Layer | When | Implemented pieces | Purpose |
| --- | --- | --- | --- |
| Authoring source | Dev-time | `skills/`, `SKILL.md`, `references/` (runtime); `dev_tools/agent/skill_evals/<skill>/` (repo-only eval suites) | Human-readable skill instructions and supporting evidence; eval suites live outside the shipped skill tree. |
| Engineering lint tool | Dev-time / CI | `dev_tools/agent/skills/checks`, `python dev_tools/agent/skills/checks/cli.py`, pytest coverage | Deterministic admission checks for frontmatter, triggers, command drift, policy coverage, fixtures, and process metrics. This is a repo-local tool validated by pytest; it is not shipped in the wheel. |
| Skill install | Install-time bridge | `npx skills add ./skills -a claude-code -a codex` (local) or `npx skills add NVIDIA/<skills-repo> -a claude-code -a codex` (published) | Standard [agentskills.io](https://agentskills.io) installer that copies the `skills/` tree into the Codex and Claude skill directories. Install the whole set together so cross-skill references (`nvflare-shared/`) resolve. NVFLARE ships no custom installer command. |
| Runtime agent surface | Runtime | Codex/Claude skill loading, `nvflare agent inspect`, recipe/job CLI | The agent reads skill instructions and uses NVFLARE commands to inspect, convert, validate, or diagnose. `nvflare agent inspect` also reports installed skills discovered from the agent skill directories. |
| Benchmark harness | Separate | Follow-up work outside this PR | Separate architecture for measuring skill impact with Docker, SDK profiles, agent plugins, and reporting. |

## Lint Engine Independence (Design Invariant)

The engineering lint tool (`dev_tools/agent/skills/checks/lints.py`) is
**independent of `docs/design/` and offline catalog docs**. Its intentional
inputs are the runtime `skills/` tree and the repo-only eval suites under
`dev_tools/agent/skill_evals/<skill>/` (via `evals_root` / `--evals-root`). This
is a deliberate invariant, not an accident of the current code:

- The lint engine MUST NOT read `docs/design/*.md`. Those are human planning
  docs; there is no `docs_root` parameter and no `--docs-root` flag. It MAY read
  the repo-only eval suites under `evals_root` — those are dev/QA tooling input,
  explicitly distinct from the forbidden `docs_root`.
- Separate the two input surfaces by check type: **runtime-boundary checks**
  validate shippable artifacts only (`skills/`, `SKILL.md`, `references/`,
  `assets/`, and the internal `skills/nvflare-shared/` skill) and reject embedded
  `evals/` directories, while
  **trigger, coverage, process-metric, and fixture checks** deliberately consume
  the repo-only eval suites under `evals_root` to verify positive/negative
  trigger coverage, global-negative coverage, policy coverage behavior IDs, and
  fixtures. Eval suites are never shipped in the wheel.
- `SKILL.md` is a **runtime artifact** loaded by the agent. Frontmatter fields
  must be runtime or public skill metadata. Do not add offline-lint-only fields
  to its frontmatter.
- `skill-trigger-overlap-lint` groups skills by deterministic skill-name
  families, such as `nvflare-convert` or `nvflare-diagnose`
  (`_trigger_overlap_group`), so the overlap guard runs purely over `skills/`
  with no external source of truth.
- Catalog/publication sync (is a skill listed in the human product catalog?) is
  a **docs concern**, not a skill-admission concern. If wanted, it belongs in a
  separate docs check that the skill engine does not import. Do not add
  `skill-catalog-category-lint` or `agent-doc-crosslink-lint` back into this
  engine.

History: an earlier change coupled the engine to `agent_integration.md` for
catalog synchronization and silently skipped checks in CI when the docs were
absent. That coupling was reverted on purpose. `category` is now public
frontmatter metadata for publishable skills, but the lint engine still must not
use design docs or category values as its trigger-overlap source of truth. If
the engine needs a grouping source, derive it from `skills/` deterministic
skill-name families or an analysis-only file stripped from the release bundle.

## Implemented Architecture

```mermaid
flowchart TB
    subgraph SkillDelivery["Skill delivery"]
        direction TB
        SkillSource["repo-root skills/ (or published repo)"] --> SkillOps["npx skills add -a claude-code -a codex"]
    end

    subgraph AgentRuntime["Agent runtime"]
        direction TB
        User["User / benchmark prompt"] --> Agent["Codex or Claude CLI"]
        AgentHome["Agent skill directory"] -->|loads SKILL.md| Agent
    end

    subgraph NVFLARESurface["NVFLARE command surface"]
        direction TB
        AgentCLI["nvflare agent runtime commands"] --> Info["info: command surface metadata"]
        AgentCLI --> Inspect["inspect: static AST scan + installed-skill discovery"]
        Recipes["recipes / job.py / simulator / job CLI"]
    end

    SkillOps ==> AgentHome
    Agent -->|skill-guided calls| AgentCLI
    Agent -->|normal NVFLARE work| Recipes

    Inspect --> Project["Local training code / FLARE job artifacts"]

    style SkillDelivery fill:#f8fbff,stroke:#4f7fb8,stroke-width:2px
    style AgentRuntime fill:#fafbfc,stroke:#334155,stroke-width:3px
    style NVFLARESurface fill:#f4fbf8,stroke:#2f7d68,stroke-width:2px
```

## Skill Source And Install Flow

```mermaid
flowchart TD
    SkillsRoot["repo-root skills/ (or published NVIDIA/<skills-repo>)"] --> SkillDirs["nvflare-orient, nvflare-convert-pytorch, nvflare-convert-lightning, nvflare-diagnose-job, nvflare-shared (internal)"]

    SkillDirs --> Lint["Engineering lint tool: dev_tools/agent/skills/checks (frontmatter validation, admission checks)"]

    SkillsRoot --> Install["npx skills add ./skills -a claude-code -a codex"]
    Install --> Target["Agent skill dirs (.claude/skills, .agents/skills, ~/.claude/skills, ~/.codex/skills)"]
    Target --> Discover["nvflare agent inspect installed_skills discovery"]

    Install --> WholeSet["Install the whole set together so nvflare-shared relative refs resolve"]
```

## What The Skills Actually Do

```mermaid
flowchart LR
    Project["User project or NVFLARE artifacts"] --> Orient["nvflare-orient: read-only router"]
    Project --> Convert["nvflare-convert-pytorch / nvflare-convert-lightning: conversion skills"]
    Project --> Diagnose["nvflare-diagnose-job: read-only diagnosis"]

    Orient --> OrientInspect["Inspect project shape and FLARE readiness"]
    Orient --> OrientDecision["Recommend the next concrete FLARE skill or workflow"]

    Convert --> ConvertInspect["Statically inspect model, data loading, requirements, and metrics"]
    Convert --> RecipeList["Discover recipes with nvflare recipe list/show"]
    Convert --> ClientAPI["Generate or update client.py, model.py, job.py (+ aggregators.py)"]
    Convert --> Validate["Validate locally, export, and report metric/artifact evidence"]

    Diagnose --> Evidence["Collect bounded logs, configs, and run artifacts"]
    Diagnose --> Patterns["Match packaged failure-pattern references"]
    Diagnose --> Cause["Report likely cause, confidence, and next action"]
```

## Key Implementation Points

Dev-time:

- Public skill source: `skills/`
- Engineering lint tool and CI gate: `dev_tools/agent/skills/checks/`

Install-time:

- Standard installer: `npx skills add ./skills -a claude-code -a codex` (local)
  or `npx skills add NVIDIA/<skills-repo> -a claude-code -a codex` (published);
  NVFLARE ships no custom install command.

Runtime (installed on the user's machine):

- Agent-facing CLI: `nvflare/tool/agent/agent_cli.py`
- Command surface metadata: `nvflare/tool/agent/command_registry.py`
- Static inspection engine: `nvflare/tool/agent/inspector.py` (framework-agnostic
  AST walk and evidence ranking; also reports installed skills discovered from
  agent skill directories)
- Per-framework detectors: `nvflare/tool/agent/frameworks/` (one module per
  framework; add a framework here, not in the engine)
- Implemented skills: `nvflare-orient`, `nvflare-convert-pytorch`, `nvflare-convert-lightning`, and `nvflare-diagnose-job`

Separate:

- Benchmark harness architecture: follow-up work outside this PR

The important boundary is that NVFLARE does not run a custom agent runtime for
these skills. NVFLARE packages, installs, validates, and measures skill files;
Codex and Claude load those files through their own skill mechanisms.
