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

## High-Level System View

```mermaid
flowchart TB
    subgraph DevBuild["Dev-time / build-time"]
        direction TB
        Authoring["Skill authoring source: skills, references, eval contracts"] --> Lint["Engineering lint tool: repo-local CLI plus pytest checks"]
        Lint --> Package["Packaging hook: setup.py and bundled_skills manifest"]
    end

    subgraph InstallBridge["Install-time bridge"]
        direction TB
        Install["Skill install CLI: nvflare agent skills install/list"]
    end

    subgraph Runtime["Runtime"]
        direction TB
        AgentRuntime["Agent runtime: Codex or Claude loads installed SKILL.md"] --> AgentCLI["Agent-facing NVFLARE CLI: info, inspect, doctor, skills"]
        AgentRuntime --> NVFLAREWork["NVFLARE workflows: recipes, job.py, simulator, job CLI"]
    end

    Package ==> Install
    Install ==> AgentRuntime

    style DevBuild fill:#f8fbff,stroke:#4f7fb8,stroke-width:2px
    style InstallBridge fill:#fafbfc,stroke:#334155,stroke-width:3px
    style Runtime fill:#f4fbf8,stroke:#2f7d68,stroke-width:2px
```

## System Layers

| Layer | When | Implemented pieces | Purpose |
| --- | --- | --- | --- |
| Authoring source | Dev-time | `skills/`, `SKILL.md`, `references/`, `evals/evals.json` | Human-readable skill instructions and supporting evidence. |
| Engineering lint tool | Dev-time / CI | `dev_tools/agent/skills/checks`, `python dev_tools/agent/skills/checks/cli.py`, pytest coverage | Deterministic admission checks for frontmatter, triggers, command drift, policy coverage, fixtures, and process metrics. This is a repo-local tool validated by pytest; it is not shipped in the wheel. |
| Python packaging hook | Build-time | `setup.py`, `nvflare.tool.agent.bundled_skills`, `manifest.json` | Wheel-build hook that copies released skills into the NVFLARE package, or writes an empty bundle for no-skill builds. |
| Skill install CLI | Install-time bridge | `nvflare agent skills install/list`, `skill_manager.py` | Managed installer that copies skills into Codex or Claude target directories with content hashes, locks, backups, local-modification checks, and symlink checks. |
| Runtime agent surface | Runtime | Codex/Claude skill loading, `nvflare agent inspect`, `nvflare agent doctor`, recipe/job CLI | The agent reads skill instructions and uses NVFLARE commands to inspect, convert, validate, or diagnose. |
| Benchmark harness | Separate | Follow-up work outside this PR | Separate architecture for measuring skill impact with Docker, SDK profiles, agent plugins, and reporting. |

## Lint Engine Independence (Design Invariant)

The engineering lint tool (`dev_tools/agent/skills/checks/lints.py`) is
**self-contained over the `skills/` tree**. This is a deliberate invariant, not
an accident of the current code:

- The lint engine MUST NOT read `docs/design/*.md`. Those are human planning
  docs; the admission engine validates shippable skill artifacts only. There is
  no `docs_root` parameter and no `--docs-root` flag.
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
        SkillSource["repo-root skills: editable checkout"] --> SkillOps["nvflare agent skills install/list"]
        WheelBundle["wheel bundled_skills package"] --> SkillOps
    end

    subgraph AgentRuntime["Agent runtime"]
        direction TB
        User["User / benchmark prompt"] --> Agent["Codex or Claude CLI"]
        AgentHome["Agent skill directory"] -->|loads SKILL.md| Agent
    end

    subgraph NVFLARESurface["NVFLARE command surface"]
        direction TB
        AgentCLI["nvflare agent runtime commands"] --> Info["info: command surface metadata"]
        AgentCLI --> Inspect["inspect: static AST scan"]
        AgentCLI --> Doctor["doctor: readiness check"]
        Recipes["recipes / job.py / simulator / job CLI"]
    end

    SkillOps ==> AgentHome
    Agent -->|skill-guided calls| AgentCLI
    Agent -->|normal NVFLARE work| Recipes

    Inspect --> Project["Local training code / FLARE job artifacts"]
    Doctor --> Env["Local NVFLARE install, startup kits, optional deps, POC workspace"]

    style SkillDelivery fill:#f8fbff,stroke:#4f7fb8,stroke-width:2px
    style AgentRuntime fill:#fafbfc,stroke:#334155,stroke-width:3px
    style NVFLARESurface fill:#f4fbf8,stroke:#2f7d68,stroke-width:2px
```

## Skill Source And Install Flow

```mermaid
flowchart TD
    SkillsRoot["repo-root skills/"] --> SkillDirs["nvflare-orient, nvflare-convert-pytorch, nvflare-convert-lightning, nvflare-diagnose-job, shared references"]

    SkillDirs --> ManifestBuild["build_skill_manifest: frontmatter validation and source hash"]
    ManifestBuild --> Editable["Editable source manifest"]

    SkillsRoot --> SetupPy["setup.py build_py"]
    SetupPy --> Bundle["wheel bundled_skills + manifest.json"]
    SetupPy --> EmptyBundle["empty bundled_skills manifest"]

    Editable --> FindSource["find_skill_source"]
    Bundle --> FindSource

    FindSource --> Install["nvflare agent skills install"]
    Install --> Target["Agent target skill dir"]
    Target --> InstallManifest[".nvflare_skill_install.json with managed_by, source_hash, skill_version"]

    Install --> Safety["symlink checks, lock dir, atomic staging, backup on replace, local modification detection"]
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

Dev-time / build-time:

- Public skill source: `skills/`
- Engineering lint tool and CI gate: `dev_tools/agent/skills/checks/`
- Manifest builder: `nvflare/tool/agent/skill_manifest.py`
- Packaging hook: `setup.py` (`AgentSkillsBuildPy`)

Runtime (installed on the user's machine):

- Skill install/list logic: `nvflare/tool/agent/skill_manager.py`
- Agent-facing CLI: `nvflare/tool/agent/agent_cli.py`
- Command surface metadata: `nvflare/tool/agent/command_registry.py`
- Static inspection: `nvflare/tool/agent/inspector.py`
- Readiness checks: `nvflare/tool/agent/doctor.py`
- Implemented skills: `nvflare-orient`, `nvflare-convert-pytorch`, `nvflare-convert-lightning`, and `nvflare-diagnose-job`

Separate:

- Benchmark harness architecture: follow-up work outside this PR

The important boundary is that NVFLARE does not run a custom agent runtime for
these skills. NVFLARE packages, installs, validates, and measures skill files;
Codex and Claude load those files through their own skill mechanisms.
