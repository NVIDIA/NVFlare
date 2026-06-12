# FLARE Agent Skill Architecture

This is a picture of what is implemented today for the `nvflare agent` skill
system: the agent-facing CLI, the packaged skills, and the install/list paths.
The benchmark harness architecture is follow-up work outside this PR's source
set.

## High-Level System View

```mermaid
flowchart TB
    Authoring["Skill Authoring Source: skills, references, eval contracts"] --> Lint["Engineering Lint Tool: module CLI plus pytest-covered checks"]
    Lint --> Package["Python Packaging Hook: setup.py and bundled_skills manifest"]
    Package --> Install["Skill Install CLI: nvflare agent skills install and list"]
    Install --> AgentHome["Agent Skill Home: Codex or Claude skill directory"]

    AgentHome --> AgentRuntime["Agent Runtime: Codex or Claude loads SKILL.md"]
    AgentRuntime --> AgentCLI["Agent-Facing NVFLARE CLI: info, inspect, doctor, skills"]
    AgentRuntime --> NVFLAREWork["NVFLARE Workflows: recipes, job.py, simulator, job CLI"]
```

## System Layers

| Layer | Implemented pieces | Purpose |
| --- | --- | --- |
| Authoring source | `skills/`, `SKILL.md`, `references/`, `evals/evals.json`, `BENCHMARK.md` | Human-readable skill instructions and supporting evidence. |
| Engineering lint tool | `nvflare.tool.agent_skill_checks`, `python -m nvflare.tool.agent_skill_checks`, pytest coverage | Deterministic admission checks for frontmatter, triggers, command drift, policy coverage, fixtures, process metrics, and doc links. The check itself is a CLI/library tool; pytest validates the tool behavior. |
| Python packaging hook | `setup.py`, `nvflare.tool.agent.bundled_skills`, `manifest.json` | Standard wheel-build hook that copies released skills into the NVFLARE package or writes an empty bundle for no-skill builds. |
| Skill install CLI | `nvflare agent skills install/list`, `skill_manager.py` | CLI copy/install tool that installs managed skills into Codex or Claude target directories with hashes, locks, backups, and symlink checks. |
| Runtime agent surface | Codex/Claude skill loading, `nvflare agent inspect`, `nvflare agent doctor`, recipe/job CLI | The agent reads skill instructions and uses NVFLARE commands to inspect, convert, validate, or diagnose. |
| Benchmark harness | Follow-up work outside this PR | Separate architecture for measuring skill impact with Docker, SDK profiles, agent plugins, and reporting. |

## Implemented Architecture

```mermaid
flowchart LR
    User["User / Benchmark Prompt"] --> Agent["Codex or Claude CLI"]

    Agent --> InstalledSkills["Agent Skill Directory: Codex CODEX_HOME skills or Claude launch add-dir"]

    NVCLI["nvflare agent CLI"] --> Info["info"]
    NVCLI --> Inspect["inspect: static AST scan"]
    NVCLI --> Doctor["doctor: readiness check"]
    NVCLI --> SkillOps["skills install/list"]

    SkillSource["repo-root skills: editable checkout"] --> SkillOps
    WheelBundle["nvflare.tool.agent.bundled_skills: wheel package bundle"] --> SkillOps
    SkillOps --> InstalledSkills

    Agent -->|follows SKILL.md| Inspect
    Agent -->|readiness| Doctor
    Agent -->|normal NVFLARE work| Recipes["NVFLARE recipes / job.py / simulator / CLI"]

    Inspect --> Project["Local training code / FLARE job artifacts"]
    Doctor --> Env["Local NVFLARE install, startup kits, optional deps, POC workspace"]
```

## Skill Source And Install Flow

```mermaid
flowchart TD
    SkillsRoot["repo-root skills/"] --> SkillDirs["nvflare-orient, nvflare-convert-pytorch, nvflare-diagnose-job, shared references"]

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
    Orient["nvflare-orient: read-only router"] --> InspectCmd["nvflare agent inspect"]
    Orient --> DoctorCmd["nvflare agent doctor"]
    Orient --> Recommend["Recommend next skill"]

    Convert["nvflare-convert-pytorch: edits files"] --> InspectCmd
    Convert --> RecipeList["nvflare recipe list"]
    Convert --> ClientAPI["Generate client.py/model.py/job.py with FLModel exchange"]
    Convert --> Validate["python job.py and export job"]

    Diagnose["nvflare-diagnose-job: read-only"] --> InspectCmd
    Diagnose --> Logs["Bounded logs / job evidence"]
    Diagnose --> Patterns["Packaged failure-pattern references"]
    Diagnose --> Cause["Likely cause + next action"]
```

## Key Implementation Points

- Public skill source: `skills/`
- Implemented skills:
  - `nvflare-orient`
  - `nvflare-convert-pytorch`
  - `nvflare-diagnose-job`
- Agent-facing CLI: `nvflare/tool/agent/agent_cli.py`
- Skill install/list logic: `nvflare/tool/agent/skill_manager.py`
- Static inspection: `nvflare/tool/agent/inspector.py`
- Readiness checks: `nvflare/tool/agent/doctor.py`
- Packaging hook: `setup.py`
- Benchmark harness architecture: follow-up work outside this PR

The important boundary: NVFLARE does not run a custom agent runtime for these
skills. It packages, installs, validates, and measures skill files that
Codex/Claude then load through their own skill mechanisms.
