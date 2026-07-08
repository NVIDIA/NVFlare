# NVFLARE Agent Skills

This directory contains NVFLARE-owned agent skills for supported coding agents.

Each skill lives in its own directory with a `SKILL.md` file and its supporting
references. This directory ships as runtime content only:

```text
skills/
  nvflare-your-skill/
    SKILL.md
    references/
```

Eval suites (grading-oracle data, not runtime guidance) live outside this tree,
one directory per skill name, and are never packaged into installed skills:

```text
dev_tools/agent/skill_evals/
  nvflare-your-skill/
    evals.json
    files/
```

`nvflare-shared/` is an internal, non-triggered skill: it holds references and
templates shared by the other skills and is installed alongside them, but it is
not user-selectable. It still follows the skill structure (a valid `SKILL.md`
with `status: internal`, plus `references/` and `assets/`).

`SKILL.md` frontmatter follows the [agentskills.io spec](https://agentskills.io/specification):
only `name`, `description`, `license`, `compatibility`, `metadata`, and
`allowed-tools` are allowed at the top level. NVFLARE's required fields
(`min_flare_version`, `blast_radius`, and public-skill `category`) are nested
under the `metadata:` map:

```yaml
---
name: nvflare-your-skill
description: Short trigger-oriented description.
metadata:
  min_flare_version: "2.8.0"
  blast_radius: read_only
  category: Orientation
---
```

The skill name above is illustrative; actual skill directories use their
published skill names. Do not place NVFLARE custom fields at the top level;
the skill validator rejects them unless they are nested under `metadata:`.

Public skills must include `category` under `metadata:` as product-facing
runtime metadata. Draft, internal, and private skills (`metadata.status`) may
omit it while they are not publishable.

`blast_radius` must be one of:

- `read_only`
- `edits_files`
- `runs_simulator`
- `submits_poc`
- `submits_production`

## Installing the skills

NVFLARE skills are installed with the standard [`skills`](https://agentskills.io)
tool via `npx skills add`. Both `claude-code` and `codex` are supported agent
targets. Install the whole set together so cross-skill references resolve:
`nvflare-shared/` is loaded by the other skills through relative references, so
installing skills individually can leave those references dangling.

From a local checkout (pre-publish):

```bash
npx skills add ./skills -a claude-code -a codex
```

From the published repository (no manual `git clone` needed — `npx skills add`
fetches the repo itself; append `#<branch>` to install from a specific branch):

```bash
npx skills add NVIDIA/<skills-repo> -a claude-code -a codex
```

Installation is git-based and does not depend on `pip install nvflare`; the
skills are not shipped inside the Python wheel. Pass every agent you use with
repeated `-a` flags. Omitting an agent skips installation for that agent; there
is no NVFLARE-specific installer command.
