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
