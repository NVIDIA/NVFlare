# NVFLARE Agent Skills

This directory contains NVFLARE-owned agent skills for supported coding agents.

Each skill lives in its own directory with a `SKILL.md` file. Skills may also
include supporting references and evaluation fixtures:

```text
skills/
  nvflare-your-skill/
    SKILL.md
    references/
    evals/
      evals.json
      files/
```

Directories whose names start with `_`, such as `_shared/`, are reference-only
support content. They are not installable skills and do not contain `SKILL.md`.

Required `SKILL.md` frontmatter fields:

```yaml
---
name: nvflare-your-skill
description: Short trigger-oriented description.
min_flare_version: "2.8.0"
blast_radius: read_only
category: Conversion
---
```

The skill name above is illustrative; actual skill directories use their
published skill names.

`blast_radius` must be one of:

- `read_only`
- `edits_files`
- `runs_simulator`
- `submits_poc`
- `submits_production`
