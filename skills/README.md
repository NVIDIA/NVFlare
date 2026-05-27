# NVFLARE Agent Skills

This directory is the source root for NVFLARE-owned agent skills.

V1 skill directories should follow the guide-compatible shape:

```text
skills/
  nvflare-example-skill/
    SKILL.md
    references/
    evals/
      evals.json
      files/
    BENCHMARK.md
```

Milestone 1 establishes the source root and minimal frontmatter validation.
Public skill content, runtime evals, packaging, and native install/list commands
land in later milestones.

Required `SKILL.md` frontmatter fields for V1:

```yaml
---
name: nvflare-example-skill
description: Short trigger-oriented description.
min_flare_version: "2.8.0"
blast_radius: read_only
---
```

`blast_radius` must be one of:

- `read_only`
- `edits_files`
- `runs_simulator`
- `submits_poc`
- `submits_production`
