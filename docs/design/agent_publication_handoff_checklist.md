# Agent Skill Publication Handoff Checklist

This document defines the FLARE-owned handoff package for external NVIDIA
skills publication. It is a checklist for artifacts produced in the NVFLARE
repository before catalog sync, NVCARPS validation, signing, or public
scoreboard publication.

External catalog registration, signing infrastructure, public installer
metadata, and public scoreboard mechanics are owned by the NVIDIA skills
publication process. FLARE owns the source, release compatibility, and evidence
bundle that the publication process consumes.

## Handoff Inputs

Each public FLARE skill submitted for external publication should provide:

- skill source directory under `skills/<skill>/`;
- valid `SKILL.md` frontmatter with `name`, `description`,
  `min_flare_version`, and `blast_radius`;
- `references/`, `scripts/`, `assets/`, and `evals/files/` needed by the skill;
- `evals/evals.json` with `schema_version`, positive trigger cases, adjacent
  negative trigger cases, mandatory behavior IDs, prohibited behavior IDs, and
  process metric contracts;
- an explicit draft/internal marker in handoff metadata when runtime evidence is
  not yet publication-grade;
- source hash, skill version when present, and target NVFLARE release version;
- command-drift lint output for every referenced `nvflare` command and flag;
- helper-script test output when the skill ships scripts;
- runtime evidence summaries from the agent benchmark harness, research
  workflow, or reviewer process when available.

## Evidence Mapping

| Publication Question | FLARE Artifact |
| --- | --- |
| What is the skill and when should it trigger? | `SKILL.md` frontmatter and trigger/use-boundary text |
| Which FLARE release supports it? | `min_flare_version`, target release notes, bundled wheel metadata |
| What source content is being published? | source hash, packaged file manifest, skill directory |
| Does the skill reference valid product commands? | command-drift lint output and CLI `--schema` checks |
| Does it avoid adjacent or unrelated prompts? | `negative_trigger_cases`, global negative coverage, runtime trigger evidence when available |
| Does it follow required workflow rules? | `nvflare.mandatory_behavior` IDs and runtime evidence records |
| Does it avoid prohibited actions? | `nvflare.prohibited_behavior` IDs and runtime evidence records |
| Does it improve agent behavior? | benchmark records and process metric summaries |
| Are helper scripts safe and tested? | helper-script tests and static inspection results |

## Handoff Output

The FLARE release handoff should produce a compact bundle or manifest that
names:

- skill names and versions included in the handoff;
- target NVFLARE release and commit;
- source hashes for each skill;
- lint/test command summaries;
- runtime evidence record locations or summaries;
- known limitations or publication-blocking gaps.

NVCARPS or the external catalog pipeline may transform these inputs into a skill
card, signed manifest, catalog entry, public installer metadata, or scoreboard
entry. Those transformed artifacts are not the source of truth inside NVFLARE.

## Publication Gates

A skill should not be handed off for public catalog publication when:

- required frontmatter is missing or invalid;
- `evals/evals.json` is missing, has an unsupported `schema_version`, or lacks
  positive and adjacent negative trigger coverage;
- referenced `nvflare` commands drift from the target release CLI schema;
- helper scripts are untested or violate JSON stdout/stderr conventions;
- runtime evidence shows unresolved significant safety or artifact-ownership
  violations;
- the target NVFLARE release does not satisfy the skill compatibility metadata.

Runtime benchmark absence can be accepted only with an explicit draft/internal
marker or release-owner approval that documents why publication is still
appropriate.
