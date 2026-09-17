# Provenance and acknowledgement

This skill is designed for an NVFlare-based Auto-FL harness that intentionally adapts ideas from the public `karpathy/autoresearch` repository.
It also adapts the public Camyla project's literature-review / QWBE-style research-loop ideas at the instruction and artifact level.

## Public ideas adapted
- `program.md` as the main agent-facing control plane
- a small bounded edit surface
- run setup through a branch-per-campaign workflow
- `results.tsv` as a run ledger
- redirected `run.log`
- keep / discard / revert experiment discipline
- the framing that `program.md` behaves like a lightweight skill

## Honesty note
The public `karpathy/autoresearch` repository does not appear to ship `AGENTS.md` or `CLAUDE.md` files. If those files exist in the target repo, treat them as new repo-local wrappers around the same control-plane philosophy rather than as upstream files copied from autoresearch.

## What this skill should acknowledge
When you update docs or prompts in the Auto-FL repo, explicitly acknowledge:
1. the NVFlare example lineage for the FL substrate
2. the public `karpathy/autoresearch` lineage for the repo-level operating model
3. the public Camyla lineage for literature search, challenge extraction, proposal scoring, QWBE-style branch allocation, and reflective memory
4. whether text was copied verbatim, lightly adapted, or newly written
