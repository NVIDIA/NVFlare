# NVFlare Auto-FL Skill Design

## Summary

Auto-FL should enter NVFlare as a skill-first product experience. Users select
an official NVFlare Auto-FL skill in a coding agent, point it at an existing
`job.py`, and state the optimization objective, environment, and budget. NVFlare
owns deterministic import, execution truth, policy boundaries, artifacts, and
reproducibility. The agent owns candidate planning, code edits within allowed
paths, experiment comparison, and narrative reporting.

This avoids introducing a new public Auto-FL command tree while still making
Auto-FL an NVFlare-owned feature.

## Product Boundary

The first production-oriented slice includes:

- A bundled `nvflare-autofl` agent skill.
- A deterministic `job.py` importer that emits reviewable `autofl.yaml`.
- A trust contract in `autofl.yaml` showing extracted facts, unresolved fields,
  and allowed edit paths.
- Documentation for using the skill with simulation, POC, and production
  environments through existing NVFlare surfaces.

The first version does not embed or vendor a coding agent, and it does not add a
public Auto-FL command family.

## Deterministic Import

The importer parses Python source with `ast`; it does not import or execute
user code. It supports known Recipe and FedJob-style patterns first:

- Recipe/FedJob constructor and class import.
- `SimEnv`, `PocEnv`, and `ProdEnv` references.
- `train_script` resolution for literal and argparse-derived values.
- Objective metric from user request, `key_metric`, or explicit unresolved
  default.
- Fixed-budget fields such as rounds, clients, and candidate budget.
- Common argparse tunables from `job.py` and the resolved train script.

Unsupported or dynamic fields are carried forward as unresolved review items
instead of being guessed by the importer or the agent.

## Trust Contract

Every import result includes:

- `import`: importer version, source path, source hash, support status, and
  confidence.
- `job`: surface, entrypoint, allowed edit paths, train script, and call
  arguments with provenance.
- `objective`, `budget`, `environment`, and `search_space`.
- `trust_contract`: extracted facts, unresolved fields, allowed edit paths, and
  agent controls.

The skill must present extracted, unresolved, and allowed sections before it
runs candidates. This is the core product guardrail: NVFlare makes the workflow
reviewable and reproducible; the agent makes it interactive.

## Execution Model

The skill uses existing NVFlare execution surfaces:

- Simulation: run the existing job script with its configured `SimEnv`.
- POC: use startup kits and standard `nvflare job` commands.
- Production: use standard startup-kit authentication, site policy, job submit,
  wait, download, and inspection commands.

Production is a valid optimization environment. The best candidate may later be
submitted or reused through the standard NVFlare job lifecycle; no separate
promotion command is needed.

## Review Questions

- Are the supported `job.py` patterns sufficient for an initial prototype?
- Are the `autofl.yaml` fields enough for reproducibility and candidate
  comparability?
- Is the skill installer target behavior acceptable, or should NVFlare add a
  dedicated generic agent-skill install command in a follow-up?
- Which metric/artifact extraction gaps should become stable NVFlare APIs next?
