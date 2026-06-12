# NVFlare Auto-FL Skill Design

## Summary

Auto-FL should enter NVFlare as a skill-first product experience. Users select
an official NVFlare Auto-FL skill in a coding agent, point it at an existing
`job.py`, and state the optimization objective, environment, and budget. NVFlare
owns deterministic import of campaign-relevant settings, execution truth, policy
boundaries, artifacts, and reproducibility. The agent owns candidate planning,
code edits within allowed paths, experiment execution through the existing
`job.py`, comparison, and narrative reporting.

This avoids introducing a new public Auto-FL command tree while still making
Auto-FL an NVFlare-owned feature.

## Product Boundary

The first production-oriented slice includes:

- A root `skills/nvflare-autofl` agent skill that follows the NVFLARE skills
  layout used by the general agent-skills work.
- A deterministic `job.py` importer that emits reviewable `autofl.yaml` for the
  Auto-FL campaign.
- A trust contract in `autofl.yaml` showing editable campaign settings,
  unresolved fields, fixed-budget constraints, and allowed edit paths.
- Documentation for using the skill with simulation, POC, and production
  environments through existing NVFlare surfaces.

The first version does not embed or vendor a coding agent, and it does not add a
public Auto-FL command family.

## Role of autofl.yaml

`autofl.yaml` is not a replacement for `job.py` and is not a second exported
job format.  The original `job.py` remains the experiment entry point the agent
uses to run candidates, and exported job folders remain the NVFlare execution
and submission artifacts.

The purpose of `autofl.yaml` is to expose the human-reviewable Auto-FL campaign
layer:

- Objective metric, requested environment, and candidate budget.
- Editable search-space settings discovered from `job.py` and related train
  scripts.
- Fixed-budget constraints that must remain comparable across candidates.
- Allowed edit paths and files that are out of scope for the agent.
- Artifact, ledger, and report locations for the campaign.
- Provenance and unresolved fields that need user review before safe execution.

By default, users should not need to edit `autofl.yaml`.  They review or modify
it only when the importer surfaces unresolved settings or when they want to
override campaign knobs explicitly.

## Deterministic Import

The importer parses Python source with `ast`; it does not import or execute
user code.  It supports known Recipe and FedJob-style patterns first and focuses
on campaign-relevant settings rather than duplicating the full exported job:

- Recipe/FedJob constructor and class import.
- `SimEnv`, `PocEnv`, and `ProdEnv` references.
- `train_script` resolution for literal and argparse-derived values.
- Objective metric from user request, `key_metric`, or explicit unresolved
  default.
- Fixed-budget fields such as rounds, clients, and candidate budget.
- Common argparse tunables from `job.py` and the resolved train script.

The exported job folder remains useful as execution truth once the job is
materialized, because it contains resolved NVFlare app and component configs.
However, it does not reliably preserve all authoring intent needed for an
Auto-FL campaign, such as editable source files, train-argument construction,
tunable-versus-fixed intent, and source provenance.  Therefore the importer uses
deterministic Python/static parsing for the campaign layer and may use exported
config inspection as a validation aid when available.

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

The skill must present editable, unresolved, and allowed sections before it runs
candidates. This is the core product guardrail: NVFlare makes the campaign
reviewable and reproducible; the agent makes it interactive and exploratory.

## Execution Model

The skill uses existing NVFlare execution surfaces:

- Simulation: run the existing `job.py` with its configured `SimEnv`.
- POC: use the existing job authoring/export flow, startup kits, and standard
  `nvflare job` commands.
- Production: use standard startup-kit authentication, site policy, job submit,
  wait, download, and inspection commands.

Production is a valid optimization environment. The best candidate may later be
submitted or reused through the standard NVFlare job lifecycle; no separate
promotion command is needed.

## Review Questions

- Are the supported `job.py` patterns sufficient for an initial prototype?
- Are the editable `autofl.yaml` fields enough for human-in-the-loop campaign
  review and candidate comparability?
- Which exported-job fields should be used as validation evidence versus static
  `job.py` parsing for authoring intent?
- Does the Auto-FL skill pass the general NVFLARE skill frontmatter, trigger,
  and eval checks after it lands under `skills/nvflare-autofl`?
- Which metric/artifact extraction gaps should become stable NVFlare APIs next?
