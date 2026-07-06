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
- A deterministic, skill-private `job.py` importer that emits reviewable
  `autofl.yaml` for the Auto-FL campaign without executing user code.
- A trust contract in `autofl.yaml` showing editable campaign settings,
  unresolved fields, fixed-budget constraints, and allowed edit paths.
- A skill-local candidate lifecycle that snapshots the current best source,
  gives the agent an isolated draft, validates the resulting patch, and keeps or
  restores source according to the campaign metric.
- Documentation for using the skill with simulation, POC, and production
  environments through existing NVFlare surfaces.

The first version does not embed or vendor a coding agent, and it does not add a
public Auto-FL command family. Users install and select the skill, then express
their intent in a prompt; they do not invoke its helper scripts directly.

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
- Existing preferred source targets declared by task-local
  `mutation_schema.yaml`, once they resolve inside the job workspace.
- Allowed creation patterns for new Python modules under the job root.
- Artifact, ledger, and report locations for the campaign.
- Provenance and unresolved fields that need user review before safe execution.

By default, users should not need to edit `autofl.yaml`.  They review or modify
it only when the importer surfaces unresolved settings or when they want to
override campaign knobs explicitly.

## Deterministic Import

The importer parses Python source with `ast`; it does not import or execute
user code. It resolves direct imports, import aliases, and module aliases for
known Recipe surfaces and NVFlare-distributed classes whose names end in
`Job`. The generic API `Job`, local subclasses, and non-NVFlare subclasses stay
explicitly unresolved. The importer focuses
on campaign-relevant settings rather than duplicating the full exported job:

- Recipe/FedJob constructor and class import.
- `SimEnv`, `PocEnv`, and `ProdEnv` references.
- `train_script` resolution for literal and argparse-derived values, or for one
  unambiguous NVFlare `ScriptRunner(script=...)` call.
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
The runner writes this reviewable `autofl.yaml` before admission and refuses to
start a baseline when the job surface or fixed comparison budget remains
safety-critical and unresolved.

## Trust Contract

Every import result includes:

- `import`: importer version, source path, source hash, support status, and
  confidence.
- `job`: surface, entrypoint, train script, and call arguments with provenance.
- `objective`, `budget`, `environment`, and `search_space`.
- `trust_contract`: extracted facts, unresolved fields, allowed edit paths, and
  allowed creation patterns and agent controls. This is the sole source of
  candidate source permissions.

The skill must present editable, unresolved, and allowed sections before it runs
candidates. This is the core product guardrail: NVFlare makes the campaign
reviewable and reproducible; the agent makes it interactive and exploratory.
During campaign initialization, the runner merges existing, workspace-local
`mutation_schema.yaml` `preferred_targets` into
`trust_contract.allowed_edit_paths`. Missing, symlinked, reserved, or
out-of-workspace targets remain unresolved rather than being silently
authorized.

## Candidate Contract

The agent, rather than the deterministic runner, owns search policy. It may
change tunables, edit the imported job's allowed source files, or implement new
client and server algorithms as Python modules. This includes creating or
editing server aggregator modules and registering them through `job.py`; the
agent is not limited to pre-enumerated FedAvg, FedAvgM, FedAdam, FedOpt, or
SCAFFOLD choices. Each attempt starts from the retained best source
in `.nvflare/autofl/candidates/<id>/source` and has a generated
`candidate_manifest.json` containing its hypothesis, base candidate, run
arguments, changed files, source and budget hashes, patch hash, artifacts, and
result.

NVFlare computes the manifest's evidence fields; the agent does not assert them.
Before execution, the helper rejects stale candidates, path traversal, symlink
escapes, unauthorized existing-file edits, and detectable fixed-budget drift.
It applies the candidate transactionally to the real job workspace, retains a
new best, and restores the previous best after a discard or crash. This works
without requiring a Git repository and leaves the best source ready for the
standard NVFlare job lifecycle.

The built-in parameter candidates are suggestion seeds only. They are returned
as machine-readable hypotheses and arguments when requested, but are not the
default search loop and are never executed without agent selection.
After each literature-triggered plateau, campaign state requests at least one
source-backed server aggregation candidate under the same comparison budget.
When the job contract makes that impossible, the agent records the reason in
the literature event instead of silently omitting aggregation exploration.

## Execution Model

The skill uses existing NVFlare execution surfaces:

- Simulation: initialize a baseline, prepare an agent-authored candidate draft,
  and evaluate it through the existing `job.py` and configured `SimEnv`.
- POC: use the existing job authoring/export flow, startup kits, and standard
  `nvflare job` commands, then record the job ID, artifacts, and metric against
  the candidate manifest.
- Production: use standard startup-kit authentication, site policy, job submit,
  wait, download, and inspection commands with the same manifest and result
  recording contract.

Production is a valid optimization environment. The best candidate may later be
submitted or reused through the standard NVFlare job lifecycle; no separate
promotion command is needed.

The runner is the sole writer of `.nvflare/autofl/campaign_state.json`. Its
`status` action rescans the ledger, pending manifests, stop files, and cap and
only rewrites state when its semantic contents change. It does not regenerate
the ledger, plot, or report during an unchanged status check. The standalone
campaign guard is a read-only diagnostic and cannot overwrite runner metadata.
A stop file takes immediate precedence: pending candidates must be safely
abandoned before final reporting. Without a stop request, pending prepared or
externally ready candidates take precedence over cap exhaustion and reporting.

Every score records its metric name, extraction source, and artifact. Structured
metric artifacts take precedence over exact, provenance-labelled text fallback.
The ledger is replaced atomically so a failed write cannot leave a partial
campaign record.

## Skill Implementation Boundary

The deterministic importer and campaign runner live under
`skills/nvflare-autofl/scripts/`. They are private executable resources of the
Agent Skill, resolved relative to `SKILL.md` by the activated coding agent. The
human-facing workflow exposes only skill installation, selection, and an intent
prompt; it does not document direct Python helper invocation.

This placement deliberately keeps the unreleased `autofl.yaml` contract out of
`nvflare.app_common` and the public `nvflare agent` CLI. A general NVFlare job
contract command should be considered only after another concrete workflow
needs the same interface and the schema has proved stable. The general,
read-only `nvflare agent inspect` surface does not acquire an Auto-FL-specific
profile in this proposal.

## Review Questions

- Are the supported `job.py` patterns sufficient for an initial prototype?
- Are the edit and creation permissions in `autofl.yaml` appropriate for
  algorithm-level candidates while preserving candidate comparability?
- Which exported-job fields should be used as validation evidence versus static
  `job.py` parsing for authoring intent?
- Does the Auto-FL skill pass the general NVFLARE skill frontmatter, trigger,
  and eval checks after it lands under `skills/nvflare-autofl`?
- Which candidate-manifest and metric/artifact fields should become stable
  NVFlare APIs after the skill-local contract proves itself?
