# NVFlare Auto-FL Skill Design

## Summary

Auto-FL is implemented in NVFlare as a skill-first product experience. Users select
an official NVFlare Auto-FL skill in a coding agent, point it at an existing
`job.py`, and state the optimization objective, environment, and budget. NVFlare
owns deterministic import of campaign-relevant settings, execution truth, policy
boundaries, artifacts, and reproducibility. The agent owns candidate planning,
code edits within allowed paths, experiment execution through the existing
`job.py`, and hypothesis-driven exploration. The companion report skill turns
the recorded evidence into deterministic final artifacts for the agent to
summarize.

This design avoids introducing a new public Auto-FL command tree while making
Auto-FL an NVFlare-owned feature.

## Product Boundary

The product boundary comprises:

- A root `skills/nvflare-autofl` agent skill that follows the NVFLARE skills
  layout used by the general agent-skills work.
- A deterministic, skill-private `job.py` importer that emits reviewable
  `autofl.yaml` for the Auto-FL campaign without executing user code.
- A trust contract in `autofl.yaml` showing editable campaign settings,
  unresolved fields, fixed-budget constraints, and allowed edit paths.
- A skill-local candidate lifecycle that snapshots the current best source,
  gives the agent an isolated draft, validates the resulting patch, and keeps or
  restores source according to the campaign metric.
- This follow-up's companion `skills/nvflare-autofl-report` skill, which
  deterministically turns a stopped campaign ledger, state, config, and
  manifests into human- and machine-readable final report artifacts.
- Documentation for using the skill with simulation, POC, and production
  environments through existing NVFlare surfaces.

The first version does not embed or vendor a coding agent, and it does not add a
public Auto-FL command family. Users install and select the skill, then express
their intent in a prompt; they do not invoke its helper scripts directly.
Installation uses the standard Agent Skills workflow, for example
`npx skills add ./skills -a codex -a claude-code`; NVFlare does not add a custom
skill installer or package skills in the Python wheel.

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
Each recorded literature review receives a persistent `literature_event_id`
(`lit-0001` style). After a review is recorded, campaign state requires an
exploration batch: `exploration_batch_size` scored source-backed candidates
(default 3, flag `--exploration-batch-size`, env
`AUTOFL_EXPLORATION_BATCH_SIZE`) linked to that review under the same
comparison budget before normal candidate flow resumes. `prepare` accepts
`--family` for the algorithm-family slug and `--literature-event` for the
review link; both are persisted in `candidate_manifest.json` and as the
`candidate_kind`, `algorithm_family`, and `literature_event_id` columns in
`results.tsv`. A literature-linked candidate must contain source edits; the
runner rejects argument-only literature-linked candidates at evaluate time.
Qualifying exploration is not limited to server aggregation: client optimizer,
loss function, learning-rate schedule, and architecture candidates within the
fixed comparison budget count equally. When the job contract makes
source-backed exploration impossible, the agent records the reason in the
literature event instead of silently omitting it.

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

Campaign state enforces the exploration batch deterministically. After a
literature review is recorded, state reports
`next_action=develop_literature_batch` and
`required_exploration=source_backed_exploration` until the review's batch of
linked scored candidates completes. The plateau counter resets only when that
batch completes — when its final linked scored candidate records — not when
the review row is recorded. After the first literature event, if the last
`family_repeat_limit` scored attempts (default 6, flag `--family-repeat-limit`,
env `AUTOFL_FAMILY_REPEAT_LIMIT`, 0 disables) are all argument-only tuning of
the same algorithm family, state reports `next_action=diversify_candidates`.
The `progress.png` plot renders source-edited candidates distinctly from
argument-only candidates and annotates literature-linked families.

Every score must be finite and records its metric name, extraction source, and
artifact. Structured metric artifacts take precedence over text fallback, which
accepts only explicit metric-field or final-evaluation lines and records the
selected line number. The ledger is replaced atomically so a failed write cannot
leave a partial campaign record.

Each simulation trial runs in an isolated temporary simulator workspace that
the runner injects through the process-level
`NVFLARE_SIMULATOR_WORKSPACE_ROOT` override, so concurrent campaigns cannot
share or delete each other's artifacts. Result names stay campaign-scoped and
the runner holds one nonblocking lifecycle lock under each job's
`.nvflare/autofl/` directory. A concurrent action for the same job exits with
code 2; separate job workspaces remain independent. A recipe without a literal
job name may run, but its result root must resolve afterwards to the printed
result path or the sole changed workspace child; ambiguous or out-of-workspace
results fail closed. POC and production results are accepted only after the
materialized candidate is re-imported and its fixed comparison budget is
verified again.

Local simulation needs to bind sockets that a restricted coding-agent sandbox
may prohibit. Authorization is therefore established once by the human during
campaign setup, using the host agent's command permission mechanism. The grant
is limited to the resolved Python interpreter and Auto-FL runner, the
`initialize` and `evaluate` actions, and the selected absolute `job.py`; all
other actions and every POC/production operation retain normal permissions.
The skill never writes permission configuration or treats subprocess output as
authorization. A detected socket denial remains an `infrastructure_retry` with
exit code 75, but state reports `await_simulation_runner_approval` and waits for
the existing grant or the human instead of initiating escalation.

This command scope does not sandbox training code inside the simulator. User
and agent-authored candidate Python executes with the runner's host privileges,
like any locally launched training job. Autonomous simulation campaigns should
therefore run in a disposable container or dedicated VM. Production execution
continues through startup-kit authentication, standard job submission, and site
policy; the local simulation grant cannot bypass those controls.

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
profile in this implementation.

## Stopped-Campaign Reporting

Reporting is a separate skill boundary because its trigger and safety posture
differ from active optimization. `nvflare-autofl` must continue an active,
uncapped campaign while state has `final_response_allowed=false`.
`nvflare-autofl-report` operates only after a clean stop, explicit cap, hard
blocker, or independently confirmed interruption.

The report helper consumes `results.tsv`, `autofl.yaml`, campaign state, and
candidate manifests. It attempts to refresh the shared `progress.png` and
writes:

- `autofl_final_report.md`, a concise review artifact with executive summary,
  selected-candidate rationale, what-helped/what-did-not-help synthesis, major
  trajectory, best-candidate lineage, exact commands, reliability, and
  reproduction guidance;
- `autofl_report_summary.json`, a machine-readable
  `nvflare.autofl.report.v1` summary for tools and future automation.

The helper does not edit source, ledger, manifests, or campaign state and does
not require Git. If an abrupt interruption leaves state active, the human must
confirm interruption after execution is independently checked; the report
records that assertion without rewriting history. This confirmation bypasses
only stale stop state. Pending state, `candidate` ledger rows, or manifests in
`prepared`/`ready_for_external_execution` status block finalization until the
active skill finalizes or abandons them. Unreadable manifests block as well,
because report generation cannot prove that their candidates were finalized.

Report finalization acquires the same nonblocking campaign lifecycle lock as
the active runner and holds it across evidence reads, plotting, and report
writes. A concurrent lifecycle action therefore causes a clean refusal rather
than a stale or mixed report. Before any artifact write, the helper also rejects
canonical or filesystem aliases between writable outputs and campaign evidence,
`job.py`, or trust-contract source paths; outputs may not match the trust
contract's allowed source-creation patterns. It also requires the plot,
Markdown, and JSON destinations to be distinct. A persisted POSIX lock file is
a stable lock target, not proof of a live owner. A read-only archive remains
reportable only when that lock file already exists and outputs point to writable
locations. Case-folded collision checks keep output configurations safe on
case-insensitive filesystems.

Relative report-helper paths, including an overridden plotter, resolve from
the campaign directory so agent execution is independent of shell location.

Plotting is optional report evidence. A missing plotting dependency or invalid
PNG does not suppress the Markdown and JSON artifacts: the helper preserves the
failed artifact, emits a warning, omits the Markdown image, and records
`artifacts.progress_plot_available=false`.

Literature reporting follows measured evidence rather than agent narrative.
Each checkpoint's `literature_event_id` links it to candidates developed from
that review, including candidates recorded after a newer checkpoint. Candidates
without an event ID are not attributed by ledger position. Their best result is
compared with the retained incumbent immediately before the review and
classified as helped, matched, not confirmed, failed, or not evaluated.
Recorded `[src: ...]` markers are preserved as campaign provenance, not
presented as independently verified citations.

The report's concise outcome synthesis builds on those same semantics. A
candidate appears under "What Helped" only when a scored `keep` row strictly
improved the retained incumbent. Representative scored discards are grouped by
their explicit algorithm family and literature event, while crashes and
unscored discards are grouped by recorded failure reason. Family outcomes can
be helped, mixed, not confirmed, or failed; missing family metadata remains
unclassified rather than being inferred from candidate names. The selected
candidate is explained as the best scored retained result with its baseline
delta, hypothesis, kind, family, and lineage.

For long campaigns, the trajectory retains the first and final running best
and selects the largest measured objective improvements for the remaining
milestone slots. This preserves the important optimization story without
weakening the underlying retained-best and literature-event contracts.

The report distinguishes retained and observed evidence. `best` is limited to
scored baseline and `keep` rows, while `best_observed` may expose an unretained
scored `discard`. Pending candidates and crashes remain attempt/failure
evidence and cannot become milestones or literature improvements. The
objective also separates measurement provenance (`metric_source`) from the
importer's metric-contract provenance (`metric_contract_source`). Per-run
metric name, extraction source and artifact, candidate kind, algorithm family,
and literature event linkage flow from the final `results.tsv` contract into
the JSON summary and best-candidate report.

The merged Auto-FL producer supports maximization only, so the report rejects
obsolete minimization contracts. It derives candidate attempts, baseline, and
improvement from the ledger, cross-checks those values against authoritative
campaign state, verifies the state's authoritative ledger pointer, and
preserves the state-derived abandoned-candidate count.

Finally, the report compares the declarative/imported budget with exact
baseline and best-candidate commands. It highlights changed compute or data
arguments, aggregation, cross-site evaluation, or final-evaluation populations,
incomplete lineage, and repeated selection on test-like metrics.
This makes the report a trust artifact rather than a polished restatement of
the agent's conclusions.

## Follow-Up Review Questions

- Is `nvflare.autofl.report.v1` sufficient for downstream review and automation
  while remaining explicitly skill-local in this follow-up?
- Which report, candidate-manifest, and metric/artifact fields should become
  stable NVFlare APIs after these skill-local contracts prove themselves?
- Which additional POC and production campaign fixtures should be added once
  externally recorded campaigns are available?
