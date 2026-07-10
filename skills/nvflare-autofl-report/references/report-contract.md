# Auto-FL Final Report Contract

## Inputs

The report generator consumes product Auto-FL artifacts from the job directory:

- `results.tsv`: append-only result evidence;
- `autofl.yaml`: requested and imported campaign contract;
- `.nvflare/autofl/campaign_state.json`: stop/finalization state;
- `.nvflare/autofl/candidates/*/candidate_manifest.json`: optional detailed
  candidate provenance;
- `progress.png`: existing plot to refresh safely.

The current ledger contract includes status, name, score, runtime, changed
files, hypothesis/diff summary, exact run command, artifacts, failure reason,
candidate manifest, base candidate, and patch hash. Older ledgers may omit
newer fields; the report should disclose missing provenance rather than fail
when the essential status and score columns remain readable.

## Termination

`final_response_allowed=true` is the preferred deterministic proof that a
campaign may be finalized. For abrupt process termination, the user and agent
may explicitly confirm interruption. This assertion is report provenance only:
it must not mutate campaign state. `--confirm-interrupted` bypasses only stale
stop state. Finalization is refused when campaign state reports pending work,
the ledger has a `candidate` row, or an available manifest remains `prepared`
or `ready_for_external_execution`. An available but unreadable manifest also
blocks finalization because the helper cannot prove that its candidate is
complete. Finalize or abandon such candidates before generating an
authoritative report.

## Candidate Lineage

Candidates are based on the current best source snapshot. A retained candidate
can therefore record `changed_files=none` while inheriting algorithm code from
its `base_candidate`. Follow `base_candidate` links back to baseline and union
changed files across the chain. Mark lineage partial when an ancestor row or
manifest is unavailable.

## Literature Evidence

A `literature` row opens a checkpoint. Associate subsequent comparable
candidate rows with that checkpoint until the next literature row. Compare the
best finalized `keep` or `discard` score in that segment with the incumbent
immediately before the review. Pending candidates and crashes are attempts but
never scored improvements, even if a malformed crash row contains a numeric
score:

- `helped`: a following candidate improved the incumbent;
- `matched`: the best following candidate tied the incumbent;
- `not_confirmed`: candidates ran but none matched or improved the incumbent;
- `failed`: attempts produced no score;
- `not_evaluated`: no candidate attempt followed the checkpoint.

Preserve `[src: ...]` markers from the checkpoint. These are campaign-recorded
source identifiers, not independently verified citations.

## Result Selection

`best` is strictly the best scored retained result: a baseline or `keep` row.
`best_observed` may identify a better scored `discard` as unretained evidence.
`candidate` and `crash` rows are excluded from retained best selection,
running-best milestones, and literature improvements. Literature tables render
status first, so scored crashes remain `crash` and unscored discards remain
`n/a`.

The objective contract records two distinct provenance fields. `metric_source`
describes where measurements came from and defaults to `NVFlare metric
artifacts`. `metric_contract_source` records how the importer selected the
metric, for example `user_request`, `arg:key_metric`, or `default`.

## Comparability

`autofl.yaml` describes the imported/declarative contract; exact commands in
`results.tsv` describe what executed. Report both. Compare baseline and best
command options and flag changes to clients, rounds, local epochs/steps, batch
sizes, data partitioning, seed, model architecture, or model-size limits.

When a test-like metric guided multiple candidate decisions, state that the
selected candidate needs one final evaluation on an untouched holdout. Do not
silently present repeated test-set selection as an unbiased final estimate.

## Outputs

- **autofl_final_report.md**: human-readable review artifact;
- `autofl_report_summary.json`: machine-readable summary using schema
  `nvflare.autofl.report.v1`;
- `progress.png`: refreshed using the `nvflare-autofl` product plotter when
  plotting is available.

The JSON summary remains `nvflare.autofl.report.v1` and includes
`artifacts.progress_plot_available` and
`objective.metric_contract_source`. Missing plotting dependencies or an
invalid existing PNG produce warnings and `progress_plot_available=false`, but
do not suppress the Markdown or JSON report. The invalid or failed plot
artifact is preserved and is not embedded in Markdown.

All relative helper path arguments, including `--plotter`, resolve from the
campaign directory rather than the invoking shell's current working directory.

Report generation must be independent of Git and must not edit campaign source,
the ledger, manifests, or state.
