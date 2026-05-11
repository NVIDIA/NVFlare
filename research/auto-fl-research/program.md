# Auto-FL NVFlare General Program

This repository applies an autoresearch operating model to NVFlare
federated-learning harnesses.

`program.md` is the general control plane. It defines the workflow that should
fit most Auto-FL applications: setup discipline, fixed-budget comparisons, FL
contract invariants, ledger handling, literature recovery, and output hygiene.

Task-specific files define the concrete workload, environment, metric, budget,
and mutation surface. For the default CIFAR-10/H100 profile, read `cifar10.md`
immediately after this file.

## Setup

To start a campaign:

1. Identify the active task profile. If the human does not specify one, use
   `cifar10.md`.
2. Read this file first, then read the task profile. The task profile owns the
   dataset, model family, runtime environment, metric, budget, and preferred edit
   surface for that campaign.
   Examples of human profile-selection prompts:
   - "Use `vlm_med.md` as the active task profile for this campaign."
   - "Use the medical VLM profile instead of the default `cifar10.md`."
   - "Create and use `my_task.md` as the task profile; follow `program.md` for
     the general loop."
   - "Default to `cifar10.md` for this run."
3. Use the interpreter and dependency instructions from the task profile. Do not
   create virtual environments, install dependencies, or search for alternate
   interpreters unless the human explicitly asks.
4. Verify the prepared interpreter before running validation, smoke tests,
   baseline, or candidates. Use the exact checks specified by the task profile.
5. Agree on a descriptive run tag derived from the current date at runtime. Use
   the pattern `<node>-<task-or-topic>-$(date +%Y%m%d)`. Do not use date-only
   tags and do not copy stale example dates from docs.
6. Before validation, smoke tests, baseline, or candidate runs, initialize the
   campaign with `bash scripts/init_run.sh <tag>`. This must create or switch to
   `autoresearch/<tag>` and initialize `results.tsv`.
7. Verify `git branch --show-current` starts with `autoresearch/`. If it does
   not, stop before running experiments; do not run campaigns on `main`,
   `upstream/main`, starter branches, or shared feature branches.
8. Inspect only the supporting files needed for the next action. Use the task
   profile and `mutation_schema.yaml` for hard bounds, and read code files only
   when they are relevant to the chosen mutation or failure.
9. Run the validation and smoke checks named by the task profile.
10. Establish the baseline before mutating code or sweeping hyperparameters.

## Fixed-Budget Comparisons

Every experiment should run under a fixed communication, data, model,
evaluation, and runtime budget. The task profile must list the fields that are
fixed within a comparable campaign.

Common fixed fields include:

- client/site count
- communication rounds
- training and evaluation batch sizes
- data partition or site mapping
- random seed policy
- model architecture and parameter cap
- cross-site evaluation settings
- final evaluation clients
- primary metric and score extraction path

Some fixed fields may be technically mutable in `mutation_schema.yaml`, but
changing one starts a new labeled budget or subcampaign. Do not compare scores
across different budgets as if they were the same experiment.

Local compute may be mutable when the task profile allows it and each candidate
stays within `RUN_TIMEOUT_SECONDS`. Vary only one local-compute mode in a narrow
sweep. Rank primarily by score, then use `runtime_seconds` as a cost and
tie-breaker.

## FL Contract

Preserve the NVFlare Client API contract unless the human explicitly approves a
protocol upgrade:

- `flare.init()`
- `while flare.is_running():`
- `input_model = flare.receive()`
- `model.load_state_dict(input_model.params, strict=True)`
- local train/evaluate behavior appropriate to the task
- `compute_model_diff(...)`
- `flare.send(output_model)`
- `output_model.params_type == ParamsType.DIFF`
- `output_model.meta["NUM_STEPS_CURRENT_ROUND"]`
- the optional `if flare.is_evaluate():` branch
- the same selected model state schema on server and clients for a run

Do not change evaluation data, score semantics, parameter key schema, dependency
set, or server-client protocol tensors inside a comparable campaign unless the
task profile labels that as a new campaign and the human approves the risk.

## Goal

The goal is to maximize one comparable score under the active task profile's
fixed budget and runtime cap.

The task profile must define:

- the primary metric
- the result artifact path used by `scripts/extract_score.py`
- whether higher or lower is better
- baseline args
- any task-specific audit or guard metrics

Use cross-site evaluation when the profile requires it. Rank the final
server/global model specified by the profile; do not switch to a best-checkpoint
or alternate metric mid-campaign.

All else being equal, simpler is better. A tiny gain that adds brittle code or
substantially higher runtime is usually not worth keeping. A similar or better
score with simpler code is a win.

## Mutation Surface

Prefer the task profile's edit order. Keep changes scoped to the smallest files
that can express the candidate.

Generally safe mutation classes:

- client-local optimizer, regularization, clipping, or scheduling knobs
- local-compute knobs within the fixed budget
- aggregation logic that preserves DIFF uploads and parameter keys
- recipe arguments that keep the protocol intact
- registered model variants inside an explicitly labeled architecture budget

Generally unsafe without human approval:

- full-weight uploads instead of DIFFs
- removing `NUM_STEPS_CURRENT_ROUND`
- removing evaluation branches used by the harness
- changing prompt, metric, validation, or dataset semantics mid-campaign
- unregistered or over-budget model architectures
- new dependencies
- server-coupled metadata outside an explicitly supported protocol mode

Use `mutation_schema.yaml` only when this file or the task profile directs you
to hard bounds, or when choosing a mutation axis.

## Logging Results

Keep a tracked `results.tsv` file on experiment branches with this header:

```text
commit	score	runtime_seconds	budget	status	target	description	artifacts
```

`scripts/init_run.sh <tag>` creates this header before the baseline. As a
guardrail, `scripts/run_iteration.sh` also creates or migrates the header before
launching a logged run, then appends the row only after the candidate exits. If
`results.tsv` is missing after an experiment command has started, treat that as
evidence the agent skipped `run_iteration.sh`, used `--no-log-results`, or is
not on the required `autoresearch/` branch.

Columns:

1. `commit` = short git commit hash
2. `score` = extracted comparable metric, or `0.000000` for failures
3. `runtime_seconds` = wall-clock seconds spent in the candidate command,
   including failures
4. `budget` = short string describing the fixed recipe budget
5. `status` = `keep`, `discard`, `crash`, `candidate`, or `literature`
6. `target` = main file edited
7. `description` = short mutation description
8. `artifacts` = result dir or log path

`candidate` is temporary. After every completed run or batch, update reviewed
rows in `results.tsv`: mark the selected survivor as `keep` when it materially
improves score or is comparable with simpler/faster behavior, and mark reviewed
non-survivors as `discard`. Do not let stale reviewed rows remain `candidate`;
the progress plot uses this status column directly.

Use helpers instead of hand-editing TSV rows:

```bash
"${PYTHON}" scripts/summarize_results.py results.tsv --status candidate --top "${PARALLEL_CANDIDATES:-1}"
"${PYTHON}" scripts/finalize_batch_status.py results.tsv --last "${PARALLEL_CANDIDATES:-1}" --keep-best --discard-others
```

If an old campaign already has many stale candidate rows, clean it once with:

```bash
"${PYTHON}" scripts/finalize_batch_status.py results.tsv --all-candidates --keep-best --discard-others
```

`literature` rows are non-scored event markers for the stall-recovery loop. They
use a blank `score`, `budget=literature_loop`,
`target=templates/literature_loop.md`, a short description of the
plateau/search/proposals, and `artifacts=templates/literature_loop.md`.

If a candidate implements a method or design choice from a research paper,
include a compact source reference in `description`, for example
`fedprox mu=0.01 [src: Li20 FedProx arXiv:1812.06127]`. Keep the TSV
single-line and tab-free; put fuller citations, URLs, and hypotheses in
`templates/mutation_report.md`.

Commit `results.tsv` after the baseline and after each completed run or
checkpoint. Commit surviving code changes on the active `autoresearch/` branch
as soon as they are kept; never launch the next batch with kept code changes
only in the working tree. Do not commit bulky runtime artifacts such as logs,
NVFlare result directories, generated progress plots, caches, or checkpoints.

## Experiment Loop

Default single-candidate loop:

1. Inspect the current branch and commit. Confirm the branch starts with
   `autoresearch/`.
2. Choose one small mutation or one CLI-only candidate within the task profile.
3. Edit the smallest possible set of files, if code is needed.
4. Commit the mutation before launching the candidate when the run depends on
   code changes.
5. Run the candidate through `scripts/run_iteration.sh` with the task profile's
   baseline args plus the one-axis change.
6. Read the summarized result from the script output.
7. If the run crashed, inspect only the failure tail, fix the smallest root
   cause, and stop after a few retries on the same failure family.
8. Rank the result against `results.tsv`.
9. Keep the change if it materially improves score or is comparable with
   simpler/faster behavior; otherwise discard it.
10. Finalize `results.tsv` statuses and commit the ledger at checkpoints.

Same-budget batch loop:

1. Confirm the branch starts with `autoresearch/`.
2. Choose one narrow sweep axis.
3. Propose up to `PARALLEL_CANDIDATES` candidates under the same fixed budget
   and runtime cap.
4. Run candidates with unique `RUN_LOG`, `--name`, and descriptions.
5. Wait for the whole batch to finish or time out.
6. Build a short comparison table from `results.tsv`.
7. Inspect only failed-run tails and the best result artifacts.
8. Keep, narrow, or discard the axis based on score, complexity, and runtime.
9. Finalize reviewed statuses in `results.tsv`.
10. Run the plateau watchdog before selecting the next sweep:

    ```bash
    "${PYTHON}" scripts/plateau_watchdog.py results.tsv
    ```

If it prints `recommendation=literature`, run the literature loop before
launching more candidates. If it prints `recommendation=continue`, keep
iterating locally unless repeated crashes share one root cause or no
non-duplicate safe axis remains.

## Never Stop

Once setup and the initial baseline are complete, continue autonomously until
manually interrupted. Do not pause to ask whether to keep going.

Cycle through:

- choose a clear allowed axis
- run one same-budget candidate, or a small same-budget batch if the profile
  allows it
- rank against the ledger
- keep, narrow, or discard
- finalize `results.tsv` statuses
- run the plateau watchdog
- switch to literature mode when triggered
- commit surviving code and ledger checkpoints
- repeat

If local ideas run out, inspect recent near-misses in `results.tsv`, reread the
task profile and `mutation_schema.yaml`, combine compatible kept settings, or
switch to the literature loop. Stay within the hard invariants and active
budget.

## Literature Loop

When progress stalls, do not randomly jitter hyperparameters. Use a
Camyla-inspired literature loop: diverse query generation, multi-source paper
triage, challenge extraction, proposal scoring, quality-weighted branch
exploration, and reflective memory.

Camyla's public repo (`https://github.com/yifangao112/Camyla`) exposes this as
a concrete pipeline: search papers, extract open challenges, generate several
proposals, score them with an assessment rubric, and then explore stronger
branches with QWBE-style competition. For this harness, adapt the process only
at the instruction/artifact level; do not import Camyla code or add new
dependencies.

The watchdog is the normal trigger for literature mode. Soft stall symptoms
should shape the next worksheet after the watchdog fires; they are not
permission to log a new literature row every few candidates.

Trigger literature mode when:

- `scripts/plateau_watchdog.py results.tsv` prints `recommendation=literature`;
- several crashes share the same root cause and a source-backed fix is needed;
- after checking recent near-misses, the task profile, `mutation_schema.yaml`,
  and known null results, no non-duplicate safe axis remains.

Do not start a new literature loop while the watchdog prints
`recommendation=continue` merely because one batch underperformed or the next
local sweep needs routine choice among allowed knobs.

Before searching, gather working memory:

- current best stack and score from `results.tsv`
- the last 20 scored rows and any repeated crashes
- confirmed null or worse axes
- active task profile and fixed budget
- available mutation surface from `mutation_schema.yaml`
- current candidate width and any accelerator pinning

Use `templates/literature_loop.md` as a compact worksheet.

Start a literature-review timer:

```bash
"${PYTHON}" scripts/log_literature_review.py --start --description "plateau after <row or batch>: <symptom>"
```

After proposal scoring and before launching the selected candidate batch, append
the review event:

```bash
"${PYTHON}" scripts/log_literature_review.py --finish --description "literature review: <plateau symptom>; selected <proposal ids or mechanisms>"
```

If you forgot to start the timer, do not interrupt the campaign; use `--log`
with a short description and, if known, `--runtime-seconds`.

Literature mode workflow:

1. Generate three distinct search queries from the observed failure mode, not
   generic method names.
2. Search at least two available paper sources when possible. Prefer primary
   papers over blog posts.
3. Triage 6-10 candidate papers down to 3-5 relevant papers. Record title,
   year, URL/arXiv id, challenge, and method family.
4. Extract exactly three challenge cards: challenge, evidence, matching
   `results.tsv` symptom, why it matters here, and allowed implementation file.
5. Generate 3-5 proposal cards with mechanism, source refs, exact implementation
   surface, proposed args or code change, expected effect, runtime cost,
   contract risk, and falsifying observation.
6. Reject duplicates, known null ideas, protocol changes, new dependencies,
   over-budget model ideas, and evaluation changes.
7. Score each proposal from 1-5 on expected gain, contract safety, simplicity,
   evidence, novelty, and runtime cost. Use:

   ```text
   2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost
   ```

8. Select the next candidate or batch QWBE-style from the top proposals. Give
   the strongest proposal a nearby variant when useful, but keep a distinct
   second idea in the batch or in reserve if any safe alternative remains.
9. Record the `literature` event in `results.tsv` before launching candidates.
10. Launch candidates under the same budget and rank after completion.
11. Record reflective memory in `templates/mutation_report.md`.

Common compatible literature-derived directions include:

- client-local regularization for drift or overfitting
- server momentum or adaptive aggregation implemented without changing DIFF keys
- an explicitly supported protocol mode tracked as a labeled subcampaign
- optimizer, weight decay, gradient clipping, label smoothing, or scheduler
  variants
- more careful `NUM_STEPS_CURRENT_ROUND` weighting or robust aggregation that
  preserves parameter keys and `ParamsType.DIFF`
- registered architecture or adapter variants when the task profile labels the
  budget and enforces the active parameter cap

Common incompatible directions without human approval include:

- new server-coupled state tensors outside an explicitly supported protocol mode
- aggregation variants that require changing the recipe protocol
- unregistered architectures, over-budget architectures, or model/data/eval
  changes that are not labeled as a separate budget
- new packages, new datasets, or altered cross-site evaluation

## Output Discipline

Redirect full training output to `run.log` or unique files under `run_logs/`.
Do not flood context with full logs. Inspect summaries and short failure tails.

At checkpoints, generate a compact progress plot with:

```bash
"${PYTHON:-python3}" scripts/plot_progress.py results.tsv --output progress.png
```
