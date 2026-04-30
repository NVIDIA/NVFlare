# Auto-FL NVFlare autoresearch program

This repository is an experiment in applying the **autoresearch** operating model to an NVFlare federated-learning harness.

The human should mostly evolve this `program.md` file and the mutation policy.
The coding agent should mostly iterate on a **small bounded code surface**.

## Setup

To set up a new experiment campaign, work with the user to:

1. Ask the human to prepare the Python environment before the agent starts the campaign:
   - create or activate the desired environment;
   - run `python -m pip install -r requirements.txt`;
   - provide the absolute interpreter path as `PYTHON=<path>`, preferably captured from the activated environment with `python -c 'import sys; print(sys.executable)'`.
2. Treat the human-provided `PYTHON` value as authoritative. Do not search for Python interpreters with filesystem globs such as `ls /usr/bin/python*`, `ls /workspace/.venv*/bin/python*`, or similar discovery commands. If `PYTHON` is missing, empty, or not executable, ask the human for the prepared interpreter path instead of guessing.
3. Do not create virtual environments or install dependencies unless the human explicitly asks you to. Use the exact human-provided `PYTHON` value for validation, smoke tests, candidate runs, plotting, summaries, and reports. Do not silently substitute `python`, `python3`, or another discovered interpreter.
4. Verify the prepared interpreter before running repo commands:
   - `test -x "$PYTHON"`
   - `"$PYTHON" -c "import sys; print(sys.executable)"`
5. Agree on a descriptive run tag derived from the current date at runtime. Use the pattern `<node>-<campaign-topic>-$(date +%Y%m%d)`, for example `h100-fedavgm-$(date +%Y%m%d)`, `h100-archsearch-$(date +%Y%m%d)`, or `h100-baseline-$(date +%Y%m%d)`. Do not use date-only tags such as `h100-$(date +%Y%m%d)`, and do not copy stale example dates from docs.
6. Create a fresh branch: `autoresearch/<tag>`.
7. Treat this file as the agent entry point, then inspect only the supporting files needed for the next action:
   - `mutation_schema.yaml` for hard mutation bounds
   - `client.py`, `custom_aggregators.py`, and `job.py` for the active code surface
   - `README.md` or `ACKNOWLEDGEMENTS.md` only when user-facing setup or provenance context is needed
8. Verify the prepared environment is ready:
   - `PYTHON=<path> make validate`
   - `PYTHON=<path> make smoke`
9. Initialize the run ledger:
   - `bash scripts/init_run.sh <tag>`
10. Confirm the setup and start with the baseline.

## Experimentation

Each experiment should run under a **fixed federated budget**. Keep the following fixed across a comparison campaign unless the human explicitly changes the campaign setup:
- `n_clients`
- `num_rounds`
- `aggregation_epochs`
- `batch_size`
- `eval_batch_size`
- `alpha`
- `seed`
- `model_arch`
- `max_model_params`
- whether cross-site evaluation is enabled
- `final_eval_clients`

Some of these values are technically mutable in `mutation_schema.yaml`, but changing them starts a new comparison budget. Do not compare scores across runs with different values for the fixed budget fields above unless the run is explicitly labeled as a new campaign or subcampaign. Architecture-search scores must be labeled with their `model_arch` and `max_model_params`; do not mix them with optimizer-only `moderate_cnn` results as if they were the same search.

Default H100 candidate budget:
- `--n_clients 8`
- `--num_rounds 10`
- `--aggregation_epochs 4`
- `--batch_size 64`
- `--alpha 0.5`
- `--seed 0`
- `--model_arch moderate_cnn`
- `--max_model_params 5000000`
- `--aggregator weighted`
- cross-site evaluation enabled
- final global evaluation on `site-1`
- `--eval_batch_size 1024`
- `RUN_TIMEOUT_SECONDS=600`
- deterministic PyTorch/DataLoader training enabled

Each candidate targets roughly a 10-minute run on one local H100. The 80 GB H100 can usually support several same-budget candidates concurrently; if runs consistently finish much sooner, increase only one budget axis at a time and label the new subcampaign. If they time out or hit CUDA OOM, reduce candidate parallelism before changing model, parameter-cap, or data contracts.
The current CIFAR-10 validation loader is identical on every simulated client, so final scoring evaluates the server/global model on `site-1` by default and keeps the output in NVFlare's `cross_site_val/cross_val_results.json` path. Use `--final_eval_clients all` only for an audit run or after changing validation to be site-specific.
Training splits are cached by fixed data-budget fields under `/tmp/cifar10_splits/autofl_cifar10_<n>sites_alpha<a>_seed<s>`. Do not make the split path depend on the candidate `--name`; repeated candidates with the same `n_clients`, `alpha`, and `seed` should reuse the same `.npy` indices.
Client training derives stable per-site RNG seeds from `--seed`, enables PyTorch deterministic algorithms, disables cuDNN benchmarking, and seeds DataLoader shuffling/workers. Treat `--no_deterministic_training` as a separate noisy subcampaign.

### Initial algorithm calibration

After the first weighted baseline run in a fresh campaign, run an algorithm calibration sequence before open-ended hyperparameter tuning. Keep the same fixed budget and compare the available baseline implementations:

| Step | Description | Extra args |
| --- | --- | --- |
| 0 | built-in FedAvg audit | `--aggregator default` |
| 1 | explicit FedAvg audit | `--aggregator fedavg` |
| 2 | FedProx light | `--aggregator weighted --fedproxloss_mu 1e-5` |
| 3 | FedProx medium | `--aggregator weighted --fedproxloss_mu 1e-4` |
| 4 | FedAvgM NVFlare-style | `--aggregator fedavgm --server_lr 1.0 --server_momentum 0.6` |
| 5 | FedAvgM accelerated | `--aggregator fedavgm --server_lr 2.0 --server_momentum 0.4` |
| 6 | FedAdam | `--aggregator fedadam --server_lr 1.0 --fedopt_beta1 0.9 --fedopt_beta2 0.99 --fedopt_tau 1e-3` |
| 7 | SCAFFOLD metadata mode | `--aggregator scaffold` |

Run the calibration steps in batches of up to `PARALLEL_CANDIDATES` concurrent runs on the same H100, with unique `RUN_LOG`, `--name`, and result descriptions. Keep the step order when assembling batches and schedule the remainder before moving to unrelated sweeps. Rank the completed calibration runs, choose the most promising algorithm family, and only then start narrower hyperparameter sweeps.

### Architecture calibration

Architecture search is allowed after the baseline algorithm family has been established. Keep the same data, round, optimizer, evaluation, and timeout budget, and keep `--max_model_params 5000000` unless the human explicitly starts a new architecture budget. Compare registered variants first:

| Step | Description | Extra args |
| --- | --- | --- |
| 0 | original architecture audit | `--model_arch moderate_cnn` |
| 1 | normalized architecture | `--model_arch moderate_cnn_norm` |
| 2 | smaller classifier head | `--model_arch moderate_cnn_small_head` |

After the first three architecture audits, use nearby optimizer settings around the current best architecture rather than increasing the parameter cap. Rank primarily by score; use `runtime_seconds` only as a coarse cost signal and tie-breaker for near-equal results. A registered architecture can change the model parameter key/shape schema by design, so compare it only inside a labeled architecture subcampaign.

### Local single-H100 parallel iteration mode

The target research machine is a local node with **one 80 GB H100 GPU**.
Use that one visible GPU for the campaign, but launch several independent NVFlare simulation candidates concurrently when memory allows.
If the environment exposes multiple GPUs but this campaign should use the local H100 only, pin each run with `CUDA_VISIBLE_DEVICES=0`; do not add multi-GPU scheduling or `GPU_IDS` lane mapping.

The initial agent instruction should set a conservative same-H100 candidate width:

```text
PARALLEL_CANDIDATES=4
```

Default to `PARALLEL_CANDIDATES=4` when the initial instruction does not specify it. Lower it to 1 or 2 if candidates hit CUDA OOM, host memory pressure, or heavy I/O contention.

Prefer **ledger-backed decisions** over ad hoc one-off guesses:
1. Keep the code state fixed.
2. Generate a small batch of hyperparameter candidates under the same fixed budget, up to `PARALLEL_CANDIDATES`.
3. Launch each candidate with a unique `--name`, `RUN_LOG`, and description value.
4. Wait for the whole batch to finish or time out.
5. Rank the completed candidates against the ledger by score, inspect crashes with `tail -n 50 <run_log>`, and only then decide the next mutation or sweep.

Example single-H100 calibration skeleton:

```bash
mkdir -p run_logs
PYTHON=${PYTHON:?set PYTHON to the prepared Python interpreter}
PARALLEL_CANDIDATES=${PARALLEL_CANDIDATES:-4}
COMMON_ARGS="--n_clients 8 --num_rounds 10 --aggregation_epochs 4 --batch_size 64 --eval_batch_size 1024 --alpha 0.5 --seed 0 --model_arch moderate_cnn --max_model_params 5000000 --final_eval_clients site-1"

launch_algo_candidate() {
  i="$1"
  case "$i" in
    0) desc="builtin FedAvg audit"; extra_args="--aggregator default"; name="algo_builtin_fedavg" ;;
    1) desc="explicit FedAvg audit"; extra_args="--aggregator fedavg"; name="algo_fedavg" ;;
    2) desc="FedProx mu 1e-5"; extra_args="--aggregator weighted --fedproxloss_mu 1e-5"; name="algo_fedprox_1e5" ;;
    3) desc="FedProx mu 1e-4"; extra_args="--aggregator weighted --fedproxloss_mu 1e-4"; name="algo_fedprox_1e4" ;;
    4) desc="FedAvgM server lr 1.0 momentum 0.6"; extra_args="--aggregator fedavgm --server_lr 1.0 --server_momentum 0.6"; name="algo_fedavgm_lr10_m06" ;;
    5) desc="FedAvgM server lr 2.0 momentum 0.4"; extra_args="--aggregator fedavgm --server_lr 2.0 --server_momentum 0.4"; name="algo_fedavgm_lr20_m04" ;;
    6) desc="FedAdam"; extra_args="--aggregator fedadam --server_lr 1.0 --fedopt_beta1 0.9 --fedopt_beta2 0.99 --fedopt_tau 1e-3"; name="algo_fedadam" ;;
    7) desc="SCAFFOLD metadata mode"; extra_args="--aggregator scaffold"; name="algo_scaffold" ;;
  esac
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  PYTHON="${PYTHON}" RUN_LOG="run_logs/${name}.log" RUN_TIMEOUT_SECONDS=600 \
    bash scripts/run_iteration.sh --description "${desc}" --target client.py -- \
    ${COMMON_ARGS} ${extra_args} --name "${name}" &
}

running=0
for i in 0 1 2 3 4 5 6 7; do
  launch_algo_candidate "$i"
  running=$((running + 1))
  if [ "$running" -ge "$PARALLEL_CANDIDATES" ]; then
    wait
    running=0
  fi
done
wait
```

Never reuse the same `RUN_LOG` for different candidates.
Use unique `--name` values so NVFlare result directories and CIFAR split directories do not collide.
If candidates repeatedly run out of memory on the local H100, reduce `PARALLEL_CANDIDATES` before reducing the fixed budget or starting a clearly labeled smaller-budget subcampaign.
After each batch, rank the ledger with:

```bash
"${PYTHON}" scripts/summarize_results.py results.tsv --status candidate --top "${PARALLEL_CANDIDATES:-4}"
```

### What you CAN do

Prefer this mutation order:
1. `client.py`
2. `custom_aggregators.py`
3. `job.py`
4. `model.py` only for registered architecture variants under the active parameter cap

Typical safe ideas:
- tune optimizer family or hyperparameters
- adjust local epochs, batch size, weight decay, scheduler parameters
- add gradient clipping or label smoothing
- enable a FedProx local term
- compare FedAvg/FedOpt custom aggregators without changing DIFF uploads
- use the explicit `--aggregator scaffold` protocol mode when the campaign is labeled as SCAFFOLD-enabled
- improve weighted or robust aggregation logic without changing parameter keys
- change recipe knobs that keep the protocol intact
- compare registered model architectures with `--model_arch` while keeping `--max_model_params` fixed and enforced

Baseline algorithm knobs:
- FedAvg is available as `--aggregator weighted`, `--aggregator fedavg`, or the built-in NVFlare path `--aggregator default`.
- FedProx is a client-local loss term: keep a FedAvg-style aggregator and set `--fedproxloss_mu`.
- FedOpt is available within the current custom aggregator surface as server optimization over aggregated DIFFs: `--aggregator fedavgm`, `--aggregator fedopt`, or `--aggregator fedadam`, with `--server_lr`, `--server_momentum`, `--fedopt_beta1`, `--fedopt_beta2`, and `--fedopt_tau`.
- SCAFFOLD is available as `--aggregator scaffold`, which opts into SCAFFOLD-specific `FLModel.meta` fields while preserving DIFF uploads, `NUM_STEPS_CURRENT_ROUND`, and the model key/shape schema.

Registered architecture knobs:
- `--model_arch moderate_cnn` keeps the original baseline architecture.
- `--model_arch moderate_cnn_norm` adds buffer-free normalization while staying close to the baseline shape.
- `--model_arch moderate_cnn_small_head` keeps the convolutional trunk and reduces the classifier head.
- `--max_model_params 5000000` is the default hard cap. Keep the cap fixed inside an architecture campaign and reject any candidate that exceeds it before running.

### What you CANNOT do

- modify the evaluation substrate in a way that breaks comparability
- change `ParamsType.DIFF` to full-weight uploads
- remove `NUM_STEPS_CURRENT_ROUND`
- remove the `flare.is_evaluate()` path
- instantiate an unregistered model architecture or exceed `--max_model_params`
- change `model_arch` or `max_model_params` mid-campaign without labeling a new architecture subcampaign
- introduce server-coupled protocol changes other than the explicit `--aggregator scaffold` mode
- add new dependencies

## Goal

The goal is to maximize a **single comparable score** under a fixed federated budget.

Use cross-site evaluation when scoring candidates. The helper script `scripts/extract_score.py` will try to read a comparable metric from:

```text
<result_dir>/server/simulate_job/cross_site_val/cross_val_results.json
```

The optimized score is the final server/global model, `SRV_FL_global_model.pt`; ignore `SRV_best_FL_global_model.pt` when ranking candidates.
Higher is better.

For architecture-search campaigns, `--max_model_params` is a hard gate, not a soft preference. Rank candidates primarily by final score under the same fixed budget. Use `runtime_seconds` as a coarse secondary signal: prefer the faster/simpler candidate when scores are within expected noise, and be skeptical of tiny gains that cost much more wall-clock time. Do not collapse score and runtime into a single `score/runtime` scalar; inspect the Pareto tradeoff and rerun finalists before trusting small differences.

### Simplicity criterion

All else being equal, simpler is better.
A tiny gain that adds brittle complexity is usually not worth keeping.
A similar or better score with simpler code is a win.

### First run

Your very first run in a fresh campaign should always be the baseline with no mutation.
Record it in `results.tsv`.

## Logging results

Keep a tracked `results.tsv` file on experiment branches with this header:

```text
commit	score	runtime_seconds	budget	status	target	description	artifacts
```

Where:
1. `commit` = short git commit hash
2. `score` = extracted comparable metric, or `0.000000` for failures
3. `runtime_seconds` = wall-clock seconds spent in the candidate command, including failures
4. `budget` = short string describing the fixed recipe budget
5. `status` = `keep`, `discard`, `crash`, or `candidate`
6. `target` = main file edited
7. `description` = short mutation description
8. `artifacts` = result dir or log path

Use `runtime_seconds` when comparing research cost. A tiny score gain that consumes much more runtime is weaker evidence than a similar gain at the same cost.

`candidate` is a temporary status for a run that has finished but has not yet gone through review. After every completed run or batch, update the reviewed rows in `results.tsv`: mark the selected survivor as `keep` when it materially improves the score or is comparable with simpler/faster behavior, and mark reviewed non-survivors as `discard`. Do not let stale reviewed rows remain `candidate`; the progress plot uses this status column directly.

Use the helper to avoid hand-editing TSV rows:

```bash
"${PYTHON}" scripts/finalize_batch_status.py results.tsv --last "${PARALLEL_CANDIDATES:-4}" --keep-best --discard-others
```

If an old campaign already has many stale candidate rows, clean it once with:

```bash
"${PYTHON}" scripts/finalize_batch_status.py results.tsv --all-candidates --keep-best --discard-others
```

If a candidate implements a method or design choice from a research paper, include a compact source reference in the `description` field, for example `fedprox mu=0.01 [src: Li20 FedProx arXiv:1812.06127]`. Keep the TSV single-line and tab-free; use semicolon-separated short refs for multiple sources. Put fuller citations, URLs, and the extracted hypothesis in `templates/mutation_report.md`.

Commit `results.tsv` after the baseline and after each completed run or checkpoint so the experiment branch records its run provenance.
Do not commit bulky runtime artifacts such as `run.log`, `run_logs/`, NVFlare result directories, or generated `progress.png`.

## The experiment loop

Default single-candidate loop:

1. Look at the git state: current branch and current commit.
2. Propose one small mutation.
3. Edit the smallest possible set of files.
4. Commit the mutation.
5. Run:

   ```bash
   PYTHON=<path> bash scripts/run_iteration.sh --description "..." --target <file> -- <fixed budget args>
   ```

   This redirects the full job output to `run.log`.
   Candidate runs default to `RUN_TIMEOUT_SECONDS=600`; set `RUN_TIMEOUT_SECONDS=0` only when deliberately disabling the guard.
   Use `--no-log-results` only for smoke checks that should not consume the first baseline row.

6. Read the summarized result from the script output.
7. If the run crashed, inspect `tail -n 50 run.log`, attempt a small fix, and stop after only a few retries.
8. Record the result in `results.tsv`.
9. If the score improved materially, or stayed comparable with simpler code, keep the commit.
10. If the score got worse or the change is not worth the added complexity, revert to the previous good commit.

Same-budget campaign loop for the local H100:

1. Look at the git state: current branch and current commit.
2. Choose one narrow sweep axis, for example LR, momentum, weight decay, scheduler, or FedProx.
3. Propose a batch of up to `PARALLEL_CANDIDATES` candidates that fit the fixed campaign budget.
4. Run the candidates concurrently on the same H100 with unique `RUN_LOG` and `--name` values.
5. Build a short comparison table from `results.tsv`: description, score, status, budget, artifact.
   Prefer `"${PYTHON}" scripts/summarize_results.py results.tsv --status candidate --top "${PARALLEL_CANDIDATES:-4}"`.
6. Inspect only failed-run tails and the best result artifacts; do not flood context with full logs.
7. Decide after the batch:
   - keep the best candidate if it materially improves score;
   - run a narrower follow-up sweep around the best candidate if results are close;
   - discard the whole axis if all candidates regress or add complexity without gain.
8. Update the `status` column for reviewed runs in `results.tsv`. Use `scripts/finalize_batch_status.py --last "${PARALLEL_CANDIDATES:-4}"` or edit the TSV carefully: promoted rows must be `keep`, reviewed non-survivors must be `discard`, crashes remain `crash`, and only unresolved active rows may remain `candidate`.
9. Commit code mutations that survive the run analysis. Also commit `results.tsv` at checkpoint boundaries, even when the run only tested runtime hyperparameters.

### Never stop

Once the experiment loop has begun after setup and the initial baseline, do **not** pause to ask the human whether to continue, whether to keep going, or whether this is a good stopping point.
The human may be asleep or away from the machine and expects the agent to continue autonomously until manually interrupted.

On the local H100, keep cycling through same-budget candidate batches:
- launch up to `PARALLEL_CANDIDATES` concurrent candidates on the same H100,
- wait for the batch to finish or time out,
- rank the results against the ledger,
- keep, narrow, or discard according to score and complexity,
- rewrite reviewed `results.tsv` statuses so completed candidates become `keep` or `discard`,
- choose the next sweep axis,
- repeat.

If you run out of obvious ideas, re-read the in-scope files, inspect recent near-misses in `results.tsv`, combine promising settings, or try a new allowed axis from `mutation_schema.yaml`.
Stay within the hard invariants and current bounds: do not change the FL protocol, evaluation substrate, dependency set, or registered architecture budget unless a human explicitly authorizes a protocol upgrade or new architecture subcampaign.

With the default 10-minute candidate budget and `PARALLEL_CANDIDATES=4`, throughput can reach roughly 24 candidates per hour, or about 192 candidates across an eight-hour overnight run, if the H100, host CPU, storage, and data loaders sustain four concurrent jobs. Reduce the width when contention hurts reliability or score comparability.
The loop runs until the human interrupts it.

### Think harder with literature

When progress stalls, do not merely jitter hyperparameters at random.
Use a Camyla-inspired literature loop: **diverse query generation -> multi-source paper triage -> challenge extraction -> proposal scoring -> quality-weighted branch exploration -> reflective memory**.
Camyla's public repo (`https://github.com/yifangao112/Camyla`) exposes this as a concrete pipeline: search papers, extract open challenges, generate several proposals, score them with an assessment rubric, and then explore stronger branches with QWBE-style competition.
For this harness, adapt the process only at the instruction/artifact level; do not import Camyla code or add new dependencies.

Trigger literature mode when any of these happen:
- two consecutive same-budget candidate batches fail to improve;
- several crashes share the same root cause;
- the next sweep axis is unclear;
- all remaining ideas are minor variations of already-tested settings.

Before searching, gather the working memory:
- current best stack and score from `results.tsv`;
- the last 20 scored rows and any repeated crashes;
- confirmed null/worse axes, including settings the human said not to retry;
- available mutation surface from `mutation_schema.yaml`;
- active budget, current candidate width, and any explicit `CUDA_VISIBLE_DEVICES` pinning.

Use `templates/literature_loop.md` as a scratch worksheet. Keep it compact enough to read during a long run.

Literature mode workflow:
1. Generate 3 distinct search queries from the observed failure mode, not generic method names. Each query should explore a different angle, such as client drift, server optimization, non-IID local overfitting, control variates, robust aggregation, or short-budget convergence.
2. Search at least two available sources when possible: arXiv, Semantic Scholar, OpenAlex, PubMed, official project pages, or paper PDFs. If API access is unavailable, browser/web search is acceptable. Prefer primary papers over blog posts.
3. Triage 6-10 candidate papers down to 3-5 relevant papers. For each kept paper, record title, year, URL/arXiv id, the training challenge it addresses, and the method family. Do not paste long paper excerpts.
4. Extract exactly 3 challenge cards. Each card must include: challenge name, evidence from the paper(s), matching symptom in `results.tsv`, why it matters for this harness, and which allowed file(s) could express it.
5. Generate 3-5 proposal cards, enough to fill the current candidate batch when safe. Each card must include: mechanism, source refs, exact implementation surface (`client.py`, `custom_aggregators.py`, `job.py`, or CLI-only), proposed args/code change, expected effect, runtime cost, contract risk, and what observation would falsify it.
6. Run duplicate/null filtering before scoring. Reject proposals that are the same core idea as existing kept/null rows or that require forbidden protocol changes, new dependencies, unregistered or over-budget model architectures, evaluation changes, or server-coupled tensors beyond explicit SCAFFOLD metadata mode.
7. Score each remaining proposal from 1-5 on: expected score gain, contract safety, implementation simplicity, evidence strength, novelty relative to `results.tsv`, and runtime cost. Use total score `2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost`.
8. Select the next candidate batch QWBE-style from the top proposals. Give the strongest proposal a nearby variant when useful, but keep a distinct second idea in the batch or in reserve if any safe alternative remains.
9. Launch the selected candidate batch under the same compute budget. Rank after the batch finishes or times out.
10. Record reflective memory: what paper-derived hypothesis helped, what failed, and which source-backed idea should not be retried. Put full source details in `templates/mutation_report.md`; include short refs in `results.tsv` descriptions, for example `[src: Li20 FedProx arXiv:1812.06127]`.

Examples of compatible literature-derived directions:
- FedProx coefficient sweeps for client drift.
- FedOpt server momentum or adaptive updates implemented only inside `custom_aggregators.py`.
- SCAFFOLD control-variate sweeps through `--aggregator scaffold`, tracked as an opt-in protocol-mode subcampaign.
- Momentum, weight decay, and LR schedule variants for non-IID convergence.
- Gradient clipping or label smoothing on the client.
- More careful `NUM_STEPS_CURRENT_ROUND` weighting or robust aggregation that preserves parameter keys and `ParamsType.DIFF`.
- Registered architecture variants in `model.py`, selected through `--model_arch`, when they stay under the active `--max_model_params` cap.

Examples that are **not** compatible without human approval:
- New server-coupled state tensors outside the existing SCAFFOLD metadata keys.
- FedOpt variants that require changing the recipe protocol or adding server-client metadata.
- Unregistered architecture changes, architectures above `--max_model_params`, or architecture/data/evaluation changes that are not labeled as a separate budget.
- New packages, new datasets, or altered cross-site evaluation.

## Output discipline

Do not flood the context with full training logs.
Always redirect job output to `run.log` or a unique file under `run_logs/`, and inspect only the summary or the tail on failure.
At checkpoints, generate a compact visual progress summary with:

```bash
"${PYTHON:-python3}" scripts/plot_progress.py results.tsv --output progress.png
```
