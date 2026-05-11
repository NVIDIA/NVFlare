# CIFAR-10 Auto-FL Task Profile

Read `program.md` first. This file is the task-specific profile for the default
CIFAR-10/H100 Auto-FL campaign.

This profile owns the concrete CIFAR-10 workload, Python/H100 environment,
metric, baseline budget, model variants, calibration schedule, and mutation
surface. The general experiment loop, ledger rules, literature workflow, and
output discipline remain in `program.md`.

## Setup

To set up a CIFAR-10 campaign, work with the user to:

1. Assume the README preflight created a Python 3.12 local environment in this directory and set `PYTHON=.venv/bin/python` by default. The default dependency set is listed in `tasks/cifar10/requirements.txt`. If the human explicitly provides a different `PYTHON` value, treat that override as authoritative, but still require Python 3.12 for this harness.
2. Do not create virtual environments or install dependencies unless the human explicitly asks you to. Do not search for Python interpreters with filesystem globs such as `ls /usr/bin/python*`, `ls /workspace/.venv*/bin/python*`, `which python`, or similar discovery commands.
3. If `PYTHON` is missing, empty, not executable, or not Python 3.12, tell the human to rerun the README preflight in this directory with `python3.12` or provide an explicit Python 3.12 override. Do not guess.
4. Verify the prepared interpreter before running repo commands:
   - `test -x "$PYTHON"`
   - `"$PYTHON" -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version; print(sys.executable)"`
5. Agree on a descriptive run tag derived from the current date at runtime. Use the pattern `<node>-<campaign-topic>-$(date +%Y%m%d)`, for example `h100-fedavgm-$(date +%Y%m%d)`, `h100-archsearch-$(date +%Y%m%d)`, or `h100-baseline-$(date +%Y%m%d)`. Do not use date-only tags such as `h100-$(date +%Y%m%d)`, and do not copy stale example dates from docs.
6. Before validation, smoke tests, baseline, or any candidate run, initialize the campaign with `bash scripts/init_run.sh <tag>`. This must create or switch to `autoresearch/<tag>` and initialize `results.tsv`.
7. Verify `git branch --show-current` starts with `autoresearch/`. If it does not, stop before running experiments; do not run campaigns on `main`, `upstream/main`, the starter branch, or a shared feature branch.
8. After reading `program.md`, treat this file as the active task profile, then inspect only the supporting files needed for the next action:
   - `tasks/cifar10/mutation_schema.yaml` for hard mutation bounds
   - `tasks/cifar10/client.py`, `tasks/cifar10/custom_aggregators.py`, and
     `tasks/cifar10/job.py` for the active code surface
   - `README.md` or `ACKNOWLEDGEMENTS.md` only when user-facing setup or provenance context is needed
9. Verify the prepared environment is ready:
   - `PYTHON=.venv/bin/python TASK_DIR=tasks/cifar10 make validate`
   - `PYTHON=.venv/bin/python TASK_DIR=tasks/cifar10 make smoke`
10. Confirm the setup and start with the baseline.

## Experimentation

Each experiment should run under a **fixed communication, data, and evaluation budget**. Keep the following fixed across a comparison campaign unless the human explicitly changes the campaign setup:
- `n_clients`
- `num_rounds`
- `batch_size`
- `eval_batch_size`
- `alpha`
- `seed`
- `model_arch`
- `max_model_params`
- whether cross-site evaluation is enabled
- `final_eval_clients`

Some of these values are technically mutable in `mutation_schema.yaml`, but changing them starts a new comparison budget. Do not compare scores across runs with different values for the fixed budget fields above unless the run is explicitly labeled as a new campaign or subcampaign. Architecture-search scores must be labeled with their `model_arch` and `max_model_params`; do not mix them with optimizer-only `moderate_cnn` results as if they were the same search.

Local compute is mutable within a campaign when each candidate stays within `RUN_TIMEOUT_SECONDS`. The default is epoch-based training with `--aggregation_epochs 4`; the agent may instead use `--local_train_steps <n>` for exact optimizer steps per client per round when that is a better search axis. Do not vary both local-compute modes in the same narrow sweep. Rank primarily by score, then use `runtime_seconds` as the cost/tie-breaker.

Default H100 candidate budget:
- `--n_clients 8`
- `--num_rounds 20`
- `--aggregation_epochs 4`
- `--local_train_steps 0`
- `--batch_size 64`
- `--alpha 0.5`
- `--seed 0`
- `--model_arch moderate_cnn`
- `--max_model_params 5000000`
- `--aggregator weighted`
- cross-site evaluation enabled
- final global evaluation on `site-1`
- `--eval_batch_size 1024`
- `RUN_TIMEOUT_SECONDS=1200`
- deterministic PyTorch/DataLoader training enabled

Each candidate targets a capped run on one local H100. The 80 GB H100 can usually support several same-budget candidates concurrently; if runs consistently finish much sooner, sweep local compute first with either `aggregation_epochs` or `local_train_steps`. If they time out or hit CUDA OOM, reduce candidate parallelism before changing communication, model, parameter-cap, or data contracts.
The current CIFAR-10 validation loader is identical on every simulated client, so final scoring evaluates the server/global model on `site-1` by default and keeps the output in NVFlare's `cross_site_val/cross_val_results.json` path. Use `--final_eval_clients all` only for an audit run or after changing validation to be site-specific.
Training splits are cached by fixed data-budget fields under `/tmp/cifar10_splits/autofl_cifar10_<n>sites_alpha<a>_seed<s>`. Do not make the split path depend on the candidate `--name`; repeated candidates with the same `n_clients`, `alpha`, and `seed` should reuse the same `.npy` indices.
Client training derives stable per-site RNG seeds from `--seed`, enables PyTorch deterministic algorithms, disables cuDNN benchmarking, and seeds DataLoader shuffling/workers. Treat `--no_deterministic_training` as a separate noisy subcampaign.

### Initial algorithm calibration

After the first weighted baseline run in a fresh campaign, run an algorithm calibration sequence before open-ended hyperparameter tuning. Keep the same communication and local-compute budget and compare the available baseline implementations:

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
2. Generate a small batch of hyperparameter candidates under the same communication budget and runtime cap, up to `PARALLEL_CANDIDATES`.
3. Launch each candidate with a unique `--name`, `RUN_LOG`, and description value.
4. Wait for the whole batch to finish or time out.
5. Rank the completed candidates against the ledger by score, inspect crashes with `tail -n 50 <run_log>`, and only then decide the next mutation or sweep.

Example single-H100 calibration skeleton:

```bash
mkdir -p run_logs
PYTHON=${PYTHON:-.venv/bin/python}
PARALLEL_CANDIDATES=${PARALLEL_CANDIDATES:-4}
COMMON_ARGS="--n_clients 8 --num_rounds 20 --aggregation_epochs 4 --local_train_steps 0 --batch_size 64 --eval_batch_size 1024 --alpha 0.5 --seed 0 --model_arch moderate_cnn --max_model_params 5000000 --final_eval_clients site-1"

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
  PYTHON="${PYTHON}" RUN_LOG="run_logs/${name}.log" RUN_TIMEOUT_SECONDS=1200 \
    TASK_DIR=tasks/cifar10 bash scripts/run_iteration.sh --description "${desc}" --target tasks/cifar10/client.py -- \
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
If candidates repeatedly run out of memory on the local H100, reduce `PARALLEL_CANDIDATES` before reducing the communication budget or starting a clearly labeled smaller-budget subcampaign.
After each batch, rank the ledger with:

```bash
"${PYTHON}" scripts/summarize_results.py results.tsv --status candidate --top "${PARALLEL_CANDIDATES:-4}"
```

After finalizing reviewed statuses, run the plateau watchdog before selecting another local sweep:

```bash
"${PYTHON}" scripts/plateau_watchdog.py results.tsv
```

If it prints `recommendation=literature`, stop local hyperparameter jittering, run the literature loop, record a `literature` row, and launch the selected source-backed candidates next.
If it prints `recommendation=continue`, do not start a literature review just because one or two small batches missed; choose a clearer local sweep axis, narrow around near-misses, or inspect `mutation_schema.yaml` for another allowed axis.

When finalizing a same-H100 batch, use the CIFAR default width:

```bash
"${PYTHON}" scripts/finalize_batch_status.py results.tsv --last "${PARALLEL_CANDIDATES:-4}" --keep-best --discard-others
```

With the default 20-minute candidate cap and `PARALLEL_CANDIDATES=4`,
throughput can reach roughly 12 candidates per hour, or about 96 candidates
across an eight-hour overnight run, if the H100, host CPU, storage, and data
loaders sustain four concurrent jobs. Reduce the width when contention hurts
reliability or score comparability.

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

## CIFAR-10 Score

Use cross-site evaluation when scoring candidates. The helper script
`scripts/extract_score.py` reads the comparable metric from:

```text
<result_dir>/server/simulate_job/cross_site_val/cross_val_results.json
```

The optimized score is the final server/global model,
`SRV_FL_global_model.pt`; ignore `SRV_best_FL_global_model.pt` when ranking
candidates. Higher is better.

For architecture-search campaigns, `--max_model_params` is a hard gate, not a
soft preference. Rank candidates primarily by final score under the same
communication/evaluation budget. Use `runtime_seconds` only as a coarse
secondary signal for near ties.

## First Run

The first logged run in a fresh CIFAR-10 campaign must be the weighted baseline
with the default H100 budget in this file. Record it in `results.tsv`, mark it
`keep`, and then run the initial algorithm calibration before open-ended
hyperparameter tuning.

All ledger handling, literature mode, continuation behavior, and output-log
discipline are inherited from `program.md`.
