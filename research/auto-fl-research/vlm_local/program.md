# Local Medical VLM Auto-FL Program

This profile shows how the Auto-FL research loop can be adapted to a different
task and running environment without copying the parent harness. The parent
`auto-fl-research` directory owns shared scripts, templates, ledger helpers,
plotting, reporting, and mature aggregation utilities. This directory owns only
the VLM task and local single-GPU environment differences.

Treat this file as the agent entry point when running the VLM profile. Use the
parent `program.md` only as background for shared harness behavior; this file
overrides the parent CIFAR-10/H100 task, budget, edit surface, candidate width,
and scoring assumptions.

## Setup

Run commands from the parent `auto-fl-research` directory:

```bash
cd /workspace/research/auto-fl-research
export PYTHON=${PYTHON:-vlm_local/.venv/bin/python}
export CLIENT_CONTRACT_PATH=vlm_local/client.py
export JOB_SCRIPT=vlm_local/job.py
export HF_HOME=/workspace/.hf_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/workspace/VLM_Benchmark:${PYTHONPATH:-}
export PARALLEL_CANDIDATES=${PARALLEL_CANDIDATES:-1}
export VLM_LITERATURE_THRESHOLD=${VLM_LITERATURE_THRESHOLD:-2}

MODEL_REPO=/workspace/.hf_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct
MODEL_REF=$(cat "$MODEL_REPO/refs/main")
export MODEL_PATH="$MODEL_REPO/snapshots/$MODEL_REF"
```

Before validation, smoke tests, baseline, or candidate runs:

1. Verify the prepared interpreter. Do not create a virtual environment, install
   dependencies, or search for another Python unless the human explicitly asks.

   ```bash
   test -x "$PYTHON"
   "$PYTHON" -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version; print(sys.executable)"
   ```

2. Verify the local model and cache paths. If `MODEL_PATH`, `HF_HOME`, or
   `/workspace/VLM_Benchmark` is missing, stop and ask the human to prepare the
   local cache or benchmark checkout. Do not silently fall back to online model
   downloads during a comparable campaign.

3. Initialize a fresh autoresearch branch before any logged run:

   ```bash
   bash scripts/init_run.sh localgpu-medvlm-$(date +%Y%m%d)
   git branch --show-current
   ```

   The current branch must start with `autoresearch/`. Do not run campaigns on
   `main`, `upstream/main`, the starter branch, or a shared feature branch.

4. Validate the VLM profile with parent utilities:

   ```bash
   "$PYTHON" scripts/validate_contract.py "$CLIENT_CONTRACT_PATH"
   "$PYTHON" scripts/pycompile_sources.py vlm_local
   ```

The parent `make validate` and `make smoke` targets are for the parent
CIFAR-10 profile unless they are explicitly updated to accept profile paths.
For VLM, use the commands above and the shared `scripts/run_iteration.sh` with
`CLIENT_CONTRACT_PATH` and `JOB_SCRIPT` set.

## Task

Train Qwen3-VL LoRA adapters across three simulated medical VLM sites:

- `site-1=vqa-rad`
- `site-2=slake`
- `site-3=path-vqa`

Clients train local adapter weights and send adapter-state DIFFs. The server
aggregates adapter tensors only. Full base model weights must remain local to
the client process and Hugging Face cache.

## Goal

Maximize one comparable score under a fixed communication, data, model, adapter,
evaluation, and runtime budget.

The score is the final server/global model's medical VQA `token_f1`, averaged
over the configured final cross-site evaluation clients. Higher is better.
Candidate comparison uses the parent score path:

```text
<result_dir>/server/simulate_job/cross_site_val/cross_val_results.json
```

Rank primarily by score. Use `runtime_seconds` as a cost and tie-breaker for
near-equal scores. Do not collapse score and runtime into one scalar, and be
skeptical of tiny gains that cost much more wall-clock time or add brittle code.

## FL Contract

Preserve the NVFlare Client API contract:

- `flare.init()`
- `while flare.is_running()`
- `input_model = flare.receive()`
- `model.load_state_dict(input_model.params, strict=True)`
- `global_model = copy.deepcopy(model)`
- `compute_model_diff(model, global_model)`
- `flare.send(output_model)`
- `output_model.params_type == ParamsType.DIFF`
- `output_model.meta["NUM_STEPS_CURRENT_ROUND"]`
- the `flare.is_evaluate()` metrics path

Do not change evaluation data, prompt format, score extraction, adapter key
schema, site mapping, or model budget inside a comparable run series. Any such
change starts a new labeled task, data, model, or evaluation campaign.

## Baseline Budget

Use this baseline first in every fresh VLM campaign:

```text
--task med-vlm
--n_clients 3
--num_rounds 20
--aggregation_epochs 1
--local_train_steps 4
--batch_size 8
--grad_accum 1
--eval_batch_size 1
--max_samples_per_site 512
--max_eval_samples 512
--site_datasets vqa-rad,slake,path-vqa
--seed 0
--model_name_or_path ${MODEL_PATH}
--hf_cache_dir /workspace/.hf_cache
--model_arch qwen3vl_lora_adapter
--max_model_params 8000000
--lora_r 16
--lora_alpha 32
--lora_dropout 0.05
--aggregator weighted
--final_eval_clients all
```

Recommended shell form:

```bash
COMMON_ARGS="--task med-vlm --n_clients 3 --num_rounds 20 \
--aggregation_epochs 1 --local_train_steps 4 \
--batch_size 8 --grad_accum 1 --eval_batch_size 1 \
--max_samples_per_site 512 --max_eval_samples 512 \
--site_datasets vqa-rad,slake,path-vqa --seed 0 \
--model_name_or_path ${MODEL_PATH} --hf_cache_dir /workspace/.hf_cache \
--model_arch qwen3vl_lora_adapter --max_model_params 8000000 \
--lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
--aggregator weighted --final_eval_clients all"
```

First run:

```bash
mkdir -p run_logs
PYTHON="$PYTHON" CLIENT_CONTRACT_PATH="$CLIENT_CONTRACT_PATH" JOB_SCRIPT="$JOB_SCRIPT" \
RUN_LOG=run_logs/vlm_baseline_weighted.log RUN_TIMEOUT_SECONDS=1200 \
  bash scripts/run_iteration.sh --description "vlm baseline weighted" --target vlm_local/client.py -- \
  ${COMMON_ARGS} --name vlm_baseline_weighted
```

After the baseline finishes, review the row and mark it as kept:

```bash
"$PYTHON" scripts/summarize_results.py results.tsv --status candidate --top 1
"$PYTHON" scripts/finalize_batch_status.py results.tsv --last 1 --keep-best --discard-others
git add results.tsv
git commit -m "Record VLM baseline"
```

## Comparison Budget

Keep these fields fixed within a comparison campaign:

- `n_clients`
- `num_rounds`
- `batch_size`
- `grad_accum`
- `eval_batch_size`
- `max_samples_per_site`
- `max_eval_samples`
- `site_datasets`
- `seed`
- `model_arch`
- `max_model_params`
- `lora_r`
- `lora_alpha`
- `lora_dropout`
- cross-site evaluation enabled
- `final_eval_clients`

`aggregation_epochs` and `local_train_steps` are local-compute knobs. Use
`local_train_steps=0` for epoch-based training with `aggregation_epochs`; use
`local_train_steps>0` for exact optimizer steps per client per round. Vary only
one local-compute mode in a narrow sweep.

Changing model, adapter rank, adapter alpha, adapter dropout, data limits, site
mapping, prompt format, or final evaluation clients starts a new labeled budget.
Do not compare those scores against the baseline as if they were the same
campaign.

## Local Single-GPU Mode

This profile targets a local machine with one visible GPU. Default to:

```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PARALLEL_CANDIDATES=${PARALLEL_CANDIDATES:-1}
```

Run one heavy VLM candidate at a time unless a representative run proves there
is enough GPU memory, host memory, and I/O headroom. If multiple GPUs are
visible, pin to `CUDA_VISIBLE_DEVICES=0`; do not add multi-GPU scheduling or GPU
lane mapping in this profile.

Every candidate must have a unique `RUN_LOG`, `--name`, and description.
Never reuse a result directory name or run log for a different candidate.

## Edit Surface

Preferred mutation order:

1. `client.py` - VLM training, evaluation telemetry, optimizer, and local loss.
2. `job.py` - VLM Recipe API wiring, CLI args, final eval clients, and shared
   parent aggregator selection.
3. `model.py` - adapter-state shape, registered architecture, and parameter
   budget checks.
4. `data/med_vlm_data_utils.py` - deterministic site mapping and VLM data
   loading for a new task profile.
5. parent `custom_aggregators.py` - shared aggregation experiments when the
   behavior is mature enough to belong in the parent harness.

Do not duplicate parent `scripts/`, `templates/`, ledger helpers, reporting
helpers, or mature aggregator files inside `vlm_local/`.

## What You Can Do

Safe VLM-profile search axes include:

- learning rate, site-specific LR scaling, or site LR decay
- exact local optimizer steps with `--local_train_steps`
- site-specific local steps with `--site_local_steps_spec`
- `aggregation_epochs` when `local_train_steps=0`
- `batch_size`, `grad_accum`, `eval_batch_size`, and `num_workers` within the
  single-GPU memory budget
- `weight_decay`
- FedProx via `--fedproxloss_mu`
- LoRA trainable module subsets with `--train_lora_modules`
- `max_pixels`, `min_pixels`, and `max_new_tokens` when the campaign explicitly
  treats them as local runtime and quality knobs
- `--aggregator weighted` versus `--aggregator default`
- improvements to parent `WeightedAggregator` that preserve DIFF aggregation
  and parameter keys

Architecture or adapter-budget experiments are allowed only as labeled
subcampaigns. Keep `--model_arch qwen3vl_lora_adapter` and enforce
`--max_model_params`; do not register new architectures or raise parameter caps
without human approval.

If you add a new client CLI knob, wire it through `job.py` before listing it in
`mutation_schema.yaml` or using it in `scripts/run_iteration.sh`. A client-only
argument is not reachable through the NVFlare recipe path.

## What You Cannot Do

- switch `ParamsType.DIFF` to full-weight uploads
- move or exchange full base VLM weights through NVFlare
- remove `NUM_STEPS_CURRENT_ROUND`
- remove or bypass the `flare.is_evaluate()` path
- change prompt format, evaluation examples, score metric, or site mapping
  inside a comparable campaign
- use unsupported aggregator names such as FedAvgM, FedAdam, or SCAFFOLD until
  `vlm_local/job.py` explicitly wires them for this profile
- add server-coupled metadata beyond the existing contract without a labeled
  protocol subcampaign and human approval
- add new dependencies without human approval
- hard-code machine-specific paths in `client.py` or `job.py`
- copy parent scripts or templates into `vlm_local/`

## Logging Results

Keep a tracked `results.tsv` file on experiment branches with this header:

```text
commit	score	runtime_seconds	budget	status	target	description	artifacts
```

`scripts/init_run.sh <tag>` creates the header before the baseline. The parent
`scripts/run_iteration.sh` also creates or migrates the header before a logged
run and appends the row only after the candidate exits.

Statuses are:

- `candidate` for completed rows that have not been reviewed
- `keep` for the baseline or a survivor worth building on
- `discard` for reviewed non-survivors
- `crash` for failed runs
- `literature` for non-scored literature-review events

After every completed run or batch, update reviewed rows. Do not let stale
reviewed rows remain `candidate`; the progress plot reads this column directly.

Use helpers instead of hand-editing TSV rows:

```bash
"$PYTHON" scripts/summarize_results.py results.tsv --status candidate --top "${PARALLEL_CANDIDATES:-1}"
"$PYTHON" scripts/finalize_batch_status.py results.tsv --last "${PARALLEL_CANDIDATES:-1}" --keep-best --discard-others
```

Commit `results.tsv` after the baseline and at checkpoint boundaries. Commit
surviving code changes on the active `autoresearch/` branch as soon as they are
kept. Do not commit bulky runtime artifacts such as run logs, NVFlare result
directories, Hugging Face caches, or generated checkpoints.

## Experiment Loop

Default VLM loop:

1. Confirm the branch starts with `autoresearch/`.
2. Inspect `results.tsv`, this file, and `mutation_schema.yaml`.
3. Choose one small allowed mutation or one CLI-only candidate.
4. Edit the smallest possible set of files, if code is needed.
5. Commit code changes before launching the candidate when the candidate
   depends on those code changes.
6. Run the candidate with the shared parent runner:

   ```bash
   PYTHON="$PYTHON" CLIENT_CONTRACT_PATH="$CLIENT_CONTRACT_PATH" JOB_SCRIPT="$JOB_SCRIPT" \
   RUN_LOG=run_logs/<unique-name>.log RUN_TIMEOUT_SECONDS=1200 \
     bash scripts/run_iteration.sh --description "<short description>" --target <main-target> -- \
     ${COMMON_ARGS} <one-axis args> --name <unique-name>
   ```

7. Read the summarized result from the script output.
8. If the run crashed, inspect only `tail -n 50 <run_log>` and fix the smallest
   root cause. Stop after a few retries on the same failure family.
9. Rank completed candidates against `results.tsv`.
10. Keep the candidate if it materially improves score or is comparable with
    simpler/faster behavior; otherwise discard it.
11. Finalize statuses and commit `results.tsv`.
12. Run the plateau watchdog before selecting the next sweep:

    ```bash
    "$PYTHON" scripts/plateau_watchdog.py results.tsv --max-scored-since-reset "$VLM_LITERATURE_THRESHOLD"
    ```

If `PARALLEL_CANDIDATES` is greater than 1, launch only same-budget candidates
with unique logs and names, wait for the whole batch to finish or time out, then
rank and finalize as a batch. Reduce candidate width before reducing the VLM
budget when memory or I/O contention appears.

## Never Stop

Once setup and the baseline are complete, continue autonomously until manually
interrupted. Do not pause to ask whether to keep going.

Cycle through:

- choose a clear allowed axis
- run one same-budget candidate, or a small same-budget batch if explicitly safe
- rank against the ledger
- keep, narrow, or discard
- finalize `results.tsv` statuses
- run the plateau watchdog
- switch to literature mode when triggered
- commit surviving code and ledger checkpoints
- repeat

If local ideas run out, inspect recent near-misses in `results.tsv`, reread
`mutation_schema.yaml`, combine compatible kept settings, or switch to the
literature loop. Stay within the hard VLM contract.

## Literature Loop

When progress stalls, stop random local jitter and use a literature-grounded
recovery loop. The VLM profile uses the parent artifacts:

- `templates/literature_loop.md`
- `templates/mutation_report.md`
- `scripts/log_literature_review.py`
- `scripts/plateau_watchdog.py`

Trigger literature mode when:

- the watchdog prints `recommendation=literature`;
- two completed VLM candidates or batches fail to improve and recent near-misses
  do not suggest a clear safe local axis;
- repeated crashes share one root cause and a source-backed fix is needed; or
- no non-duplicate safe axis remains under `mutation_schema.yaml`.

Do not log literature rows for routine one-off misses while the next local
sweep is obvious.

Before searching, gather working memory:

- current best stack and score from `results.tsv`
- last 20 scored rows and repeated crashes
- confirmed null or worse axes
- allowed mutation surface from `mutation_schema.yaml`
- current fixed budget and `CUDA_VISIBLE_DEVICES` pinning

Start a literature timer:

```bash
"$PYTHON" scripts/log_literature_review.py --start --description "vlm plateau: <symptom>"
```

Use the parent `templates/literature_loop.md` as the worksheet. Keep the work
source-backed and compact:

1. Generate three distinct search queries from the observed VLM/FL failure mode.
2. Search at least two available paper sources when possible. Prefer primary
   papers over blog posts.
3. Triage 6-10 candidate papers down to 3-5 relevant papers.
4. Extract exactly three challenge cards: challenge, evidence, matching
   `results.tsv` symptom, why it matters here, and allowed implementation file.
5. Generate 3-5 proposal cards with mechanism, source refs, implementation
   surface, exact args or code change, expected effect, runtime cost, contract
   risk, and falsifying observation.
6. Reject duplicates, known null ideas, protocol changes, new dependencies,
   prompt/evaluation changes, full-model exchange, and over-budget model ideas.
7. Score proposals from 1-5 on expected gain, contract safety, simplicity,
   evidence, novelty, and runtime cost. Use:

   ```text
   2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost
   ```

8. Select the next candidate or small batch from the top proposals.
9. Finish the literature event before launching candidates:

   ```bash
   "$PYTHON" scripts/log_literature_review.py --finish --description "vlm literature: <symptom>; selected <mechanisms>"
   ```

10. Put fuller citations and hypotheses in `templates/mutation_report.md`.
    Include compact refs in candidate descriptions, for example
    `[src: Li20 FedProx arXiv:1812.06127]`.

Compatible literature-derived directions include:

- FedProx or local regularization for client drift
- site-specific local step or LR scheduling for heterogeneous medical sites
- robust weighted aggregation that preserves DIFF parameter keys
- bounded adapter module selection or adapter-rank subcampaigns
- optimizer, weight decay, gradient clipping, or image-token budget changes
- prompt-preserving evaluation and prediction-audit diagnostics

Incompatible directions without human approval include:

- changing prompt or metric definitions
- adding datasets to the comparison budget
- exchanging full VLM weights
- adding new server-client protocol tensors
- importing a new research framework or scheduler dependency
- copying parent scripts/templates into this profile

## Output Discipline

Redirect full training output to `RUN_LOG` and inspect only summaries or failure
tails. Keep context compact during long runs. At checkpoints, generate a
progress plot with the parent helper:

```bash
"${PYTHON:-python3}" scripts/plot_progress.py results.tsv --output progress.png
```
