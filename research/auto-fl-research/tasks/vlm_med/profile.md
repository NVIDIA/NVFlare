# Medical VLM Auto-FL Task Profile

Read `program.md` first. This file is the task-specific profile for a 3-site
medical VLM Auto-FL campaign.

This profile owns the medical VQA workload, prepared VLM environment, local
single-GPU assumptions, Qwen3-VL adapter budget, score definition, and
VLM-specific mutation surface. The general experiment loop, ledger rules,
literature workflow, and output discipline remain in `program.md`.

This profile owns its implementation files under `tasks/vlm_med/`, including
`client.py`, `job.py`, `model.py`, `train_utils.py`,
`med_vlm_data_utils.py`, `custom_aggregators.py`, `mutation_schema.yaml`, and
`requirements.txt`.

## Setup

To set up a medical VLM campaign, work with the user to:

1. Use the prepared local VLM environment by default:
   `PYTHON=/workspace/vlm_env/bin/python`. The VLM dependency set is listed in
   `tasks/vlm_med/requirements.txt`. If the human explicitly provides a
   different `PYTHON` value, treat that override as authoritative, but require it
   to be a prepared VLM/NVFlare environment with Torch, Transformers, PEFT,
   Datasets, Safetensors, and NVFlare installed.
2. Do not create virtual environments or install dependencies unless the human
   explicitly asks you to. Do not search for Python interpreters with filesystem
   globs such as `ls /usr/bin/python*`, `ls /workspace/.venv*/bin/python*`,
   `which python`, or similar discovery commands.
3. If `PYTHON` is missing, empty, not executable, or cannot import the required
   VLM/NVFlare stack, tell the human to activate or repair the local VLM
   environment, or provide an explicit compatible `PYTHON` override. Do not
   guess.
4. Verify the prepared interpreter before running repo commands:
   - `test -x "$PYTHON"`
   - `"$PYTHON" -c "import sys, torch, transformers, peft, datasets, safetensors, nvflare; assert sys.version_info[:2] >= (3, 11), sys.version; print(sys.executable); print(nvflare.__version__)"`
5. Use cached Hugging Face assets by default and avoid accidental downloads
   during controlled runs:
   - `export HF_HOME=/workspace/.hf_cache`
   - `export HF_HUB_OFFLINE=1`
   - `export HF_DATASETS_OFFLINE=1`
   - `export TRANSFORMERS_OFFLINE=1`
   - `export TOKENIZERS_PARALLELISM=false`
   - `export PYTHONPATH=/workspace/VLM_Benchmark:${PYTHONPATH:-}`
   - `export TMPDIR=/workspace/tmp`
   - `export AUTOFL_SIM_WORKSPACE_ROOT=/workspace/nvflare_simulation`
   - `export VLM_LITERATURE_THRESHOLD=${VLM_LITERATURE_THRESHOLD:-2}`
6. Use Qwen3-VL-2B as the default local model:
   - `MODEL_REPO=/workspace/.hf_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct`
   - `MODEL_REF=$(cat "$MODEL_REPO/refs/main")`
   - `MODEL_PATH="$MODEL_REPO/snapshots/$MODEL_REF"`
7. Agree on a descriptive run tag derived from the current date at runtime. Use
   the pattern `<node>-medvlm-<topic>-$(date +%Y%m%d)`, for example
   `rtx6000ada-medvlm-baseline-$(date +%Y%m%d)` or
   `localgpu-medvlm-fedprox-$(date +%Y%m%d)`. Do not use date-only tags and do
   not copy stale example dates from docs.
8. Before validation, smoke tests, baseline, or any candidate run, initialize
   the campaign with `bash scripts/init_run.sh <tag>`. This must create or
   switch to `autoresearch/<tag>` and initialize `results.tsv`.
9. Verify `git branch --show-current` starts with `autoresearch/`. If it does
   not, stop before running experiments; do not run campaigns on `main`,
   `upstream/main`, the starter branch, or a shared feature branch.
10. After reading `program.md`, treat this file as the active task profile, then
    inspect only the supporting files needed for the next action:
    - `tasks/vlm_med/mutation_schema.yaml` for hard mutation bounds
    - `tasks/vlm_med/client.py`, `tasks/vlm_med/job.py`,
      `tasks/vlm_med/model.py`, `tasks/vlm_med/train_utils.py`, and
      `tasks/vlm_med/med_vlm_data_utils.py` for the active VLM code surface
    - `README.md` or `ACKNOWLEDGEMENTS.md` only when user-facing setup or
      provenance context is needed
11. Verify the prepared environment is ready:
    - `PYTHON=/workspace/vlm_env/bin/python TASK_DIR=tasks/vlm_med make validate`
    - run a no-ledger VLM smoke with the fixed task args below, reduced to
      `--num_rounds 1 --local_train_steps 1 --batch_size 1 --max_samples_per_site 2 --max_eval_samples 2`:

      ```bash
      PYTHON=/workspace/vlm_env/bin/python TASK_DIR=tasks/vlm_med \
      SMOKE_ARGS="--task med-vlm --n_clients 3 --num_rounds 1 \
      --local_train_steps 1 --batch_size 1 --grad_accum 1 --eval_batch_size 1 \
      --max_samples_per_site 2 --max_eval_samples 2 \
      --site_datasets vqa-rad,slake,path-vqa --seed 0 \
      --model_name_or_path ${MODEL_PATH} --hf_cache_dir /workspace/.hf_cache \
      --model_arch qwen3vl_lora_adapter --max_model_params 8000000 \
      --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
      --aggregator weighted --final_eval_clients all --name smoke_medvlm" \
      make smoke
      ```
12. Confirm the setup and start with the baseline.

## Task

Train Qwen3-VL LoRA adapters across three simulated medical VLM sites:

- `site-1=vqa-rad`
- `site-2=slake`
- `site-3=path-vqa`

Clients train local adapter weights and send adapter-state DIFFs. The server
aggregates adapter tensors only. Full base-model weights must remain local to
the client process and Hugging Face cache.

The 3-site mapping is fixed by `--site_datasets`. Do not make the site mapping
depend on candidate `--name`. Keep VQA-RAD's train-derived validation holdout
reserved by default so validation examples are not trained on.

## Experimentation

Each experiment should run under a fixed communication, data, model, adapter,
evaluation, and runtime budget. Keep the following fixed across a comparison
campaign unless the human explicitly changes the campaign setup:

- `n_clients`
- `num_rounds`
- `batch_size`
- `grad_accum`
- `eval_batch_size`
- `max_samples_per_site`
- `max_eval_samples`
- `site_datasets`
- `seed`
- `model_name_or_path`
- `model_arch`
- `max_model_params`
- `lora_r`
- `lora_alpha`
- `lora_dropout`
- whether cross-site evaluation is enabled
- `final_eval_clients`

Some of these values are technically mutable in `mutation_schema.yaml`, but
changing them starts a new comparison budget. Do not compare scores across runs
with different fixed-budget fields unless the run is explicitly labeled as a new
campaign or subcampaign.

Local compute is mutable within a campaign when each candidate stays within
`RUN_TIMEOUT_SECONDS`. Use exact optimizer-step training by default with
`--local_train_steps 4`, `--batch_size 8`, and `--grad_accum 1`. Use
epoch-based training with `--local_train_steps 0 --aggregation_epochs <n>` only
in a clearly labeled local-compute subcampaign. Do not vary both local-compute
modes in the same narrow sweep.

Site-specific local steps are available only for labeled local-compute
subcampaigns via `--site_local_steps_spec`, and require
`--local_train_steps > 0`. Treat unequal local steps as a direct server-weight
and data-exposure lever unless the selected aggregator explicitly normalizes
steps.

Default local GPU candidate budget:

- `--task med-vlm`
- `--n_clients 3`
- `--num_rounds 20`
- `--local_train_steps 4`
- `--batch_size 8`
- `--grad_accum 1`
- `--eval_batch_size 1`
- `--max_samples_per_site 512`
- `--max_eval_samples 512`
- `--site_datasets vqa-rad,slake,path-vqa`
- `--seed 0`
- `--model_name_or_path "$MODEL_PATH"`
- `--hf_cache_dir /workspace/.hf_cache`
- `--model_arch qwen3vl_lora_adapter`
- `--max_model_params 8000000`
- `--lora_r 16`
- `--lora_alpha 32`
- `--lora_dropout 0.05`
- `--aggregator weighted`
- cross-site evaluation enabled
- final global evaluation on `all`
- `RUN_TIMEOUT_SECONDS=1200`
- `CUDA_VISIBLE_DEVICES=0`
- deterministic PyTorch/DataLoader training enabled

Recommended shell form:

```bash
COMMON_ARGS="--task med-vlm --n_clients 3 --num_rounds 20 \
--local_train_steps 4 \
--batch_size 8 --grad_accum 1 --eval_batch_size 1 \
--max_samples_per_site 512 --max_eval_samples 512 \
--site_datasets vqa-rad,slake,path-vqa --seed 0 \
--model_name_or_path ${MODEL_PATH} --hf_cache_dir /workspace/.hf_cache \
--model_arch qwen3vl_lora_adapter --max_model_params 8000000 \
--lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
--aggregator weighted --final_eval_clients all"
```

Rank primarily by score. Use `runtime_seconds`, sustained GPU utilization, peak
memory, and power draw as cost/tie-breakers. Do not treat one `nvidia-smi`
sample as utilization evidence; NVFlare round transitions, aggregation,
evaluation, and logging create short idle windows.

## Local Single-GPU Mode

This profile targets a local single-GPU VLM node. Default to:

```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PARALLEL_CANDIDATES=${PARALLEL_CANDIDATES:-1}
```

Run one heavy VLM candidate at a time unless a representative run proves there
is enough GPU memory, host memory, and I/O headroom. If multiple GPUs are
visible but this campaign should remain local/single-GPU, pin to
`CUDA_VISIBLE_DEVICES=0`; do not add multi-GPU scheduling or GPU lane mapping in
this profile.

Every candidate must have a unique `RUN_LOG`, `--name`, and description. Never
reuse a result directory name or run log for a different candidate.

After finalizing reviewed statuses, run the plateau watchdog with the VLM
threshold before selecting another local sweep:

```bash
"$PYTHON" scripts/plateau_watchdog.py results.tsv --max-scored-since-reset "$VLM_LITERATURE_THRESHOLD"
```

If `PARALLEL_CANDIDATES` is greater than 1, launch only same-budget candidates
with unique logs and names, wait for the whole batch to finish or time out, then
rank and finalize as a batch. Reduce candidate width before reducing the VLM
budget when memory or I/O contention appears.

The root overlay filesystem can fill during repeated NVFlare simulations. Use
`/workspace/tmp` for Python temporary directories and
`/workspace/nvflare_simulation` for `SimEnv` workspaces on this local node. A
candidate that fails before deployment because `/tmp` is full is an
infrastructure crash, not an algorithm result; fix workspace routing and rerun
the same candidate before judging the method.

## Edit Surface

Preferred mutation order:

1. `client.py` - VLM training, evaluation telemetry, optimizer, local loss, and
   adapter-only DIFF upload behavior.
2. `job.py` - VLM Recipe API wiring, CLI args, final evaluation clients, and
   aggregator selection.
3. `model.py` - Qwen3-VL adapter-state shape, registered architecture variants,
   and parameter-budget checks.
4. `train_utils.py` - prompt-preserving aggregate VLM evaluation helpers.
5. `med_vlm_data_utils.py` - deterministic site mapping and VLM data
   loading.
6. `custom_aggregators.py` - task-local aggregation experiments.

Do not duplicate shared `scripts/`, `templates/`, ledger helpers, reporting
helpers, or plotting utilities for the VLM profile. Prefer `TASK_DIR`,
`JOB_SCRIPT`, and `CLIENT_CONTRACT_PATH` overrides when shared tooling can
support the profile.

## What You Can Do

Safe medical VLM search axes include:

- learning rate, site-specific LR scaling, or site LR decay
- exact local optimizer steps with `--local_train_steps`
- site-specific local steps with `--site_local_steps_spec`
- `aggregation_epochs` when `local_train_steps=0`
- `batch_size`, `grad_accum`, `eval_batch_size`, and `num_workers` within the
  single-GPU memory budget
- `weight_decay`
- `max_grad_norm`
- FedProx via `--fedproxloss_mu`
- client-local FedDyn-style regularization via `--feddyn_alpha`
- sharpness-aware minimization via `--sam_rho` and `--sam_eps`
- LoRA+ via `--lora_plus_ratio` and bounded site-specific LoRA+ ratios
- LoRA trainable module or layer subsets when the adapter state keys remain
  stable
- adapter-only architecture variants under an explicitly labeled adapter-budget
  subcampaign
- `max_pixels`, `min_pixels`, and `max_new_tokens` only when the campaign is
  explicitly labeled as a resolution or generation-budget subcampaign
- source-backed aggregation variants that preserve DIFF aggregation, parameter
  keys, and `NUM_STEPS_CURRENT_ROUND`

Architecture or adapter-budget experiments are allowed only as labeled
subcampaigns. Keep `--model_arch qwen3vl_lora_adapter`, `--max_model_params`,
rank, alpha, and dropout fixed unless the campaign is explicitly labeled as an
architecture or adapter-budget subcampaign.

If you add a new client CLI knob, wire it through `job.py` before listing it in
`mutation_schema.yaml` or using it in a candidate. A client-only argument is not
reachable through the NVFlare recipe path.

## What You Cannot Do

- switch `ParamsType.DIFF` to full-weight uploads
- move or exchange full base VLM weights through NVFlare
- remove `NUM_STEPS_CURRENT_ROUND`
- remove or bypass the `flare.is_evaluate()` path
- change prompt format, evaluation examples, score metric, or site mapping
  inside a comparable campaign
- change `model_arch`, adapter rank, adapter alpha, adapter dropout, or
  `max_model_params` mid-campaign without a labeled subcampaign
- add server-coupled metadata beyond the existing contract without a labeled
  protocol subcampaign and human approval
- add new dependencies without human approval
- hard-code machine-specific paths in `client.py` or `job.py`
- log prompts, answers, predictions, site names, dataset names, or sample-level
  validation details from controlled medical VLM runs

## Medical VLM Score

Use cross-site evaluation when scoring candidates. Candidate comparison uses:

```text
<result_dir>/server/simulate_job/cross_site_val/cross_val_results.json
```

The optimized score is the final server/global model's medical VQA `token_f1`,
averaged over the configured final cross-site evaluation clients. Higher is
better. Rank `SRV_FL_global_model.pt`; do not switch to
`SRV_best_FL_global_model.pt` mid-campaign.

The shared `scripts/extract_score.py` extractor supports task metrics including
`accuracy` and `token_f1`; this profile's comparable metric is `token_f1`.

The default final evaluation clients are `all`, giving one aggregate over
VQA-RAD, SLAKE, and PathVQA validation slices. A task/data/evaluation
subcampaign is required before changing prompt format, answer normalization,
validation examples, site mapping, or metric semantics.

For full-suite promotion work, use a cheap VQA-v2 or generalization guard only
when the campaign explicitly includes that guard. Do not mix guard metrics into
the primary medical VQA score.

## First Run

The first logged run in a fresh medical VLM campaign must be the weighted
baseline with the default local GPU budget in this file. Record it in
`results.tsv`, mark it `keep`, and only then launch candidate mutations.

All ledger handling, literature mode, continuation behavior, and output-log
discipline are inherited from `program.md`.
