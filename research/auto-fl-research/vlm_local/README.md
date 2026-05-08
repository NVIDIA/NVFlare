# VLM Local Profile

This directory is a VLM-specific profile layered on the parent Auto-FL harness.
It shows how the same Auto-FL concept can be adapted to another scenario with a
different task and running environment. It is not self-contained by design.

Keep shared tooling in the parent folder:

- parent `scripts/`
- parent `templates/`
- parent `custom_aggregators.py`
- parent reporting/plotting helpers

This profile only keeps files that differ from the parent default:

- `program.md` - local VLM task and single-GPU instructions
- `client.py` - medical VLM client loop and LoRA training
- `job.py` - VLM recipe wiring and shared parent aggregator import
- `model.py` - Qwen3-VL adapter-state model
- `data/med_vlm_data_utils.py` - VQA-RAD/SLAKE/PathVQA site mapping
- `requirements.txt` - VLM runtime dependencies
- `mutation_schema.yaml` - VLM-specific mutation bounds

## Local Single-GPU Contract

Use this profile when the campaign target is a 3-site medical VLM workload on a
local machine with one visible GPU. The local-machine constraint is part of the
experiment contract unless the profile is explicitly rewritten for a multi-GPU
environment.

- Set `CUDA_VISIBLE_DEVICES=0` when multiple GPUs are visible but the campaign
  should stay comparable to a one-GPU baseline.
- Keep `PARALLEL_CANDIDATES` low enough that concurrent runs do not contend for
  GPU memory. For this VLM profile, use `1` unless the profile is updated to
  document a different safe width.
- Keep `RUN_TIMEOUT_SECONDS`, `batch_size`, `grad_accum`, `max_pixels`,
  `max_eval_samples`, and `max_samples_per_site` inside the documented budget so
  candidate scores remain comparable.
- Put machine-specific paths in environment variables such as
  `VLM_BENCHMARK_ROOT`, `HF_HOME`, and the simulator workspace root. Avoid
  hard-coding local paths in `client.py` or `job.py`.
- Use adapter-only exchange for the VLM so NVFlare does not move full base model
  weights every round.

The profile files are authoritative for task-specific behavior:

- `program.md` defines the VLM task contract, local single-GPU assumptions, fixed
  budget, allowed edit surface, and stop conditions.
- `mutation_schema.yaml` defines the bounded mutation surface for this task and
  local runtime budget.
- `client.py`, `job.py`, `model.py`, `train_utils.py`, and `data/` must agree on
  the task, metric, model state, sites, and environment assumptions.

Continue to use the shared parent runner, ledger, literature loop, plotting, and
reporting utilities from the parent directory.

## Adapting to a New Task or Environment

For a new VLM task or a different local environment, compose a thin task profile
that keeps only the files that must differ from the parent harness. Do not copy
parent `scripts/`, `templates/`, or reporting utilities unless the shared
version cannot support the profile through path or environment overrides.

1. Define the task contract in `program.md`, `README.md`, and
   `mutation_schema.yaml`. Record the fixed comparison budget: sites, rounds,
   local training budget, data limits, seed, metric, timeout, model/adapter
   budget, and final evaluation clients.

2. Replace the data layer in `data/`. Implement deterministic site mapping and
   train/eval splits. Document any dataset root variables, cache directories,
   gated data access, and expected on-disk layout.

3. Update `client.py`. Keep the NVFlare loop intact: `flare.init()`, receive the
   global model, strict `load_state_dict`, local train/eval, `compute_model_diff`,
   set `NUM_STEPS_CURRENT_ROUND`, and send `ParamsType.DIFF`. Add only the
   task-specific loader, collator, optimizer, and evaluation logic around that
   contract.

4. Update `model.py`. Register the server state exchanged by NVFlare. For a large
   VLM, expose only LoRA/adapters or another bounded trainable state. Keep budget
   checks so a candidate cannot silently switch to full-model exchange.

5. Update `job.py`. Match the client arguments, site names, GPU assumptions,
   simulator workspace, cross-site validation, final evaluation clients, and
   parent aggregator imports. Add a profile-local aggregator only when a campaign
   needs behavior that should not live in the shared parent
   `custom_aggregators.py`.

6. Update `train_utils.py` and, when needed, the parent `scripts/extract_score.py`.
   Make the final score path stable before running candidate comparisons. For
   VQA-style tasks, prefer one token-F1 or exact-match definition and use the
   same server/global model key for every run.

7. Update `requirements.txt`. Include the exact task dependencies and auth/cache
   expectations, such as `transformers`, `datasets`, `peft`, `qwen-vl-utils`,
   `HF_TOKEN`, and `HF_HOME`. Keep these dependencies local to the task profile.

8. Validate before handing the profile to an agent:

   ```bash
   "${PYTHON:-python3}" scripts/validate_contract.py <profile>/client.py
   "${PYTHON:-python3}" scripts/pycompile_sources.py <profile>
   ```

For a heavy VLM task, first run a no-ledger smoke through the parent
`scripts/run_iteration.sh` with `CLIENT_CONTRACT_PATH` and `JOB_SCRIPT` set to
the profile files. Start the autoresearch branch and baseline run only after the
smoke path is stable.

## Validate

From the parent `auto-fl-research` directory:

```bash
"${PYTHON:-python3}" scripts/validate_contract.py vlm_local/client.py
"${PYTHON:-python3}" scripts/pycompile_sources.py vlm_local
```

## Run

From the parent `auto-fl-research` directory, use the shared runner with the VLM job and client paths:

```bash
export PYTHON=vlm_local/.venv/bin/python
export CLIENT_CONTRACT_PATH=vlm_local/client.py
export JOB_SCRIPT=vlm_local/job.py
export HF_HOME=/workspace/.hf_cache

MODEL_REPO=/workspace/.hf_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct
MODEL_REF=$(cat "$MODEL_REPO/refs/main")
export MODEL_PATH="$MODEL_REPO/snapshots/$MODEL_REF"

bash scripts/init_run.sh localgpu-medvlm-$(date +%Y%m%d)

PYTHON="$PYTHON" CLIENT_CONTRACT_PATH="$CLIENT_CONTRACT_PATH" JOB_SCRIPT="$JOB_SCRIPT" \
RUN_LOG=run_logs/vlm_baseline.log RUN_TIMEOUT_SECONDS=1200 \
  bash scripts/run_iteration.sh --description "vlm baseline weighted" --target vlm_local/client.py -- \
  --task med-vlm --n_clients 3 --num_rounds 20 --aggregation_epochs 1 \
  --local_train_steps 4 --batch_size 8 --grad_accum 1 --eval_batch_size 1 \
  --max_samples_per_site 512 --max_eval_samples 512 \
  --site_datasets vqa-rad,slake,path-vqa --seed 0 \
  --model_name_or_path "$MODEL_PATH" --hf_cache_dir /workspace/.hf_cache \
  --model_arch qwen3vl_lora_adapter --max_model_params 8000000 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --aggregator weighted --final_eval_clients all --name vlm_baseline_weighted
```

`job.py` imports `WeightedAggregator` from the parent `custom_aggregators.py`. Add a local VLM-specific aggregator file only when a campaign actually needs one.
