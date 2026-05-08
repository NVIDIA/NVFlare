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

## Validate

From the parent `auto-fl-research` directory:

```bash
PYTHON="$PYTHON" scripts/validate_contract.py vlm_local/client.py
PYTHON="$PYTHON" scripts/pycompile_sources.py vlm_local
```

## Run

From the parent `auto-fl-research` directory, use the shared runner with the VLM job and client paths:

```bash
export PYTHON=vlm_local/.venv/bin/python
export CLIENT_CONTRACT_PATH=vlm_local/client.py
export JOB_SCRIPT=vlm_local/job.py

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
