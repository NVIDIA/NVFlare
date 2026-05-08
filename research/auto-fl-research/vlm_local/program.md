# Local Medical VLM Profile

Use this profile when the task/environment differs from the parent starter.

## Scope

The parent folder owns shared harness pieces: scripts, templates, result ledger format, plotting/reporting, and mature aggregation utilities. This folder owns only VLM-specific files.

## Task

Train Qwen3-VL LoRA adapters across three simulated medical VLM sites:

- `site-1=vqa-rad`
- `site-2=slake`
- `site-3=path-vqa`

Clients send adapter-state DIFFs. The server aggregates adapter tensors only.

## Contract

Preserve:

- `flare.init()`
- `while flare.is_running()`
- `input_model = flare.receive()`
- `model.load_state_dict(input_model.params, strict=True)`
- `compute_model_diff(model, global_model)`
- `flare.send(output_model)`
- `output_model.params_type == ParamsType.DIFF`
- `output_model.meta["NUM_STEPS_CURRENT_ROUND"]`
- the `flare.is_evaluate()` metrics path

Do not change evaluation data, prompt format, score extraction, or adapter key schema inside a comparable run series.

## Environment

```bash
export PYTHON=vlm_local/.venv/bin/python
export CLIENT_CONTRACT_PATH=vlm_local/client.py
export JOB_SCRIPT=vlm_local/job.py
export HF_HOME=/workspace/.hf_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/workspace/VLM_Benchmark:${PYTHONPATH:-}

MODEL_REPO=/workspace/.hf_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct
MODEL_REF=$(cat "$MODEL_REPO/refs/main")
MODEL_PATH="$MODEL_REPO/snapshots/$MODEL_REF"
```

Run shared scripts from the parent `auto-fl-research` directory with `CLIENT_CONTRACT_PATH` and `JOB_SCRIPT` set as above.

## Baseline Budget

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

Keep communication, data, model, seed, and evaluation fields fixed in a campaign. Sweep one axis at a time.

## Edit Surface

- `client.py`: VLM training/evaluation changes.
- `job.py`: VLM recipe and CLI wiring.
- `model.py`: adapter-state shape/rank logic.
- `data/med_vlm_data_utils.py`: site mapping or a new VLM task.
- parent `custom_aggregators.py`: shared aggregation experiments, unless a campaign needs a local VLM-only aggregator.

Do not duplicate parent scripts/templates in this folder.
