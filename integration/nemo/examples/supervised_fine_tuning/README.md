# Federated Full-Model SFT with NeMo AutoModel and Nemotron 3 Nano

This example fine-tunes all trainable weights of a Nemotron 3 Nano language model with NeMo AutoModel in an NVFlare
simulation. It uses the modern NVFlare Recipe API (`FedAvgRecipe` with `SimEnv`) and an external training client that
exchanges full model weights through the NVFlare Client API.

Full-model supervised fine-tuning is much heavier than the PEFT example. The federated payload is the full model state,
not a LoRA adapter. Keep clients sequential for local simulations unless you have enough GPU and host memory for
multiple full models at once.

## Environment

Use the NeMo AutoModel container:

```bash
docker run --runtime=nvidia -it --rm --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/NVFlare:/workspace -w /workspace/integration/nemo/examples/supervised_fine_tuning \
  nvcr.io/nvidia/nemo-automodel:26.04
```

Install NVFlare from the repository checkout inside the container:

```bash
python -m pip install -e /workspace
```

The example defaults to `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, the smallest local Nemotron 3 Nano Edge model. The
30B-A3B model is not the right default for full-model SFT because transferring and averaging the full 30B state is much
more expensive than adapter-only PEFT.

## Data

Create a tiny deterministic instruction-following split for smoke tests:

```bash
python data/create_synthetic_sft_data.py --out_dir data/synthetic_sft
```

The generated JSONL files use the same `{"input": ..., "output": ...}` schema produced by the legacy Alpaca, Dolly, and
OpenAssistant preprocessing utilities in `utils/`.

## Initial Model

Create the full-model checkpoint that the NVFlare server persists and aggregates:

```bash
python prepare_initial_model.py \
  --model_name_or_path nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --output models/nemotron3_nano_4b_sft_init.pt \
  --device_map auto
```

This materializes the full 4B model state. Expect a multi-GiB checkpoint and enough CPU/GPU memory to load the base
model.

## Run

Start with a one-client, one-round H100 smoke:

```bash
python job.py \
  --n_clients=1 \
  --num_rounds=1 \
  --num_threads=1 \
  --gpu="[0]" \
  --max_steps=1 \
  --seq_length=512 \
  --limit_train_samples=1 \
  --limit_validation_samples=1 \
  --no-use_chat_template \
  --initial_model_ckpt=models/nemotron3_nano_4b_sft_init.pt
```

Then run a small federated simulation sequentially:

```bash
python job.py \
  --n_clients=3 \
  --num_rounds=2 \
  --num_threads=1 \
  --gpu="[0]" \
  --max_steps=1 \
  --seq_length=512 \
  --limit_train_samples=1 \
  --limit_validation_samples=1 \
  --no-use_chat_template \
  --initial_model_ckpt=models/nemotron3_nano_4b_sft_init.pt
```

Keeping `--num_threads=1` avoids multiplying GPU memory by running all clients at the same time. This is still a
federated simulation: clients use different site files and FedAvg aggregates their full-model updates after each round.

Use `--backend=mock` for a CPU/static smoke of NVFlare full-model exchange only. This does not run NeMo AutoModel.

## Prediction

Generate from the final global model checkpoint:

```bash
python predict.py \
  --model_name_or_path nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --checkpoint /tmp/nvflare/nemotron3_nano_sft/nemotron3-nano-sft/server/simulate_job/app_server/FL_global_model.pt \
  --output_json models/nemotron3_nano_sft_predictions.json
```

## Full-Model Continuity Across Rounds

This example uses multi-round FedAvg for full-model SFT. The external AutoModel process may restart on each client task
to release GPU memory, but the model state does not restart from the initial checkpoint:

1. The server sends the current global full-model state at the start of every round.
2. The client saves that state as `incoming_global_model.pt`.
3. AutoModel builds the base model and warm-starts all matching tensors from `incoming_global_model.pt`.
4. The client sends the full updated model state.
5. FedAvg averages the full-model tensors and replaces the global model with the aggregate.

This is intentionally heavier than PEFT. For lower-memory experimentation, use the PEFT example. For full-model SFT,
use small step counts for smoke tests and increase clients/rounds only on hardware sized for full-model transfer and
aggregation.

## H100 Validation

The H100 validation command and observed results are recorded after running the end-to-end smoke for this branch.

## Legacy Megatron Path

The legacy `nemo_nvflare/`, `jobs/templates/`, and `utils/create_configs.py` files are kept for reference to the older
Megatron/Lightning workflow. The modern path for this example is `job.py` plus the NVFlare Recipe and Client APIs.
