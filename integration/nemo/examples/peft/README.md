## Federated PEFT with NeMo AutoModel and Nemotron 3

This example fine-tunes a Nemotron 3 language model with LoRA adapters in an NVFlare simulation. It uses the modern
NVFlare API surface:

- `job.py` builds a `FedAvgRecipe` and runs it with `SimEnv`.
- `automodel_peft_client.py` uses explicit NVFlare Client API calls: `flare.init()`, `flare.receive()`, and
  `flare.send()`.
- The server uses an example-local `PTFileModelPersistor` extension with an adapter-only PyTorch checkpoint, so it
  does not instantiate the base language model and keeps every round aggregate for verification.

The default fine-tuning target is `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, the small local "Edge" Nano variant. This
keeps the example practical on a single high-memory GPU while staying in the Nemotron 3 family. For larger Nano 30B-A3B
deployment, use `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` for base-model inference, or merge/quantize the tuned
LoRA adapter after training. The NVFP4 checkpoint minimizes inference memory; the PEFT training path still needs enough
GPU memory to fine-tune the selected Nano model.

Smaller NVIDIA models such as Llama-Nemotron 8B are useful, but they are not Nemotron 3 family models and are not the
target of this example.

The optional `--model_profile=lightning35` profile targets
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`. It starts from NVIDIA's Lightning LoRA recipe: Transformer Engine
attention, PyTorch linear and expert backends, rank 8, alpha 32, dropout 0, `*.out_proj` exclusion, two repeated MTP
iterations, MTP loss scale 0.1, and tensor/context/expert parallel sizes of one. Explicit CLI options override profile
defaults. Omitting the profile preserves the Nano defaults.

The Lightning extension keeps BF16 base weights and one GPU per client. Quantization, full-model SFT, tool-use data,
serving optimization, and distributed training within a client are outside its scope.

## Dependencies

Use a current NeMo AutoModel environment with the `automodel` CLI available. The NVIDIA NeMo AutoModel docs recommend
either `pip install nemo-automodel` or the `nvcr.io/nvidia/nemo-automodel` container. From the NVFlare repository root:

```bash
DOCKER_IMAGE="nvcr.io/nvidia/nemo-automodel:26.08"
docker run --gpus all -it --rm --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${PWD}:/nvflare" \
  -w /nvflare/integration/nemo/examples/peft \
  "${DOCKER_IMAGE}"
```

Inside the container, install NVFlare from this repository. The `main` branch may depend on unreleased NVFlare features,
so use the local checkout until the matching package is published:

```bash
pip install -e /nvflare
```

You also need access to the gated Hugging Face model. Log in before preparing the adapter or running the simulation:

```bash
huggingface-cli login
```

## Data

This example keeps the Financial PhraseBank task from the original PEFT notebook. Download `FinancialPhraseBank-v1.0`
from the dataset provider, then run the NeMo preprocessing script referenced in the legacy notebook so the following
files exist:

```text
data/FinancialPhraseBank-v1.0/financial_phrase_bank_train.jsonl
data/FinancialPhraseBank-v1.0/financial_phrase_bank_val.jsonl
data/FinancialPhraseBank-v1.0/financial_phrase_bank_test.jsonl
```

Split the training data into federated site files:

```bash
python data/split_financial_phrase_data.py \
  --alpha=10.0 \
  --data_path=data/FinancialPhraseBank-v1.0/financial_phrase_bank_train.jsonl \
  --validation_path=data/FinancialPhraseBank-v1.0/financial_phrase_bank_val.jsonl \
  --test_path=data/FinancialPhraseBank-v1.0/financial_phrase_bank_test.jsonl \
  --remove_train_overlap \
  --random_seed=0 \
  --num_clients=3 \
  --out_dir=data/FinancialPhraseBank-v1.0_split
```

The splitter groups duplicate training sentences onto one site. With `--remove_train_overlap`, it removes training rows
whose sentence occurs in validation or test while leaving validation and test unchanged. It writes the prepared training
file and `split_manifest.json` with source/prepared hashes, removal counts, class counts, and the sentence-disjointness
check. Create this split once and reuse it for every training seed. Omit `--remove_train_overlap` to reject such input
overlap instead of resolving it.

## Initial Adapter

Create an adapter-only checkpoint for the server. This is the only model artifact the NVFlare server persists and
aggregates.

```bash
python prepare_initial_adapter.py \
  --model_name_or_path nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --output models/nemotron3_nano_lora_init.pt \
  --lora_rank 8 \
  --lora_alpha 16 \
  --device_map auto
```

Add `--load_in_4bit` if your environment has bitsandbytes and you want to reduce memory while materializing the initial
adapter shapes.

If you already have a Hugging Face PEFT adapter directory, convert it without loading the base model:

```bash
python prepare_initial_adapter.py \
  --from_adapter_dir /path/to/adapter \
  --output models/nemotron3_nano_lora_init.pt
```

This conversion command applies to the Nano path. Lightning initialization and reload use NeMo AutoModel's native
model factory and checkpoint implementation; Hugging Face PEFT interoperability is not a supported deliverable unless
the adapter is separately converted and tested.

For Lightning, resolve one Hugging Face revision and use it for both model and tokenizer:

```bash
MODEL_REVISION=$(python -c "from huggingface_hub import HfApi; print(HfApi().model_info('nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16').sha)")
python prepare_initial_adapter.py \
  --model_profile=lightning35 \
  --model_revision="${MODEL_REVISION}" \
  --tokenizer_revision="${MODEL_REVISION}" \
  --seed=42
```

## Run

Start with a one-client, one-round tiny smoke on GPU:

```bash
python job.py \
  --n_clients=1 \
  --num_rounds=1 \
  --num_threads=1 \
  --gpu="[0]" \
  --max_steps=1 \
  --seq_length=512 \
  --no-use_chat_template \
  --initial_adapter_ckpt=models/nemotron3_nano_lora_init.pt
```

Then run the default three-client sequential simulation. Keeping `--num_threads=1` avoids multiplying GPU memory by
running all clients at the same time. When the local training sample window is capped, the client uses deterministic
label-balanced sampling by default so short demo runs see neutral, positive, and negative examples instead of only the
first rows from each site split. The default dataset format is raw prompt-completion text, matching the prediction
prompts below; pass `--use_chat_template` only if you also plan to evaluate with chat-formatted prompts.

```bash
python job.py \
  --n_clients=3 \
  --num_rounds=3 \
  --num_threads=1 \
  --gpu="[0]" \
  --seq_length=512 \
  --no-use_chat_template \
  --initial_adapter_ckpt=models/nemotron3_nano_lora_init.pt
```

To reproduce the 30B H100 result below, prepare the initial adapter from the 30B model, then run three rounds with
300 local steps per client and a lower learning rate:

```bash
python prepare_initial_adapter.py \
  --model_name_or_path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --output models/nemotron3_nano_30b_lora_init.pt \
  --lora_rank 8 \
  --lora_alpha 16 \
  --device_map auto
```

```bash
python job.py \
  --n_clients=3 \
  --num_rounds=3 \
  --num_threads=1 \
  --gpu="[0]" \
  --model_name_or_path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --max_steps=300 \
  --seq_length=512 \
  --limit_validation_samples=256 \
  --learning_rate=5e-5 \
  --no-use_chat_template \
  --initial_adapter_ckpt=models/nemotron3_nano_30b_lora_init.pt
```

Use `--no-balance_train_labels` if you want the capped training subset to preserve the original site-file order.

Check the notebook sentiment prompts against the final global adapter. For the default 4B run, use this as a functional
adapter-conversion smoke and rely on the exact split evaluation below for quality metrics. The documented 30B H100
reference reproduced all sample prompts.

```bash
python predict_sentiment.py \
  --server_model /tmp/nvflare/nemotron3_nano_peft/nemotron3-nano-peft/server/simulate_job/app_server/FL_global_model.pt \
  --output_dir models/nemotron3_nano_lora_final \
  --output_json models/nemotron3_nano_prediction_summary.json
```

The expected classifications for the 30B H100 reference are:

```text
The products have a low salt and fat content . sentiment: neutral
The agreement is valid for four years . sentiment: neutral
Diluted EPS rose to EUR3 .68 from EUR0 .50 . sentiment: positive
Profit before taxes decreased by 9 % to EUR 187.8 mn in the first nine months of 2008 , compared to EUR 207.1 mn a year earlier . sentiment: negative
```

For split-level accuracy and Macro-F1, score the validation and test files by exact label log probability. Use the
same `--model_name_or_path` that was used to create and train the adapter; the command below shows the default 4B path:

```bash
python evaluate_sentiment.py \
  --model_name_or_path nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --adapter_dir models/nemotron3_nano_lora_final \
  --validation_file data/FinancialPhraseBank-v1.0/financial_phrase_bank_val.jsonl \
  --test_file data/FinancialPhraseBank-v1.0/financial_phrase_bank_test.jsonl \
  --output_dir models/nemotron3_nano_exact_eval \
  --batch_size 8
```

The client stages each received global adapter in a Hugging Face PEFT-style directory named `incoming_adapter`. The
default AutoModel config uses `automodel_adapter_loader.py` to build the base model, inject LoRA modules, and warm-start
those modules from `incoming_adapter` before every local training segment. If you need to customize the AutoModel YAML,
pass a template with `--automodel_config_template`; placeholders such as `${incoming_adapter_dir}` are available for
custom adapter-loading flows:

```bash
python job.py \
  --automodel_config_template=/path/to/custom_automodel_config.yaml \
  --initial_adapter_ckpt=models/nemotron3_nano_lora_init.pt
```

Use `--backend=mock` for a CPU/static smoke of NVFlare adapter exchange only. This does not run NeMo AutoModel.

### Lightning H100 validation

Run the complete staged validation from a retained `tmux` session on the H100 host:

```bash
tmux new -s nvflare-lightning35
CHECKOUT_DIR=/path/to/NVFlare \
  RUN_ROOT=/path/to/validation-output \
  ./integration/nemo/examples/peft/run_h100_lightning35.sh
```

The runner pins `nvcr.io/nvidia/nemo-automodel:26.08`, records the resolved image digest, creates a dedicated model
cache, resolves one model/tokenizer revision, installs the mounted NVFlare checkout, selects one idle GPU, and runs the
stages sequentially. It stops when a required gate fails. If the two-step smoke runs out of memory, it retries that
workload with activation checkpointing; a second failure is recorded as a single-GPU feasibility failure.

The cache normally lives below the timestamped run directory. To resume after a runner failure without downloading the
pinned snapshot again, set `CACHE_ROOT` to the previous attempt's `cache/huggingface` directory. The model and tokenizer
revision checks still run before training.

The runner preserves client adapters and manifests, every server round aggregate, exact-label evaluation output,
commands, package versions, exit codes, timing logs, and GPU telemetry under a timestamped directory in
`RUN_ROOT` (or `/tmp/nvflare/` by default). `verify_federated_run.py` independently computes every aggregate in FP32
using the actual optimizer-step weights. The final acceptance check requires both seeds to lower held-out validation
response-token loss and their mean final test Macro-F1 to equal or exceed the base model. Intermediate rounds evaluate
validation only; only the predetermined final round is evaluated on test.

The smoke reload gate requires identical accuracy, Macro-F1, confusion matrix, prediction counts, and response-token
count across two clean native loads. It allows an absolute response-token-loss difference of at most `5e-4` for
non-bit-exact Transformer Engine reductions.

Docker access must work before launching. The runner reports a blocked prerequisite and exits without changing
`/var/run/docker.sock` ownership or permissions. `run_h100_nano_regression.sh` supplies the separate two-client,
two-round Nano regression in its documented `26.04` image.

Nano preserves its native adapter dtype during exchange by default. Add `--fp32_adapter_exchange` to a Nano `job.py`
run when the server must accumulate adapters in FP32, such as an independently verified model-comparison campaign.
Lightning exchange is always FP32.

## Adapter Continuity Across Rounds

This example uses multi-round FedAvg for the federated setting. The external AutoModel process may restart on each
client task to release GPU memory, but the fine-tuning state does not restart from the initial adapter:

1. The server sends the current global LoRA adapter at the start of every round.
2. The client saves that adapter as `incoming_adapter`.
3. AutoModel builds the base model and injects LoRA modules. Lightning then loads the complete incoming adapter through
   AutoModel's native checkpointer before the first local optimizer step; Nano retains its existing loader.
4. The client sends the full updated adapter.
5. FedAvg averages the adapter tensors and replaces the global adapter with the aggregate.

Lightning clients reject incomplete, unexpected, duplicate-normalized, shape-incompatible, non-finite, or
manifest-conflicting adapters. They export only after successful training and record received, loaded, and outgoing
hashes, tensor counts, actual optimizer steps, update norm, and checkpoint location. Each local segment creates a fresh
optimizer and scheduler.

Use `--backend=mock` for a CPU/static continuity check of the same Recipe, Client API, and full-adapter path before
running GPU training.

## Reproducing the H100 Three-Round FL Result

The June 5, 2026 H100 validation used the 30B-A3B model and the real three-client, three-round federated workflow:

- Code: PR branch `codex/nemo-peft-nemotron3` at commit `ddd548cec`.
- Container: `nvcr.io/nvidia/nemo-automodel:26.04` with NeMo AutoModel `0.4.0+9687b04c`.
- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`; the H100 validation used a local cached copy at
  `/base_out/combined_30b_with_4b_tokenizer_20260605` inside the container.
- Setup: `n_clients=3`, `num_rounds=3`, `num_threads=1`, `--gpu="[0]"`.
- Transfer: full LoRA adapter tensors with `TransferType.FULL`.
- Data: Financial PhraseBank split into three `alpha=10.0` client files.
- Local training: 300 steps per client per round, label-balanced capped training window, validation capped at 256 rows.
- PEFT: LoRA rank 8, alpha 16, dropout 0.05, target modules `all-linear`.
- Optimizer settings: learning rate `5e-5`, micro/global batch size 1, gradient accumulation 1.
- Prompt format: raw prompt-completion text, `"{sentence} sentiment:"`, no chat template.
- Exact evaluation: batch size 8, label choices `neutral`, `positive`, and `negative`.

FedAvg ran all three rounds and clients loaded 12,008/12,008 incoming adapter tensors in later rounds, including
round 2. The AutoModel subprocess restarts between tasks to free GPU memory, but the adapter state is the current global
adapter, not the initial adapter. LoRA training reported about 64 GiB GPU memory during local steps on the H100.

Before and after exact-label scoring:

| Model / scoring | Val accuracy | Val Macro-F1 | Test accuracy | Test Macro-F1 | Test prediction counts |
| --- | ---: | ---: | ---: | ---: | --- |
| 30B BF16 base, no adapter | 0.3943 | 0.4410 | 0.4031 | 0.4449 | positive 834, negative 87, neutral 49 |
| 30B BF16 + 3-round FL LoRA | 0.8570 | 0.8475 | 0.8454 | 0.8391 | neutral 528, positive 281, negative 161 |
| 30B BF16 + 3-round FL LoRA, validation-selected label bias | 0.8595 | 0.8518 | 0.8423 | 0.8395 | neutral 498, positive 312, negative 160 |

The validation-selected bias adds `+1.5` to the positive label score and `+0.0` to the negative label score after model
scoring. It is post-hoc calibration only; it does not change the trained adapter. The raw adapter result is the main
before/after number because it does not require post-hoc calibration.

The same run reproduced the notebook-style prediction prompts with all expected labels:

```text
The products have a low salt and fat content . sentiment: neutral
The agreement is valid for four years . sentiment: neutral
Diluted EPS rose to EUR3 .68 from EUR0 .50 . sentiment: positive
Profit before taxes decreased by 9 % to EUR 187.8 mn in the first nine months of 2008 , compared to EUR 207.1 mn a year earlier . sentiment: negative
```

For lower-memory experimentation, keep the default 4B Edge model and the same Recipe, Client API, sequential
multi-client pattern, and full-adapter transfer. The matching 4B H100 reference used 900 local steps per client per
round and reached validation Macro-F1 0.6695 raw / 0.7242 calibrated and test Macro-F1 0.6605 raw / 0.6998 calibrated.
For lower-memory local deployment of the 30B-A3B family, use the NVFP4 30B-A3B checkpoint or a merged, quantized tuned
checkpoint after training.

## Advanced Path

Megatron Bridge remains the right backend for larger 8+ GPU training runs that need Megatron-scale parallelism. For this
example, NeMo AutoModel PEFT is the default because NVIDIA documents it for Hugging Face base models, JSONL datasets,
LoRA/QLoRA, and small GPU-count experiments.

## References

- [Nemotron 3 family](https://research.nvidia.com/labs/nemotron/Nemotron-3/)
- [Nemotron 3 Nano 4B BF16 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)
- [Nemotron 3 Nano 30B-A3B BF16 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [Nemotron 3 Nano in Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html)
- [Choose a PEFT Backend](https://docs.nvidia.com/nemotron/nightly/train-models/how-to/choose-peft-backend.html)
- [NeMo AutoModel PEFT guide](https://docs.nvidia.com/nemo/automodel/latest/guides/llm/finetune.html)
- [Nemotron QAD / NVFP4 note](https://research.nvidia.com/labs/nemotron/nemotron-qad/)
- [NVFP4 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4)
