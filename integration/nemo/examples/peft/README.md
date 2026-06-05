## Federated PEFT with NeMo AutoModel and Nemotron 3 Nano

This example fine-tunes a Nemotron 3 language model with LoRA adapters in an NVFlare simulation. It uses the modern
NVFlare API surface:

- `job.py` builds a `FedAvgRecipe` and runs it with `SimEnv`.
- `automodel_peft_client.py` uses explicit NVFlare Client API calls: `flare.init()`, `flare.receive()`, and
  `flare.send()`.
- The server uses `PTFileModelPersistor` with an adapter-only PyTorch checkpoint, so it does not instantiate the
  base language model.

The default fine-tuning target is `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, the small local "Edge" Nano variant. This
keeps the example practical on a single high-memory GPU while staying in the Nemotron 3 family. For larger Nano 30B-A3B
deployment, use `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` for base-model inference, or merge/quantize the tuned
LoRA adapter after training. The NVFP4 checkpoint minimizes inference memory; the PEFT training path still needs enough
GPU memory to fine-tune the selected Nano model.

Smaller NVIDIA models such as Llama-Nemotron 8B are useful, but they are not Nemotron 3 family models and are not the
target of this example.

## Dependencies

Use a current NeMo AutoModel environment with the `automodel` CLI available. The NVIDIA NeMo AutoModel docs recommend
either `pip install nemo-automodel` or the `nvcr.io/nvidia/nemo-automodel` container. From the NVFlare repository root:

```bash
DOCKER_IMAGE="nvcr.io/nvidia/nemo-automodel:26.04"
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
  --num_clients=3 \
  --out_dir=data/FinancialPhraseBank-v1.0_split
```

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

To mirror the original notebook prediction check more closely, run five rounds with 200 local steps per client:

```bash
python job.py \
  --n_clients=3 \
  --num_rounds=5 \
  --num_threads=1 \
  --gpu="[0]" \
  --max_steps=200 \
  --seq_length=512 \
  --limit_validation_samples=256 \
  --no-use_chat_template \
  --initial_adapter_ckpt=models/nemotron3_nano_lora_init.pt
```

Use `--no-balance_train_labels` if you want the capped training subset to preserve the original site-file order.

Check the notebook sentiment prompts against the final global adapter:

```bash
python predict_sentiment.py \
  --server_model /tmp/nvflare/nemotron3_nano_peft/nemotron3-nano-peft/server/simulate_job/app_server/FL_global_model.pt \
  --output_dir models/nemotron3_nano_lora_final \
  --output_json models/nemotron3_nano_prediction_summary.json
```

The expected classifications are:

```text
The products have a low salt and fat content . sentiment: neutral
The agreement is valid for four years . sentiment: neutral
Diluted EPS rose to EUR3 .68 from EUR0 .50 . sentiment: positive
Profit before taxes decreased by 9 % to EUR 187.8 mn in the first nine months of 2008 , compared to EUR 207.1 mn a year earlier . sentiment: negative
```

For split-level accuracy and Macro-F1, score the validation and test files by exact label log probability:

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

## Adapter Continuity Across Rounds

This example uses multi-round FedAvg for the federated setting. The external AutoModel process may restart on each
client task to release GPU memory, but the fine-tuning state does not restart from the initial adapter:

1. The server sends the current global LoRA adapter at the start of every round.
2. The client saves that adapter as `incoming_adapter`.
3. AutoModel builds the base model, injects LoRA modules, and warm-starts those modules from `incoming_adapter`.
4. The client sends the full updated adapter.
5. FedAvg averages the adapter tensors and replaces the global adapter with the aggregate.

Use `--backend=mock` for a CPU/static continuity check of the same Recipe, Client API, and full-adapter path before
running GPU training.

## Reproducing the H100 30B Smoke Result

The default 4B Edge path keeps the example approachable and uses a true federated setting above. The larger
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` result below was a single-client, one-round H100 smoke of the model,
adapter, prediction, and exact-evaluation path. It is not a federated benchmark. The verified smoke run used:

- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- GPU: one H100, with GPU memory around 64-68 GiB during training/evaluation.
- Setup: `n_clients=1`, `num_rounds=1`, `num_threads=1`.
- Data: original shuffled Financial PhraseBank training split, not label-balanced.
- Train rows: 3100 total, with 1852 neutral, 866 positive, and 382 negative examples.
- PEFT: LoRA rank 8, alpha 16, dropout 0.05, target modules `all-linear`.
- Optimizer settings: learning rate `5e-5`, micro/global batch size 1, gradient accumulation 1.
- Sequence length: 512.
- Training length: 1500 local steps, checkpoint/validation every 1500 steps.
- Prompt format: raw prompt-completion text, `"{sentence} sentiment:"`, no chat template.
- Exact evaluation: batch size 8, label choices `neutral`, `positive`, and `negative`.

If your 30B BF16 snapshot does not include tokenizer files, create a local model directory that contains the 30B model
files plus the tokenizer files from `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, then use that local directory for
`H100_MODEL_NAME_OR_PATH`.

Create a single-client shuffled train split:

```bash
python data/split_financial_phrase_data.py \
  --alpha=10.0 \
  --random_seed=20260605 \
  --data_path=data/FinancialPhraseBank-v1.0/financial_phrase_bank_train.jsonl \
  --num_clients=1 \
  --out_dir=data/FinancialPhraseBank-v1.0_30b_single
```

Prepare the 30B initial adapter:

```bash
export H100_MODEL_NAME_OR_PATH=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

python prepare_initial_adapter.py \
  --model_name_or_path "${H100_MODEL_NAME_OR_PATH}" \
  --output models/nemotron3_nano_30b_lora_init_rank8.pt \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --device_map auto
```

Run the single-client H100 smoke:

```bash
python job.py \
  --n_clients=1 \
  --num_rounds=1 \
  --num_threads=1 \
  --gpu="[0]" \
  --workspace=/tmp/nvflare/nemotron3_nano_peft_30b_h100 \
  --max_steps=1500 \
  --ckpt_every_steps=1500 \
  --val_every_steps=1500 \
  --seq_length=512 \
  --limit_validation_samples=256 \
  --learning_rate=5e-5 \
  --lora_rank=8 \
  --lora_alpha=16 \
  --lora_dropout=0.05 \
  --no-use_chat_template \
  --no-balance_train_labels \
  --train_split_dir=data/FinancialPhraseBank-v1.0_30b_single \
  --validation_file=data/FinancialPhraseBank-v1.0/financial_phrase_bank_val.jsonl \
  --initial_adapter_ckpt=models/nemotron3_nano_30b_lora_init_rank8.pt \
  --model_name_or_path "${H100_MODEL_NAME_OR_PATH}"
```

For a federated 30B run, keep clients sequential but use multiple clients and rounds, for example
`--n_clients=3 --num_rounds=3 --num_threads=1`, with a three-client train split. This preserves the federated FedAvg
setting without multiplying GPU memory across simultaneous client processes.

Convert the final server adapter and run the notebook prediction prompts:

```bash
python predict_sentiment.py \
  --model_name_or_path "${H100_MODEL_NAME_OR_PATH}" \
  --server_model /tmp/nvflare/nemotron3_nano_peft_30b_h100/nemotron3-nano-peft/server/simulate_job/app_server/FL_global_model.pt \
  --output_dir models/nemotron3_nano_30b_lora_final \
  --output_json models/nemotron3_nano_30b_prediction_summary.json \
  --lora_rank 8 \
  --lora_alpha 16 \
  --target_modules down_proj,in_proj,out_proj,up_proj
```

Run exact validation/test evaluation before and after fine-tuning:

```bash
python evaluate_sentiment.py \
  --model_name_or_path "${H100_MODEL_NAME_OR_PATH}" \
  --validation_file data/FinancialPhraseBank-v1.0/financial_phrase_bank_val.jsonl \
  --test_file data/FinancialPhraseBank-v1.0/financial_phrase_bank_test.jsonl \
  --output_dir models/nemotron3_nano_30b_base_exact_eval \
  --batch_size 8

python evaluate_sentiment.py \
  --model_name_or_path "${H100_MODEL_NAME_OR_PATH}" \
  --adapter_dir models/nemotron3_nano_30b_lora_final \
  --validation_file data/FinancialPhraseBank-v1.0/financial_phrase_bank_val.jsonl \
  --test_file data/FinancialPhraseBank-v1.0/financial_phrase_bank_test.jsonl \
  --output_dir models/nemotron3_nano_30b_exact_eval \
  --batch_size 8
```

The June 5, 2026 H100 smoke run produced the following exact-label results:

| Model / scoring | Val accuracy | Val Macro-F1 | Test accuracy | Test Macro-F1 | Test prediction counts |
| --- | ---: | ---: | ---: | ---: | --- |
| 30B BF16 base, no adapter | 0.3943 | 0.4410 | 0.4031 | 0.4449 | positive 834, negative 87, neutral 49 |
| 30B BF16 + LoRA | 0.8698 | 0.8538 | 0.8474 | 0.8392 | neutral 642, positive 202, negative 126 |
| 30B BF16 + LoRA, validation-selected label bias | 0.8827 | 0.8715 | 0.8619 | 0.8578 | neutral 596, positive 234, negative 140 |

The validation-selected bias adds `+1.8` to the positive label score and `+1.8` to the negative label score after model
scoring. It is post-hoc calibration only; it does not change the trained adapter.

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
