# Federated Evo2 Fine-Tuning for Splice-Site Classification

This example fine-tunes `evo2/1b-8k-bf16:1.0` for three-class splice-site classification with BioNeMo Recipes,
Megatron Bridge, and NVIDIA FLARE. Three simulated sites train rank-16 LoRA adapters and a classification head.
NVFlare distributes one common initialization and applies sample-weighted FedAvg to those trainable tensors.

Only the adapters and classification head cross the federation boundary. They are exchanged and saved as CPU
float32 tensors, while the BF16 Evo2 backbone stays at each site. Every task creates fresh local optimizer,
scheduler, RNG, and Transformer Engine state, then discards that state after training. Every client verifies that
the backbone stays frozen during local training.

The workflow runs sequentially on one GPU. The sites are deterministic partitions of one public benchmark and
represent simulated institutions rather than independent data owners.

**Start here:** [walkthrough.ipynb](./walkthrough.ipynb)

## Pinned environment and inputs

| Component | Version |
| --- | --- |
| PyTorch container | `nvcr.io/nvidia/pytorch:26.02-py3` |
| BioNeMo Recipes | [`101d88b7e0b0ee40e5af303575eb9661620a3706`](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/tree/101d88b7e0b0ee40e5af303575eb9661620a3706) |
| Megatron Bridge | [`fcbb6031103d0ca845c1a54d4fee55ecfcca17b6`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/fcbb6031103d0ca845c1a54d4fee55ecfcca17b6) (v0.5.0) |
| causal-conv1d | [`c51519fb10d92ffa257c1982e1831d3243454b2f`](https://github.com/Dao-AILab/causal-conv1d/tree/c51519fb10d92ffa257c1982e1831d3243454b2f) (v1.6.1) |
| Evo2 checkpoint | `evo2/1b-8k-bf16:1.0`, converted from NeMo 2 to Megatron Bridge format |
| Dataset | [`InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised/tree/851f9946252e90c665cdb3cc3eedb78f1f26197c), revision `851f9946252e90c665cdb3cc3eedb78f1f26197c` |
| Task | `splice_sites_all`: 30,000 training and 3,000 test sequences of 600 bases |

The classifier, pooling, loss, and LoRA configuration are adapted from the pinned BioNeMo
[`evo2_classifier.py`](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/101d88b7e0b0ee40e5af303575eb9661620a3706/recipes/evo2_megatron/examples/evo2_classifier.py)
and its [Evo2 LoRA tutorial](https://docs.nvidia.com/bionemo-recipes/latest/main/recipes/recipes/evo2_megatron/examples/lora-fine-tuning-tutorial/).
The output classes are `no-splice`, `acceptor`, and `donor`.

## Requirements

Use Linux with an NVIDIA GPU, a recent NVIDIA driver, Docker with NVIDIA Container Toolkit, and enough storage for
the image, source-data cache, converted checkpoint, and NVFlare workspaces. The upstream tutorial documents Evo2-1B
LoRA on one 48 GB GPU.

From the NVFlare repository root, build and enter the pinned image:

```bash
cd examples/advanced/bionemo/evo2
docker build -t nvflare-evo2 .
./start_evo2.sh
```

The image installs the pinned packages in `requirements-runtime.txt`. The helper mounts the current NVFlare
checkout at `/workspace/nvflare`, installs it in editable mode, exposes one GPU, and persists caches and
`/tmp/nvflare` outputs under this example directory. It also runs a CUDA forward/backward check for the
causal-convolution extension before opening a shell. Override its defaults with `NVFLARE_EVO2_GPU`,
`NVFLARE_EVO2_IMAGE`, `NVFLARE_EVO2_CACHE`, or `NVFLARE_EVO2_WORKSPACE`.

If you use an equivalent environment, install `requirements.txt`, which supports NVFlare 2.9.0 and newer and
includes the runtime packages above. The container already supplies PyTorch, Transformer Engine, BioNeMo Evo2,
and Megatron Bridge.

```bash
python3 -m pip install -r requirements.txt
```

Set `BIONEMO_EVO2_CLASSIFIER` or pass `--classifier-file` when the pinned BioNeMo Recipes checkout is not at the
path configured by `start_evo2.sh`.

All remaining commands run from `examples/advanced/bionemo/evo2` inside the container. To use the notebook, start
Jupyter and open the URL it prints:

```bash
jupyter lab --ip=0.0.0.0 --no-browser --allow-root --port=8888
```

### Optional example tests

The example includes an opt-in test suite for its data preparation, checkpoint exchange, aggregation, process
launcher, evaluation, and mock training paths. It lives beside the example and is not collected by the repository's
default unit-test jobs. Run it from this directory in an environment with `pytest` and the requirements above
installed:

```bash
python3 -m pytest tests -q
```

## 1. Prepare the dataset

The preparation script downloads only the pinned `splice_sites_all` Parquet files. It preserves the official test
split, reserves 10% of the source training split for validation, checks exact-sequence and genomic-window leakage,
and creates deterministic site partitions.

Create the 3,000-sequence showcase with three IID sites:

```bash
python3 prepare_data.py \
  --output-dir ./data \
  --num-sites 3 \
  --partition iid \
  --showcase-size 3000 \
  --validation-fraction 0.1 \
  --seed 42
```

Use `--showcase-size 0` to retain all 27,000 rows left after reserving 3,000 validation rows. A deterministic
label-skew partition can be prepared with `--partition dirichlet --dirichlet-alpha 0.5`.

The command writes:

```text
data/
├── manifest.json
├── test.jsonl
├── validation.jsonl
└── train/
    ├── site-1.jsonl
    ├── site-2.jsonl
    └── site-3.jsonl
```

`manifest.json` records the dataset revision, preparation settings, row counts, label histograms, file hashes, and
leakage audit. Each JSONL row retains its sequence, genomic coordinate, integer label, and task. Before training,
the job checks the path, row count, byte count, and SHA-256 digest of every required input against this manifest.

## 2. Download and convert Evo2-1B

Download `evo2/1b-8k-bf16:1.0` with BioNeMo and convert it once to the Megatron Bridge checkpoint format:

```bash
python3 prepare_base_checkpoint.py --output ./models/evo2_1b_bf16_mbridge
```

The conversion is published only after its files and completion markers validate. Its provenance file records the
pinned model and tokenizer configuration plus a content inventory. All sites use this same converted checkpoint.

## 3. Create the common trainable initialization

Instantiate the frozen backbone, LoRA modules, and classification head once, then export only the trainable tensors:

```bash
torchrun --standalone --nproc_per_node=1 prepare_initial_model.py \
  --base-checkpoint ./models/evo2_1b_bf16_mbridge \
  --data-file ./data/validation.jsonl \
  --output ./models/evo2_lora_init.pt \
  --work-dir /tmp/nvflare/evo2_initialize \
  --seq-length 600 \
  --seed 1234 \
  --lora-dim 16 \
  --lora-alpha 32 \
  --lora-dropout 0.1
```

This zero-step setup uses the shared validation split only to instantiate the model and data module; it does not
optimize on those rows. The server sends the common initialization to every site. Tensor names, shapes, dtypes,
and finite values are validated whenever the state is loaded, aggregated, or saved. The exchange checkpoint
metadata binds the LoRA configuration, seed, validation data, base checkpoint, classifier source, and float32
exchange dtype.

## 4. Run a final-commit GPU smoke test

Before a longer run, exercise the real BioNeMo path with two clients, one round, and four local optimizer steps.
One warmup step can be too small to change a BF16 model tensor, so the smoke budget must be large enough to produce
a visible update. Use a new workspace path for every attempt.

```bash
python3 job.py \
  --data-dir ./data \
  --initial-checkpoint ./models/evo2_lora_init.pt \
  --base-checkpoint ./models/evo2_1b_bf16_mbridge \
  --workspace /tmp/nvflare/evo2_splice_smoke \
  --num-clients 2 \
  --num-rounds 1 \
  --local-steps 4 \
  --micro-batch-size 4 \
  --global-batch-size 32 \
  --gpu '[0]'
```

The smoke test passes when both trainers exit cleanly, the run summary records two contributors and changed
trainable tensors, the client logs report successful frozen-backbone checks, and the printed global checkpoint
reloads successfully with the evaluator below.

The recorded four-step H100 validation produced 0.3467 accuracy and 0.2402 macro-F1 on the 3,000-row official test
split. This deliberately short run is expected to remain near initialization; it validates the integration path,
tensor exchange, checkpoint reload, and teardown rather than model quality.

## 5. Run three-site federated LoRA training

The primary example runs three sites for ten rounds. Each site performs 20 optimizer steps per round with a
microbatch of four and gradient accumulation to global batch 32.

```bash
python3 job.py \
  --data-dir ./data \
  --initial-checkpoint ./models/evo2_lora_init.pt \
  --base-checkpoint ./models/evo2_1b_bf16_mbridge \
  --workspace /tmp/nvflare/evo2_splice_fedavg \
  --num-clients 3 \
  --num-rounds 10 \
  --local-steps 20 \
  --seq-length 600 \
  --micro-batch-size 4 \
  --global-batch-size 32 \
  --learning-rate 0.0005 \
  --min-learning-rate 0.00005 \
  --warmup-iters 2 \
  --eval-iters 10 \
  --seed 1234 \
  --lora-dim 16 \
  --lora-alpha 32 \
  --lora-dropout 0.1 \
  --gpu '[0]'
```

For every task, an external process rebuilds Evo2 from the unchanged base checkpoint, loads the current global
LoRA and head tensors, synchronizes the optimizer's trainable parameters, trains on that site's JSONL file, and
returns a float32 difference. The aggregator rejects missing, unexpected, or incompatible tensors and weights each
valid update by the site's audited training-example count. A workspace lock keeps the external GPU trainers
sequential even if their lightweight NVFlare launchers overlap.

At completion, `job.py` prints the result directory, global checkpoint, and `run_summary.json` path. The summary
records aggregation weights, final checkpoint identity, changed trainable-tensor count, audited inputs, and the
training configuration. Each client log identifies its attempt directory and reports the successful
frozen-backbone check; a detected backbone change aborts the task.

## 6. Reload and evaluate the final global checkpoint

Pass the exact `Global checkpoint:` path printed by `job.py` to the evaluator. For the default workspace, it is
typically under `/tmp/nvflare/evo2_splice_fedavg/evo2-splice-fedavg-lora/server/simulate_job/app_server/`.

```bash
GLOBAL_CHECKPOINT=/path/printed/by/job.py/FL_global_model.pt

torchrun --standalone --nproc_per_node=1 evaluate.py \
  --checkpoint "${GLOBAL_CHECKPOINT}" \
  --base-checkpoint ./models/evo2_1b_bf16_mbridge \
  --test-file ./data/test.jsonl \
  --manifest ./data/manifest.json \
  --output ./results/fedavg_lora/evaluation.json \
  --confusion-matrix ./results/fedavg_lora/confusion_matrix.png \
  --work-dir /tmp/nvflare/evo2_splice_evaluation \
  --seq-length 600 \
  --micro-batch-size 4 \
  --global-batch-size 32 \
  --seed 1234 \
  --lora-dim 16 \
  --lora-alpha 32 \
  --lora-dropout 0.1
```

Evaluation rebuilds the same frozen backbone, strictly loads the saved float32 trainable tensors at the BF16 model
boundary, and verifies that `test.jsonl` matches the prepared-data manifest. It requires every official test row
exactly once and writes accuracy, macro-F1, per-class metrics, the confusion matrix, runtime, peak GPU memory, and
checkpoint identity to the JSON report.

## Precursor H100 reference results

The following measurements come from a longer 37-round precursor implementation, not the ten-round command above.
That campaign used persistent client training state, checkpoint averaging, three simulated IID sites, and matched
independent local-only runs. The maintained example starts every task with fresh training state and no longer
includes checkpoint averaging, local-only campaigns, model selection, or baseline reporting. Treat these values as
reference evidence rather than acceptance thresholds or directly reproducible output.

| Endpoint | Validation accuracy / macro-F1 | Test accuracy / macro-F1 |
| --- | ---: | ---: |
| Raw federated round 36 | 0.9607 / 0.9606 | 0.9533 / 0.9535 |
| Federated mean, rounds 27–36 | 0.9617 / 0.9617 | 0.9560 / 0.9562 |
| Local site 1 | 0.9163 / 0.9165 | 0.9080 / 0.9083 |
| Local site 2 | 0.9110 / 0.9117 | 0.9010 / 0.9023 |
| Local site 3 | 0.8680 / 0.8690 | 0.8497 / 0.8514 |
| Descriptive local mean | 0.8984 / 0.8991 | 0.8862 / 0.8873 |

The raw final federated model exceeded the validation-selected best local model on test by 4.53 percentage points
in accuracy and 4.52 points in macro-F1. It exceeded the descriptive local mean by 6.71 and 6.62 points,
respectively.

The rounds 27–36 mean was selected using validation results. The local mean is the equal-site arithmetic mean of
three separately evaluated local models; it is not an ensemble, an averaged checkpoint, or a model trained on
combined local data. The official test aggregates were observed during development. These results demonstrate
feasibility in a controlled IID experiment; they do not establish gains across heterogeneous institutions.

## One-GPU notes

- `job.py` configures `SimEnv(num_threads=1)`, and `--gpu '[0]'` selects the physical GPU used by its worker.
- The workspace lock covers trainers launched by one job. Do not run separate Evo2 jobs concurrently on one GPU.
- A ten-round, three-site run initializes Evo2 30 times, so model loading can dominate a short demonstration.
- Lower `--micro-batch-size` if needed while keeping `--global-batch-size` divisible by it.
- Tensor and pipeline parallelism, multi-node execution, long-context tuning, and full-backbone federation are
  outside this example.

## Data source and terms

The repository does not bundle sequences. The benchmark was curated by InstaDeep from public genomics resources
described in the [Nucleotide Transformer paper](https://www.nature.com/articles/s41592-024-02523-z). The pinned
dataset card attributes the splice-site task to
[GENCODE V44 human gene annotations](https://www.gencodegenes.org/human/release_44.html) after excluding level 3
transcripts.

Before using the data, review the pinned
[dataset card and files](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised/tree/851f9946252e90c665cdb3cc3eedb78f1f26197c),
the [Nucleotide Transformer license](https://github.com/instadeepai/nucleotide-transformer/blob/main/LICENSE.md),
and the [GENCODE data-use terms](https://www.gencodegenes.org/pages/data_access.html). The historical loader links
to CC BY-NC-SA 4.0, while the pinned dataset card does not declare a standalone SPDX license for the sequence
content. Confirm the terms for this exact snapshot with the data owner. Evo2 weights and BioNeMo software have
separate terms presented by their respective sources.
