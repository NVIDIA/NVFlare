# Federated Evo2 Fine-Tuning for Splice-Site Classification

This example fine-tunes the `evo2/1b-8k-bf16:1.0` model for three-class splice-site classification with
BioNeMo Recipes, Megatron Bridge, and NVIDIA FLARE. Three simulated institutions train LoRA adapters and a
classification head. NVFlare distributes one common initialization and performs sample-weighted FedAvg over
only those trainable tensors. Floating trainables cross the federation boundary and remain in server checkpoints
as CPU float32 tensors. The Evo2 backbone, optimizer state, and Transformer Engine state stay local.

The default run uses one H100. `SimEnv(num_threads=1)` dispatches site tasks through one simulator worker, and a
workspace process lock keeps each fresh external trainer on the GPU until it exits before the next trainer in that
job starts. This is a compact simulator demonstration of the federated workflow; the three sites are partitions of
one public benchmark rather than real institutions.

**Start here:** [walkthrough.ipynb](./walkthrough.ipynb)

## Components and pinned inputs

| Component | Pinned version |
| --- | --- |
| PyTorch container | `nvcr.io/nvidia/pytorch:26.02-py3` |
| BioNeMo Recipes | [`101d88b7e0b0ee40e5af303575eb9661620a3706`](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/tree/101d88b7e0b0ee40e5af303575eb9661620a3706) |
| Megatron Bridge | [`fcbb6031103d0ca845c1a54d4fee55ecfcca17b6`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/fcbb6031103d0ca845c1a54d4fee55ecfcca17b6) (v0.5.0) |
| causal-conv1d | [`c51519fb10d92ffa257c1982e1831d3243454b2f`](https://github.com/Dao-AILab/causal-conv1d/tree/c51519fb10d92ffa257c1982e1831d3243454b2f) (v1.6.1), rebuilt against the container's PyTorch ABI |
| Evo2 checkpoint | `evo2/1b-8k-bf16:1.0`, converted once from NeMo 2 to Megatron Bridge format |
| Dataset | [`InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised/tree/851f9946252e90c665cdb3cc3eedb78f1f26197c), revision `851f9946252e90c665cdb3cc3eedb78f1f26197c` |
| Task | `splice_sites_all`: 30,000 train and 3,000 official test sequences, each 600 bases |

The image build verifies the full PEP 610 source commit for both `megatron-bridge` and `megatron-core` after the
upstream recipe installs them. Data preparation fails if the pinned task no longer has the documented split counts,
600-base sequences, or integer labels 0, 1, and 2.

The classifier, loss, pooling, and LoRA modules come from BioNeMo's pinned
[`evo2_classifier.py`](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/101d88b7e0b0ee40e5af303575eb9661620a3706/recipes/evo2_megatron/examples/evo2_classifier.py)
and its
[Evo2 LoRA tutorial](https://docs.nvidia.com/bionemo-recipes/latest/main/recipes/recipes/evo2_megatron/examples/lora-fine-tuning-tutorial/).
The default LoRA rank is 16, and the classification head predicts `no-splice`, `acceptor`, or `donor`.

## Accuracy and training exposure

The compact defaults demonstrate the complete federated path; they do not use the training exposure of the
BioNeMo tutorial's reported 0.9663 test accuracy. The retained executed notebook at BioNeMo Recipes commit
[`1b3f803dae2750ea339dd169cf5e1a16f1f6a02a`](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/1b3f803dae2750ea339dd169cf5e1a16f1f6a02a/bionemo-recipes/recipes/evo2_megatron/examples/lora-fine-tuning-tutorial.ipynb)
records eight workers, 1,000 optimizer steps, microbatch 32, and global batch 256. That run presented 256,000
training sequences. A one-GPU run with 1,000 steps and global batch 32 presents 32,000 sequences, one eighth of
that amount, so its accuracy is not a like-for-like reproduction.

Match sequence exposure when comparing federated and centralized runs, and report both optimizer steps and
`steps × global batch size`. With sequential simulated sites, the total exposure is
`rounds × sites × local steps × client global batch size`. By default, round boundaries change optimization because
each fresh client task resets its optimizer before loading the current global trainable weights. The optional
site-private training-state mode described below preserves optimizer dynamics across those process boundaries.

## Requirements

Use Linux with an NVIDIA GPU, a recent NVIDIA driver, Docker with NVIDIA Container Toolkit, and enough storage for
the image, source dataset cache, converted checkpoint, and run workspaces. The upstream tutorial documents the
model on a single 48 GB GPU; the default configuration here targets one H100. Actual memory use and runtime depend
on the driver, container, microbatch, and LoRA target modules.

From the repository root, build the pinned example image, then use the helper to enter it:

```bash
cd examples/advanced/bionemo/evo2
docker build -t nvflare-evo2 .
./start_evo2.sh
```

The helper exposes GPU 0, mounts this NVFlare checkout at `/workspace/nvflare`, persists Hugging Face and BioNeMo
caches under this example's `.evo2_cache`, and persists `/tmp/nvflare` outputs under this example's
`.evo2_workspace`. It installs the mounted checkout in editable mode and runs a CUDA forward/backward check of the
Evo2 causal-convolution extension before opening the shell. Override the defaults with `NVFLARE_EVO2_GPU`,
`NVFLARE_EVO2_IMAGE`, `NVFLARE_EVO2_CACHE`, or `NVFLARE_EVO2_WORKSPACE`.

The image supplies PyTorch, Transformer Engine, BioNeMo Evo2, and Megatron Bridge. If you use an equivalent
prebuilt environment, install the small example dependencies:

```bash
python3 -m pip install -r requirements.txt
```

On the NVFlare `main` branch, `nvflare[PT]>=2.10.0rc0,<2.11` may refer to an unreleased package. Install NVFlare
from this repository until that package is published:

```bash
python3 -m pip install -e /workspace/nvflare
```

All remaining commands run from `examples/advanced/bionemo/evo2` inside the container.
To use the walkthrough, start Jupyter from that shell and open the URL it prints:

```bash
jupyter lab --ip=0.0.0.0 --no-browser --allow-root --port=8888
```

## 1. Prepare the dataset

The preparation script downloads only the pinned `splice_sites_all` Parquet files. It keeps the official test
split, creates a deterministic 10% validation split from training data, checks exact-sequence and genomic-window
leakage, and partitions the remaining examples into simulated institutions.

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

Use `--showcase-size 0` for the full training split. To demonstrate heterogeneous label distributions, replace
`--partition iid` with `--partition dirichlet --dirichlet-alpha 0.5`. The command writes:

```text
data/
├── manifest.json
├── test.jsonl
├── validation.jsonl
└── train/
    ├── pooled.jsonl
    ├── site-1.jsonl
    ├── site-2.jsonl
    └── site-3.jsonl
```

`manifest.json` records the source revision, settings, per-site counts and label histograms, file paths,
SHA-256 content identities, and leakage-audit result. The JSONL rows retain `sequence`, genomic-coordinate `name`,
integer `label`, and `task`. Before a job launches, it verifies the row count, byte count, and hash of every pooled,
site, validation, and test JSONL. This prevents a passed leakage audit from being reused after an output file was
edited. The run summary records the verified identities and observed counts.

### Optional CPU orchestration smoke test

The mock backend exercises checkpoint serialization, two-client weighted aggregation for two rounds, external
process execution, run-summary collection, and evaluation plumbing without loading Evo2:

```bash
python3 prepare_initial_model.py \
  --backend mock \
  --output ./models/evo2_mock_init.pt

python3 job.py \
  --backend mock \
  --mode fedavg \
  --data-dir ./data \
  --initial-checkpoint ./models/evo2_mock_init.pt \
  --workspace /tmp/nvflare/evo2_splice_mock \
  --num-clients 2 \
  --num-rounds 2 \
  --local-steps 2 \
  --gpu '[0]' \
  --num-threads 1

python3 evaluate.py \
  --backend mock \
  --checkpoint /tmp/nvflare/evo2_splice_mock/evo2-splice-fedavg-lora/server/simulate_job/app_server/FL_global_model.pt \
  --test-file ./data/test.jsonl \
  --manifest ./data/manifest.json \
  --split-role test \
  --output ./results/mock/evaluation.json
```

The mock metrics do not measure model quality. The scoped unit tests assert exact weighted aggregation, strict
payload validation, backbone exclusion, and checkpoint round trips:

```bash
python3 -m pytest \
  ../../../../tests/unit_test/examples/advanced/bionemo/evo2_adapter_checkpoint_test.py \
  ../../../../tests/unit_test/examples/advanced/bionemo/evo2_aggregator_test.py \
  ../../../../tests/unit_test/examples/advanced/bionemo/evo2_client_mock_test.py \
  ../../../../tests/unit_test/examples/advanced/bionemo/evo2_prepare_base_checkpoint_test.py \
  ../../../../tests/unit_test/examples/advanced/bionemo/evo2_prepare_data_test.py \
  ../../../../tests/unit_test/examples/advanced/bionemo/evo2_job_test.py \
  ../../../../tests/unit_test/examples/advanced/bionemo/evo2_persistor_test.py \
  ../../../../tests/unit_test/examples/advanced/bionemo/evo2_runtime_test.py \
  ../../../../tests/unit_test/examples/advanced/bionemo/evo2_evaluate_test.py \
  ../../../../tests/unit_test/examples/advanced/bionemo/evo2_sequential_launcher_test.py \
  ../../../../tests/unit_test/examples/advanced/bionemo/evo2_summarize_baselines_test.py
```

Continue with the BioNeMo backend for the GPU acceptance run.

## 2. Download and convert Evo2-1B

Download `evo2/1b-8k-bf16:1.0` with BioNeMo and convert it once to the Megatron Bridge format expected by the
classifier. The command converts into a temporary directory, validates the completion markers, and only then
publishes the requested output path:

```bash
python3 prepare_base_checkpoint.py --output ./models/evo2_1b_bf16_mbridge
```

Every simulated site must use this same converted checkpoint and tokenizer configuration. Successful preparation
writes `nvflare_conversion_provenance.json`, which records the pinned conversion settings, tokenizer identity, and
a content inventory. Remove and regenerate an existing converted directory that predates this marker; the command
rejects unmarked, partial, modified, or incompatible output instead of reusing it.

## 3. Create the common trainable initialization

Instantiate the frozen backbone, LoRA modules, and classification head once, then export the trainable tensors in
NVFlare checkpoint format:

```bash
torchrun --standalone --nproc_per_node=1 prepare_initial_model.py \
  --base-checkpoint ./models/evo2_1b_bf16_mbridge \
  --data-file ./data/train/pooled.jsonl \
  --output ./models/evo2_lora_init.pt \
  --work-dir ./initialization \
  --peft-mode lora \
  --lora-dim 16 \
  --lora-alpha 32
```

The server loads this small checkpoint and sends identical adapter and head values to every site. BF16 model
parameters are promoted to the canonical float32 exchange dtype when this initialization is created. Tensor names,
shapes, dtypes, and finite values are validated before training and aggregation, and the checkpoint metadata binds
the initialization to `exchange_dtype=float32`.

Regenerate trainable initialization checkpoints created by an earlier version of this example. Checkpoints that
contain BF16 exchange tensors or omit the `exchange_dtype=float32` initialization metadata are rejected. Float32
doubles the adapter/head raw tensor value and checkpoint bytes relative to the earlier BF16 exchange while leaving
the frozen backbone and GPU model dtype unchanged.

Evaluate this initialization with the command in step 5, using
`--checkpoint ./models/evo2_lora_init.pt --output ./results/initialization/evaluation.json` and omitting the three
comparison flags. This provides the before-training accuracy and macro-F1 needed to verify that fine-tuning
improves the model.

### Staged GPU acceptance

Before the default three-site run, reproduce one local Evo2 training task:

```bash
python3 job.py \
  --mode local \
  --site-index 1 \
  --data-dir ./data \
  --initial-checkpoint ./models/evo2_lora_init.pt \
  --base-checkpoint ./models/evo2_1b_bf16_mbridge \
  --workspace /tmp/nvflare/evo2_splice_single_site \
  --num-rounds 1 \
  --local-steps 20 \
  --gpu '[0]' \
  --num-threads 1
```

Evaluate the printed global checkpoint. Once it reloads successfully and reports finite metrics, run the
two-site, two-round acceptance case:

```bash
python3 job.py \
  --mode fedavg \
  --data-dir ./data \
  --initial-checkpoint ./models/evo2_lora_init.pt \
  --base-checkpoint ./models/evo2_1b_bf16_mbridge \
  --workspace /tmp/nvflare/evo2_splice_two_by_two \
  --num-clients 2 \
  --num-rounds 2 \
  --local-steps 20 \
  --gpu '[0]' \
  --num-threads 1
```

After that job completes, check that all four task metrics exist and that both adapter and head values changed:

```bash
python3 - <<'PY'
import json
from pathlib import Path

import torch

from evo2_adapter_checkpoint import load_nvflare_checkpoint

initial = load_nvflare_checkpoint("models/evo2_lora_init.pt")
summary = json.loads(Path("/tmp/nvflare/evo2_splice_two_by_two/run_summary.json").read_text())
trained = load_nvflare_checkpoint(summary["global_checkpoint"])
changed = {name for name in initial if not torch.equal(initial[name], trained[name])}

assert len(summary["round_metrics"]) == 4
assert any("classification_head" in name for name in changed)
assert any("classification_head" not in name for name in changed)
print(f"changed tensors: {len(changed)} / {len(initial)}")
print(f"peak GPU memory: {summary['peak_client_gpu_memory_mebibytes']:.1f} MiB")
print(
    "raw trainable tensor payload: "
    f"{summary['total_received_mebibytes'] + summary['total_sent_mebibytes']:.1f} MiB"
)
PY
```

The exchange helpers reject any backbone key, and their unit tests confirm strict loading leaves frozen backbone
values unchanged. A successful `job.py` return and fresh four-entry summary show that every external task
finished before the simulator released the run.

## 4. Run federated LoRA training

The following command runs three sites for ten rounds. Each client performs 20 local optimizer steps with a
microbatch of four and gradient accumulation to global batch 32:

```bash
python3 job.py \
  --mode fedavg \
  --data-dir ./data \
  --initial-checkpoint ./models/evo2_lora_init.pt \
  --base-checkpoint ./models/evo2_1b_bf16_mbridge \
  --workspace /tmp/nvflare/evo2_splice_fedavg \
  --num-clients 3 \
  --num-rounds 10 \
  --local-steps 20 \
  --micro-batch-size 4 \
  --global-batch-size 32 \
  --peft-mode lora \
  --lora-dim 16 \
  --gpu '[0]' \
  --num-threads 1
```

By default, each external process rebuilds Evo2 from the unchanged base checkpoint, loads the round's global
float32 adapter and head after model initialization, casts them at the BF16 model boundary, reloads the optimizer
master parameters, trains, and returns a float32 `DIFF`. The client measures that update against the rounded BF16
baseline and applies it to the original float32 global state, preserving server residuals that BF16 cannot
represent when a local round makes no update. Optimizer and Transformer Engine bookkeeping are never aggregated.
Each attempt has its own directory and its own float32 `local_trainable_model.pt`. A workspace-level POSIX lock is
acquired before the inner Python or `torchrun` trainer starts, and its file descriptor remains open until that
trainer exits. This serializes GPU use across all simulated sites even if their lightweight launcher processes
overlap. Before the data loaders are built, the exchange callback advances Megatron's cyclic-sampler cursor by
`round × local_steps × global_batch_size` samples. The pinned cyclic sampler derives each epoch's permutation from
that epoch's state; the configured seed still binds common initialization and the training configuration. Fresh
processes therefore select the same per-round data slices as one uninterrupted deterministic sample stream. The
optimizer, scheduler, and runtime RNG state still restart in every stateless task. Each task also verifies a SHA-256
fingerprint of every frozen backbone parameter before and after local training, and its metrics record the round's
starting sample offset. The server writes both the latest global checkpoint and a round-numbered checkpoint after
every aggregation, so evaluation can reload any round without relying on a file that later rounds overwrite.

### Site-private training state and stateless continuation

Add `--persist-client-training-state` to preserve each site's native Megatron Bridge optimizer, scheduler, RNG,
and sampler state across fresh external processes. The state is stored under
`<workspace>/client_training_state/<site>/round_<round>` and remains private to that site; NVFlare still exchanges
and aggregates only the LoRA and head tensors. A resumed task first restores its local native state, then applies
the current global trainable tensors and reloads the optimizer's master parameters before training. Manifests bind
each state directory to its site, round, configuration, trainable schema, incoming global model, local result, and
predecessor manifest. The configuration records SHA-256 content identities for the base checkpoint, classifier,
training JSONL, and validation JSONL. Resumption fails if any of those inputs changes, including a same-path edit.
Native checkpoint payloads can be much larger than the federated trainable checkpoint, so plan for the workspace to
grow with every site and round.

A site commits its private state before it returns that round's update. If a later site or the server fails before
the round is aggregated, the completed sites can therefore contain private `round_<round>` state while no matching
global aggregate exists. Preserve that workspace as failure evidence and restart persistent-state training from
round zero in a fresh workspace; do not reuse the partial state chain. `--start-round` cannot recover a persisted
run because a global trainable checkpoint does not contain the site-private optimizer, scheduler, RNG, or sampler
state.

Persistent training state must begin at round zero so every site has a complete state chain. It is opt-in because
retaining site-local optimizer moments changes the optimization method from the default fresh-optimizer FedAvg.
For a persisted run, add this flag to the command above:

```bash
--persist-client-training-state
```

`--start-round N` serves a different purpose: it continues a stateless FedAvg job from a saved global checkpoint
whose metadata identifies round `N - 1` and carries the same continuation signature. That signature binds the mode,
site names, sample weights, audited training identities, dataset source and partition settings, validation identity,
the seed, local-step, microbatch, and global-batch sampler budget, and the learning-rate schedule. Changing any of
those values rejects continuation before launch. A valid continuation preserves the original common initialization
metadata while numbering new global checkpoints from `N`. Do not combine `--start-round` with
`--persist-client-training-state`; a new process cannot reconstruct the missing site-private state chain from a
global trainable checkpoint.

### BioNeMo accuracy-matching configuration

The following three-site configuration matches the BioNeMo tutorial's 256,000-presentation budget within one of
the tutorial's 256-sequence global batches: `37 rounds × 3 sites × 24 steps × global batch 96 = 255,744`
presentations. Prepare the full IID training split in step 1 with `--showcase-size 0`, then run:

```bash
python3 job.py \
  --backend bionemo \
  --mode fedavg \
  --data-dir ./data \
  --initial-checkpoint ./models/evo2_lora_init.pt \
  --base-checkpoint ./models/evo2_1b_bf16_mbridge \
  --workspace /tmp/nvflare/evo2_splice_fedavg_37x24_gbs96 \
  --num-clients 3 \
  --start-round 0 \
  --num-rounds 37 \
  --local-steps 24 \
  --seq-length 600 \
  --micro-batch-size 32 \
  --global-batch-size 96 \
  --learning-rate 0.0005 \
  --min-learning-rate 0.00005 \
  --warmup-iters 30 \
  --eval-iters 1 \
  --seed 1234 \
  --peft-mode lora \
  --lora-dim 16 \
  --lora-alpha 32 \
  --lora-dropout 0.1 \
  --persist-client-training-state \
  --gpu '[0]' \
  --num-threads 1
```

This is the predeclared accuracy-matching experiment, rather than the compact demonstration configuration. The
acceptance thresholds below target the reported 0.9663 accuracy and 0.966 macro-F1 within one percentage point,
while requiring recall and F1 of at least 0.94 for every class.

## 5. Evaluate the saved global checkpoint

At completion, `job.py` prints `Result can be found in:`, `Global checkpoint:`, and `Run summary:` paths.
Pass the exact `Global checkpoint:` value to evaluation. For the compact federated command at the start of step 4,
the checkpoint is
`/tmp/nvflare/evo2_splice_fedavg/evo2-splice-fedavg-lora/server/simulate_job/app_server/FL_global_model.pt`:

```bash
torchrun --standalone --nproc_per_node=1 evaluate.py \
  --checkpoint /tmp/nvflare/evo2_splice_fedavg/evo2-splice-fedavg-lora/server/simulate_job/app_server/FL_global_model.pt \
  --base-checkpoint ./models/evo2_1b_bf16_mbridge \
  --test-file ./data/test.jsonl \
  --manifest ./data/manifest.json \
  --split-role test \
  --output ./results/fedavg_lora/evaluation.json \
  --confusion-matrix ./results/fedavg_lora/confusion_matrix.png \
  --reference-report ./results/initialization/evaluation.json \
  --require-improvement \
  --improvement-metric macro_f1 \
  --work-dir /tmp/nvflare/evo2_splice_fedavg_evaluation \
  --peft-mode lora \
  --lora-dim 16 \
  --lora-alpha 32
```

The accuracy-matching protocol evaluates every raw checkpoint on validation, then limits selection to raw rounds 27
through 36 and one uniform parameter average of those ten rounds. It ranks those 11 candidates by validation
macro-F1, then validation accuracy, with the earliest raw round as the raw-checkpoint tie breaker. It performs one
official-test evaluation only after selecting and freezing the candidate. Set `SELECTED_CHECKPOINT` to that exact
validation-selected raw or averaged checkpoint; do not automatically use the final-round checkpoint:

```bash
: "${SELECTED_CHECKPOINT:?Set this to the validation-selected checkpoint}"

torchrun --standalone --nproc_per_node=1 evaluate.py \
  --checkpoint "${SELECTED_CHECKPOINT}" \
  --base-checkpoint ./models/evo2_1b_bf16_mbridge \
  --test-file ./data/test.jsonl \
  --manifest ./data/manifest.json \
  --split-role test \
  --output ./results/fedavg_lora_37x24_gbs96/evaluation.json \
  --confusion-matrix ./results/fedavg_lora_37x24_gbs96/confusion_matrix.png \
  --reference-report ./results/initialization/evaluation.json \
  --require-improvement \
  --improvement-metric macro_f1 \
  --min-accuracy 0.9563 \
  --min-macro-f1 0.956 \
  --min-class-recall 0.94 \
  --min-class-f1 0.94 \
  --work-dir /tmp/nvflare/evo2_splice_fedavg_37x24_gbs96_evaluation \
  --peft-mode lora \
  --lora-dim 16 \
  --lora-alpha 32
```

The command writes the complete gate assessment to `performance_gate` in the metrics JSON before returning a
nonzero exit status if any criterion fails. Record the selected checkpoint path and SHA-256 digest alongside the
validation decision so the official-test result cannot be attributed to another round.

Evaluation requires paired `--manifest` and `--split-role` arguments by default. Use
`--allow-unbound-evaluation` only when intentionally evaluating a JSONL file outside the prepared-data manifest;
that escape hatch cannot be combined with either manifest-binding argument.

Evaluation writes a machine-readable metrics JSON file and a confusion-matrix image. With the comparison flags
above, it also records accuracy and macro-F1 deltas and exits with an error if macro-F1 did not improve over the
reloaded initialization. The four optional minimum-threshold flags check aggregate accuracy, aggregate macro-F1,
and recall and F1 for every class. They are disabled unless explicitly supplied. It hashes the held-out JSONL
file, converted base checkpoint, and classifier source,
records the evaluated trainable-checkpoint hash and normalized evaluation settings, and rejects a reference report
produced with different inputs or settings. `--manifest` and `--split-role` bind evaluation to the selected audited
validation or official-test split: the resolved path, row count, byte count, and SHA-256 digest must match the
prepared-data manifest. Training and evaluation also reject a trainable checkpoint whose saved
backend, PEFT topology, seed, base-checkpoint hash, or classifier-source hash differs from the requested run. The
report includes test accuracy, macro-F1, per-class
counts, the confusion matrix, runtime, peak GPU memory, and trainable checkpoint size. The job's `run_summary.json`
contains aggregation weights, every task's metrics, aggregate client runtime, maximum client GPU memory, and raw
trainable tensor-value MiB sent and received. This payload accounting excludes serialization, protocol, and
transport overhead. The summary also records SHA-256 identities for the initial and global trainable checkpoints,
every training/validation JSONL, the converted base-checkpoint tree, and the resolved classifier source, plus
dataset provenance and the complete training configuration. Per-task metrics also include validation loss/accuracy
when supplied by BioNeMo, runtime, sample count, raw trainable tensor payload sizes, frozen-backbone verification,
and the path and SHA-256 identity of that task's separately validated trainable checkpoint.

The pinned classifier uses Megatron's cyclic sampler, which drops an incomplete microbatch before beginning the
next epoch. `evaluate.py` therefore requires the test row count to be divisible by `--micro-batch-size`. During
evaluation it wraps the pinned dataset with a zero-based JSONL row index that stays on CPU and is never passed to
Evo2. After trimming the sampler's final partial epoch, it requires every source row index exactly once and verifies
the observed label against that row. The metrics report records the expected, observed, and unique row counts plus a
SHA-256 digest of sampler order. The 3,000-row splice-site test split is divisible by the default microbatch of four.

| Output | Default location for the compact federated command |
| --- | --- |
| Result directory printed by `job.py` | `/tmp/nvflare/evo2_splice_fedavg/evo2-splice-fedavg-lora` |
| Global checkpoint printed by `job.py` | `<result-dir>/server/simulate_job/app_server/FL_global_model.pt` |
| Per-round global checkpoints | `<result-dir>/server/simulate_job/app_server/FL_global_model_round_<round>.pt` |
| Aggregate run summary printed by `job.py` | `/tmp/nvflare/evo2_splice_fedavg/run_summary.json` |
| Per-site, per-round metrics and trainable checkpoints | `/tmp/nvflare/evo2_splice_fedavg/client_work/<site>/` |
| Evaluation metrics | `./results/fedavg_lora/evaluation.json` |
| Confusion matrix | `./results/fedavg_lora/confusion_matrix.png` |

### H100 precursor results

The following measurements came from a precursor source snapshot with the same pinned BioNeMo environment,
model, dataset, and `37 rounds × 24 local steps` training protocol. That snapshot predates this example's port to
the current NVFlare Recipe API and the final metadata and atomic-persistence integrity checks, so these values have
not yet been reproduced from the exact source in this PR. They provide a reference for expected behavior, not a
guarantee for a new run.

| Fixed endpoint | Validation accuracy | Validation macro-F1 | Test accuracy | Test macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Raw federated round 36 | 0.9607 | 0.9606 | 0.9533 | 0.9535 |
| Federated mean, rounds 27–36 | 0.9617 | 0.9617 | 0.9560 | 0.9562 |
| Local site 1 | 0.9163 | 0.9165 | 0.9080 | 0.9083 |
| Local site 2 | 0.9110 | 0.9117 | 0.9010 | 0.9023 |
| Local site 3 | 0.8680 | 0.8690 | 0.8497 | 0.8514 |
| Local equal-site mean | 0.8984 | 0.8991 | 0.8862 | 0.8873 |

On the official test split, raw federated round 36 exceeded the validation-selected best local model (site 1) by
4.53 accuracy points and 4.52 macro-F1 points. It exceeded the descriptive local mean by 6.71 accuracy points and
6.62 macro-F1 points. The secondary federated mean was 1.03 accuracy points below the BioNeMo tutorial's reported
0.9663 result; raw round 36 was 1.30 points below it. The equal-site local mean is a descriptive summary of three
models, not an ensemble or a pooled-data result.

All endpoints used the same manifest-bound evaluator with evaluation microbatch 8 and global batch 32. The
official test split had already been observed in earlier engineering runs before this fixed-endpoint comparison,
so these results are reproducibility evidence rather than an untouched-test confirmation. A new run should treat
its generated manifests, checkpoint hashes, and evaluation reports as the authoritative evidence.

## Baseline comparisons

Use the full prepared split, held-out test data, seed, base checkpoint, LoRA configuration, learning-rate schedule,
microbatch 32, and global batch 96 for all comparisons. Give each run a distinct workspace and results directory.
The walkthrough derives checkpoint identities from the current run and verifies that every local job and the
federated job use the same trainable initialization.

The primary local-only comparison uses fixed training endpoints. The federated endpoint is raw round 36, the last
zero-based round of the 37-round run. Each local endpoint is the final model from one uninterrupted 888-step task.
This gives each local model `1 × 888 × 96 = 85,248` sequence presentations, exactly the presentations contributed
by one site during federation (`37 × 24 × 96`). The three local runs together use 2,664 optimizer steps and 255,744
presentations, matching the federated system total. A local run does not use `--persist-client-training-state`:
with only one training task, its optimizer, scheduler, RNG, and sampler already remain continuous for all 888
steps, and there is no later process in which to restore them.

For these one-contributor rounds, the aggregator performs the same schema, shape, provenance, metric, and round
validation as FedAvg, then clones the validated client difference directly. It does not multiply and divide the
sole FP32 update by its 9,000-example sample weight, so the aggregated difference stays bitwise identical to the
client difference.

Local-only LoRA for each simulated institution:

```bash
for site in 1 2 3; do
  python3 job.py \
    --mode local --site-index "${site}" \
    --data-dir ./data \
    --initial-checkpoint ./models/evo2_lora_init.pt \
    --base-checkpoint ./models/evo2_1b_bf16_mbridge \
    --workspace "/tmp/nvflare/evo2_splice_local_${site}" \
    --require-fresh-workspace \
    --backend bionemo --start-round 0 \
    --num-rounds 1 --local-steps 888 \
    --seq-length 600 \
    --micro-batch-size 32 --global-batch-size 96 \
    --learning-rate 0.0005 --min-learning-rate 0.00005 \
    --warmup-iters 30 --eval-iters 1 --seed 1234 \
    --peft-mode lora --lora-dim 16 --lora-alpha 32 --lora-dropout 0.1 \
    --gpu '[0]' --num-threads 1
done
```

`--require-fresh-workspace` fails before recipe construction when a selected workspace already exists, preventing a
prior local run from being mistaken for the current endpoint. Use a distinct path for each campaign.

`--mode local --site-index N` passes only `data/train/site-N.jsonl` to that trainer. The common preflight still
verifies the row count, byte count, and SHA-256 digest of every JSONL named by the full prepared-data manifest.
Those shared hashes prove that all four runs use the same audited partition snapshot; they do not grant a local
trainer access to pooled training data or another site's training file. Retain the manifest, each run summary, and
the per-task training-file identity to demonstrate both common provenance and training-data isolation.

Freeze the raw-round-36 federated checkpoint and all three local final checkpoints, including their SHA-256
digests, before running the primary official-test comparison. Evaluate those four fixed endpoints with one frozen
evaluator configuration and report every local result. Report the arithmetic mean, population standard deviation,
and minimum-to-maximum spread across the three sites for accuracy, macro-F1, and the minimum per-class recall and
F1.
The same 3,000 test sequences are reused for each model, so this spread is descriptive; do not combine the three
confusion matrices and describe them as 9,000 independent test examples.

Keep the validation-selected uniform parameter mean of federated rounds 27 through 36 as a secondary result. It
is not the raw-round-36 primary endpoint. As a second local summary, rank the three fixed local final checkpoints
on validation macro-F1, then validation accuracy, then the lowest numeric site index, and identify that checkpoint
as the validation-selected best local model. Test metrics never participate in that tie-break. Report the selected
local result alongside all three site results and the all-site mean and spread; do not replace the all-site results
with the selected model.

The comparison workflow uses evaluation microbatch 8 and global batch 32 for every validation and test endpoint:
raw FL round 36, each local final, validation-based local ranking, and the secondary federated average. Portable
commands may use a different microbatch that divides every evaluated split exactly, but the initialization and all
comparison models must share the same microbatch and global batch so their evaluation signatures remain comparable.

Pooled-data LoRA is an upper-reference run with centralized access to all simulated training data. Its 111 rounds
match the federated run's 2,664 total optimizer steps (`3 sites × 37 rounds × 24 steps`), 111 fresh process
launches, and 255,744 sequence presentations:

```bash
python3 job.py \
  --mode pooled \
  --data-dir ./data \
  --initial-checkpoint ./models/evo2_lora_init.pt \
  --base-checkpoint ./models/evo2_1b_bf16_mbridge \
  --workspace /tmp/nvflare/evo2_splice_pooled \
  --backend bionemo --start-round 0 \
  --num-rounds 111 --local-steps 24 \
  --seq-length 600 \
  --micro-batch-size 32 --global-batch-size 96 \
  --learning-rate 0.0005 --min-learning-rate 0.00005 \
  --warmup-iters 30 --eval-iters 1 --seed 1234 \
  --peft-mode lora --lora-dim 16 --lora-alpha 32 --lora-dropout 0.1 \
  --persist-client-training-state \
  --gpu '[0]' --num-threads 1
```

Federated head-only training requires a matching head-only initialization:

```bash
torchrun --standalone --nproc_per_node=1 prepare_initial_model.py \
  --base-checkpoint ./models/evo2_1b_bf16_mbridge \
  --data-file ./data/train/pooled.jsonl \
  --output ./models/evo2_head_init.pt \
  --work-dir ./head_initialization \
  --peft-mode head-only

python3 job.py \
  --backend bionemo --mode fedavg --peft-mode head-only \
  --data-dir ./data \
  --initial-checkpoint ./models/evo2_head_init.pt \
  --base-checkpoint ./models/evo2_1b_bf16_mbridge \
  --workspace /tmp/nvflare/evo2_splice_head_only \
  --num-clients 3 --start-round 0 --num-rounds 37 --local-steps 24 \
  --seq-length 600 \
  --micro-batch-size 32 --global-batch-size 96 \
  --learning-rate 0.0005 --min-learning-rate 0.00005 \
  --warmup-iters 30 --eval-iters 1 --seed 1234 \
  --persist-client-training-state \
  --gpu '[0]' --num-threads 1
```

Evaluate both the head-only initialization and final global checkpoint with `--peft-mode head-only`; the step-5
evaluation command otherwise defaults to the LoRA model topology.

This example requires reproducible end-to-end execution and improvement over the common initialization.
Federated LoRA outperforming local-only LoRA, pooled LoRA, or federated head-only training is an experimental
result to measure rather than an assumed property. The one-task local-only endpoint intentionally omits
`--persist-client-training-state`; pooled and federated baselines with multiple fresh tasks must otherwise use the
same persistent or stateless policy as the federated run they compare against.

## One-GPU execution notes

- Keep `--num-threads 1` and `--gpu '[0]'`; the job rejects parallel simulator threads.
- The bundled `sequential_launcher.py` holds `<workspace>/.evo2_training.lock` across each inner trainer's complete
  lifetime. It serializes trainers belonging to that workspace; it does not arbitrate separate jobs that use
  different workspaces. Run the federated, pooled, local-only, and head-only jobs sequentially when they share one
  GPU. The `for site in 1 2 3` loop above is sequential; do not put its three local jobs in the background on a
  one-H100 host.
- `--gpu` is the simulator worker's device assignment. On a host or container where several physical GPUs are
  visible, `--gpu '[2]'` assigns physical GPU 2; an outer `CUDA_VISIBLE_DEVICES=2` does not make `--gpu '[0]'`
  relative to that selection because the simulator worker sets the child process environment from `--gpu`.
- A ten-round, three-site run launches and initializes Evo2 30 times. Startup and base-checkpoint loading may be a
  large part of total runtime even though only LoRA and head tensors cross the federation boundary.
- Start with `--num-rounds 2 --local-steps 2 --num-clients 2` to validate the environment. The mock backend can
  test orchestration and aggregation without BioNeMo or a GPU by adding `--backend mock` to initialization and job
  commands.
- If memory is tight, lower `--micro-batch-size` while keeping `--global-batch-size` divisible by it. Measure peak
  memory in your environment before increasing either value.
- Tensor parallelism, pipeline parallelism, multi-node training, long-context fine-tuning, and full-backbone
  federation are outside this example.

## Data source and use

The repository does not bundle sequences. `prepare_data.py` downloads the two pinned splice-site Parquet files
from the original Hugging Face repository and records their revision in `manifest.json`. The dataset rows retain
public genomic coordinates so split provenance and overlap can be audited.

The benchmark was curated by InstaDeep from public genomics resources described in the
[Nucleotide Transformer paper](https://www.nature.com/articles/s41592-024-02523-z). The pinned dataset card says
this splice-site task comes from [GENCODE V44 human gene annotations](https://www.gencodegenes.org/human/release_44.html)
after excluding level 3 transcripts. Review the pinned
[dataset card and files](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised/tree/851f9946252e90c665cdb3cc3eedb78f1f26197c),
the [Nucleotide Transformer license](https://github.com/instadeepai/nucleotide-transformer/blob/main/LICENSE.md),
and the [GENCODE data-use terms](https://www.gencodegenes.org/pages/data_access.html) before using the data. The
historical dataset loader points to that Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license, while
the pinned dataset card does not declare a standalone SPDX license for the sequence content. Confirm the terms for
this exact snapshot with the data owner, and do not infer data-use rights from the Apache-2.0 license on this
example's code.

The Evo2 checkpoint and BioNeMo software have their own terms. Review the notices presented by BioNeMo's
`load()` command and the pinned BioNeMo Recipes repository before downloading or using model weights.

## File map

| File | Purpose |
| --- | --- |
| `Dockerfile`, `start_evo2.sh`, `verify_environment.py` | Build, validate, and enter the pinned H100 environment |
| `requirements.txt` | Lightweight data, evaluation, notebook, and NVFlare dependencies |
| `prepare_data.py` | Download, validate, split, partition, and describe the public dataset |
| `prepare_base_checkpoint.py` | Atomically download, convert, and validate the pinned Evo2 base checkpoint |
| `prepare_initial_model.py` | Export the common LoRA/head or head-only initialization |
| `job.py` | Configure FedAvg or local/pooled baseline execution in `SimEnv` |
| `evo2_aggregator.py` | Reject incompatible client updates and apply sample-weighted FedAvg |
| `evo2_persistor.py` | Keep the server's trainable-only global state on CPU across NVFlare versions |
| `client.py` | Run one external BioNeMo training task and return trainable-tensor differences |
| `evo2_runtime.py` | Adapt the pinned BioNeMo classifier and Megatron Bridge callbacks |
| `sequential_launcher.py` | Hold the shared trainer lock across each external process lifetime |
| `evo2_adapter_checkpoint.py` | Validate, serialize, load, and diff trainable tensors |
| `provenance.py` | Hash inputs and create self-verifying stateless-continuation signatures |
| `evaluate.py` | Reload a saved model and evaluate the official test split |
| `summarize_baselines.py` | Validate and summarize the fixed federated and three-site local comparison |
| `walkthrough.ipynb` | Execute and inspect the workflow interactively |
