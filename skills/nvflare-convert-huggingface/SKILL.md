---
name: nvflare-convert-huggingface
description: "Convert a Hugging Face Trainer, Seq2SeqTrainer, or supported TRL trainer into an NVFLARE federated job; do not use for Lightning, manual PyTorch, inference-only code, deployment, or failed-job diagnosis."
license: Apache-2.0
metadata:
  author: "NVIDIA FLARE Team <federatedlearning@nvidia.com>"
  version: "0.1.0"
  min_flare_version: "2.9.0"
  blast_radius: runs_simulator
  category: Conversion
  tags: "nvflare, federated-learning, hugging-face, trl, conversion"
  languages: "python"
  frameworks: "hugging-face, pytorch, nvflare"
  domain: ml
---

# NVFLARE Convert Hugging Face

## Use When

Use when a Hugging Face `Trainer`, `Seq2SeqTrainer`, or supported TRL trainer
owns training.

## Do Not Use When

Route Lightning to `nvflare-convert-lightning`, manual loops to
`nvflare-convert-pytorch`, unclear ownership to `nvflare-orient`, and failed
jobs to `nvflare-diagnose-job`. Inference-only code, deployment, FSDP,
DeepSpeed, and unsupported auxiliary-model RL trainers are out of scope.

## Workflow

1. Read the active training entry point and apply
   `../nvflare-shared/references/framework-routing.md`. Continue only when a
   supported Hugging Face or TRL trainer owns training.
2. Extract the model and trainer construction, tokenizer/processor, datasets,
   `TrainingArguments` or TRL config, callbacks, `compute_metrics`, checkpoints,
   launcher, site/round counts, and data-split policy. Preserve these unless the
   public API rejects an unsupported combination.
3. Use the PyTorch recipe family. For ordinary FedAvg, inspect
   `nvflare recipe show fedavg-pt --format json` and construct the returned
   public recipe. For another requested algorithm, use
   `../nvflare-shared/references/pytorch-family-recipe-selection.md`. After
   `recipe show`, apply
   `../nvflare-shared/references/pytorch-family-recipe-construction.md`; pass
   only exposed recipe parameters.
4. Keep the trainer as the local training owner. Initialize
   `nvflare.client.hf`, construct the trainer once, call `flare.patch(trainer)`
   once, then call `trainer.evaluate()` before `trainer.train()` on every train
   task; reversing them drops server-selection metrics. Do not add manual
   `FLModel` receive/load/send code.
5. Keep patch defaults unless source requirements need an option.
   `params_scope="auto"` exchanges full-model weights normally and adapter
   weights for PEFT. Preserve checkpoint state.
6. Configure the server model with an importable `class_path` and explicit
   constructor `args`, never a live model. Ensure its exchanged keyspace and
   tensor shapes match the trainer model or PEFT adapter.
7. Preserve exact trainer metric names. Select a finite, higher-is-better key
   actually returned by `trainer.evaluate()`. For a lower-is-better metric, emit
   and select an explicitly negated companion.
8. Apply `../nvflare-shared/references/conversion-workflow.md`, including scoped
   generation, site-local data, exact argument parsing, bounded simulation,
   export inspection, and evidence reporting.

## Requirements

- For `torchrun` or another distributed launcher, initialize distributed state
  before patching and make every rank call the same trainer methods in the same
  order. Do not silently replace the source process model.
- Discover recipe capabilities through public CLI output. Do not inspect
  NVFLARE implementation source or invent recipe, export, or simulator APIs.

Load `../nvflare-shared/references/dependency-install.md` only when dependencies
must be installed and
`../nvflare-shared/references/validation-evidence.md` before validation. Do not
load every reference preemptively.
