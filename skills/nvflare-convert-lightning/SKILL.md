---
name: nvflare-convert-lightning
description: "Convert existing PyTorch Lightning training code into an NVFLARE federated job using the Lightning Client API patch, local validation, and job export; do not use for plain PyTorch, other frameworks, or deployment-only tasks."
min_flare_version: "2.8.0"
blast_radius: edits_files
category: Conversion
skill_version: "0.1.0"
---

# NVFLARE Convert PyTorch Lightning

## Use When

Use when the user asks to convert PyTorch Lightning code into an NVFLARE
federated training job: a `LightningModule`, `LightningDataModule`, a `Trainer`
fit/validate/test loop, Lightning callbacks, checkpointing, loggers, or a
Lightning DDP/multi-GPU job.

## Do Not Use When

Do not use for plain `torch.nn.Module` manual training loops without Lightning
(route to `nvflare-convert-pytorch`), Hugging Face Trainer, TensorFlow, XGBoost,
scikit-learn, an already exported FLARE job (route to `nvflare-job-lifecycle`),
a failed existing job (route to `nvflare-diagnose-job`), Kubernetes deployment,
production submission, or generic Lightning debugging that does not ask for
FLARE conversion.

## Workflow

1. Before Python import/inspect commands, load
   `../_shared/dependency-install.md` and install applicable source
   `requirements*.txt` files in the active `nvflare` environment.
2. Inspect before editing with `nvflare agent inspect <path> --format json` to
   confirm Lightning evidence and conversion state. Use
   `references/lightning-detection.md` to confirm Lightning versus plain PyTorch
   and to hand off to `nvflare-convert-pytorch` when no Lightning evidence
   exists.
3. Follow the shared conversion workflow contract in
   `../_shared/nvflare-job-lifecycle.md`. Use
   `../_shared/runtime-output-guidance.md` when choosing generated output,
   export, and simulator workspace locations.
4. Identify the existing `LightningModule`, `LightningDataModule`, trainer
   construction, callbacks, checkpointing, metrics, and logger usage. Determine
   the concrete model constructor values that server and client models must
   share before creating `job.py`.
5. Reuse PyTorch recipe discovery. Lightning is a PyTorch-family training
   framework, not a separate recipe family: run
   `nvflare recipe list --framework pytorch --format json` and select the recipe
   from the requested FL workflow, not from Lightning alone. Use FedAvg for
   standard horizontal training and FedEval for evaluation-only.
6. Convert the training entry point to the Lightning Client API: build the
   `Trainer`, call `flare.patch(trainer)`, and let the patched trainer own model
   load/send through its callbacks. Follow `references/lightning-conversion.md`.
7. Add or update a `job.py` that uses the selected PyTorch recipe or job API
   path. Follow the shared lifecycle for generated layout and export behavior.
8. Validate and export through the shared lifecycle. Load
   `../_shared/validation-evidence.md` and
   `../_shared/metrics-and-artifact-reporting.md` for generic evidence
   reporting, then use `references/lightning-validation.md` for
   Lightning-specific checks before calling the conversion complete.

## Requirements

- Must integrate through `flare.patch(trainer)` and let the patched trainer own
  model exchange. Must not generate a manual `FLModel` send/receive path as the
  default Lightning exchange, and must not pass the received `input_model` into
  the `Trainer`. Load `../_shared/pytorch-model-exchange.md` and
  `references/lightning-conversion.md` for the exact patch pattern.
- Must not inspect NVFLARE SDK source or docstrings to choose, override, or
  recover the Lightning conversion strategy. If public commands, recipe
  metadata, import checks, or validation prove the installed NVFLARE version
  cannot support the canonical skill path, report a version mismatch or
  skill/reference gap instead of switching to a source-discovered strategy.
- Must treat `flare.receive()` inside the patched loop as optional metadata or
  task-progression access only, not as a second model-load path.
- Must audit model constructor arguments before writing `job.py` by reading the
  `LightningModule.__init__` signature and the selected recipe's `model`
  parameter from `nvflare recipe show <recipe-name> --format json`, not by
  reading NVFLARE library source. If the `LightningModule` has required
  non-default `__init__` parameters, generate explicit recipe model config with
  `path` or `class_path` and `args`, then verify recipe construction and export
  preserve those arguments.
- Must use the PyTorch recipe family; must not invent a Lightning-only recipe.
- Must use external process launch such as `launch_external_process=True` for
  Lightning DDP or multi-GPU training instead of running distributed workers
  inside an in-process executor. See `references/lightning-ddp-and-tracking.md`.
- Must not make non-PyTorch-family skills load
  `../_shared/pytorch-model-exchange.md`.

## Agent Responsibilities

- Run project inspection and PyTorch recipe discovery before selecting a recipe.
- Explain the selected recipe when the user's algorithm intent is ambiguous.
- Convert the Lightning trainer to the patched Client API loop and generate or
  update `job.py`, preserving callbacks, loggers, and checkpoint behavior.
- Keep Lightning conversion choices, validation blockers, recipe comparisons,
  DDP/tracking decisions, and data-prep decisions within this skill, its
  references, and the shared lifecycle guidance.
- Report Lightning-specific blockers such as trainer construction that cannot be
  patched, callbacks or loggers incompatible with the FL round loop, checkpoint
  loading that conflicts with the patched model exchange, or DDP launch settings
  that the chosen recipe cannot express.

## User Input And Approval

- Ask the user to clarify FL workflow intent when recipe selection is uncertain.
- Ask before enabling experiment tracking unless the user requests it or the
  source code already uses a Lightning logger.
- Follow the shared lifecycle approval boundary for data-path changes,
  non-fixture validation data, POC, production, and startup-kit based runtime
  submission.

Load only the shared references needed for the current phase:
`../_shared/dependency-install.md` before Python import/inspect commands,
`../_shared/nvflare-job-lifecycle.md` for every conversion,
`../_shared/runtime-output-guidance.md` before choosing runtime/export
locations, `../_shared/validation-evidence.md` before validation, and
`../_shared/metrics-and-artifact-reporting.md` before final reporting. Load
`../_shared/pytorch-model-exchange.md` only for PyTorch-family model/state-dict
exchange.

Load the smallest Lightning-specific reference needed for the current phase:
`references/lightning-detection.md` before confirming Lightning versus plain
PyTorch, `references/lightning-conversion.md` when adding `flare.patch(trainer)`
and the patched training loop, `references/lightning-validation.md` before
validation or export, and `references/lightning-ddp-and-tracking.md` only for
DDP/multi-GPU or experiment-tracking work. Do not load every reference
preemptively, and do not depend on NVFLARE repository examples being present in
the user's environment.
