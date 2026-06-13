---
name: nvflare-convert-pytorch
description: "Convert existing PyTorch training code into an NVFLARE federated job using Client API model exchange, local validation, and job export; do not use for other frameworks or deployment-only tasks."
min_flare_version: "2.8.0"
blast_radius: edits_files
skill_version: "0.1.0"
---

# NVFLARE Convert PyTorch

## Use When

Use when the user asks to convert an existing PyTorch training script,
`torch.nn.Module`, `state_dict` workflow, data loader, checkpoint, or metric
loop into an NVFLARE federated training job.

## Do Not Use When

Do not use for PyTorch Lightning, Hugging Face Trainer, TensorFlow, XGBoost,
scikit-learn, Kubernetes deployment, production submission, or generic PyTorch
debugging that does not ask for FLARE conversion.

## Workflow

1. Before Python import/inspect commands, install applicable source
   `requirements*.txt` files in the active `nvflare` environment. Use `uv pip`
   when available; see the shared lifecycle for interpreter selection and avoid
   `uv pip install --system` with virtual environments.
2. Follow the shared conversion workflow contract in
   `../_shared/nvflare-job-lifecycle.md`.
3. Identify the PyTorch model definition, required `nn.Module.__init__` arguments,
   training loop, data loading, metrics, and checkpoint behavior. Determine the
   concrete constructor values that server and client models must share before
   creating `job.py`.
4. Run `nvflare recipe list --framework pytorch --format json` and select the
   recipe from the requested FL workflow, not from PyTorch alone. Use FedAvg
   only for standard horizontal model-parameter aggregation.
   For standard FedAvg, use the portable fast path in
   `references/recipe-selection.md`; do not add per-site recipe config unless
   the sites actually need different training scripts, arguments, or launch
   settings.
5. Convert training exchange to the FLARE Client API: initialize FLARE, receive
   an `FLModel`, load `params` into the PyTorch model, train or evaluate, and
   send an `FLModel` with updated `params`, metrics, and useful metadata.
6. Add or update a `job.py` that uses the selected PyTorch recipe or job API
   path. Follow the shared lifecycle for generated layout, validation, export,
   runtime locations, and evidence reporting.
7. Validate and export through the shared lifecycle. Use
   `references/job-validation.md` for PyTorch-specific checks before calling the
   conversion complete.

## Requirements

- Must audit model constructor arguments before writing `job.py`. If the model
  has required non-default `__init__` parameters, generate explicit recipe model
  config with `path` or `class_path` and `args`, then verify recipe
  construction and export preserve those arguments.
- Must follow the shared job lifecycle guidance for validation evidence,
  including final/best metrics, round/per-site metrics, and artifact paths when
  those artifacts are present.
- Must not require `rg` to be installed; the shared lifecycle defines fallback
  search options.

## Agent Responsibilities

- Run project inspection and recipe discovery before selecting a recipe.
- Explain the selected recipe when the user's algorithm intent is ambiguous.
- Convert PyTorch Client API model exchange and generate or update `job.py`.
- Keep PyTorch conversion choices, validation blockers, recipe comparisons, and
  data-prep decisions within this skill, its references, and the shared
  lifecycle guidance.
- Report PyTorch-specific blockers such as non-`state_dict` model state,
  incompatible checkpoint loading, unsupported metric serialization, or data
  loaders that cannot be parameterized per site.

## User Input And Approval

- Ask the user to clarify FL workflow intent when recipe selection is uncertain.
- Follow the shared lifecycle approval boundary for data-path changes,
  non-fixture validation data, POC, production, and startup-kit based runtime
  submission.

Load `../_shared/nvflare-job-lifecycle.md` for every conversion. Load the
smallest PyTorch-specific reference needed for the current phase:
`references/recipe-selection.md` before selecting or constructing a recipe,
`references/pytorch-client-api-conversion.md` when converting training code to
Client API model exchange, and `references/job-validation.md` before validation,
export, or debugging PyTorch-specific validation failures. Do not load every
reference preemptively, and do not depend on NVFLARE repository examples being
present in the user's environment.
