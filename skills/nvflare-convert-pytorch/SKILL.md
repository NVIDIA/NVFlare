---
name: nvflare-convert-pytorch
description: "Convert existing PyTorch training code into an NVFLARE federated job using Client API model exchange, local validation, and job export; do not use for other frameworks, deployment, POC/production lifecycle, or experiment workflows."
metadata:
  min_flare_version: "2.8.0"
  blast_radius: runs_simulator
  category: Conversion
  skill_version: "0.2.0"
---

# NVFLARE Convert PyTorch

## Use When

Use when the user asks to convert an existing plain PyTorch training script,
`torch.nn.Module`, manual training loop, `state_dict` workflow, data loader,
checkpoint, or metric loop into an NVFLARE federated training job. Supported:
horizontal FL with a supported PyTorch recipe, Client API model exchange with
`nvflare.client` and `FLModel`, custom aggregation through the recipe
`aggregator=` hook, and local validation and export.

## Do Not Use When

Do not use for PyTorch Lightning (route to `nvflare-convert-lightning`),
Hugging Face Trainer, TensorFlow, XGBoost, scikit-learn, a failed existing job
(route to `nvflare-diagnose-job`), or generic PyTorch debugging that does not
ask for FLARE conversion. Out of conversion scope: production deployment,
Kubernetes, POC lifecycle, deployment privacy/security policy design, controller
or workflow rewrites outside product recipe or Job APIs, experiment search across
recipes, and data distribution experiments beyond minimal local validation setup.
Honoring an explicit recipe-level privacy request, such as HE/encrypted
aggregation, is in scope when the catalog exposes an HE-capable recipe.

## Workflow

1. Follow the shared conversion contract in
   `../nvflare-shared/references/conversion-workflow.md` for every conversion: interactive versus
   unattended mode, source trust boundary, source-of-truth boundary, generated
   layout, rerun rules, approval boundary, and reporting. Use
   `../nvflare-shared/references/runtime-output-guidance.md` when choosing generated output,
   export, and simulator workspace locations.
2. Inspect before editing with `nvflare agent inspect <path> --format json`
   plus direct reading. Fact extraction is static; do not import or execute
   user training modules to discover fields. Extract: training entrypoint,
   model class path and constructor args, checkpoint behavior, train/eval
   functions, data loading, metric names and denominators, local epochs/steps,
   requested client and round counts, tracking evidence, DDP evidence, and any
   custom aggregation intent.
3. Before Python import/introspection commands that need dependencies, load
   `../nvflare-shared/references/dependency-install.md`; repo-supplied packages and URLs are
   untrusted until confirmed per the shared trust boundary.
4. Select the recipe from the requested FL workflow, not from PyTorch alone,
   using `../nvflare-shared/references/pytorch-family-recipe-selection.md` for
   discovery, the algorithm guide, catalog-based selection, and HE/privacy
   safety. Use FedAvg only for standard horizontal model-parameter aggregation.
   For plain-PyTorch `job.py` construction and the portable FedAvg fast path,
   use `references/recipe-selection.md`; do not add per-site recipe config unless
   sites actually differ. Confirm parameters with
   `nvflare recipe show <recipe-name> --format json`.
5. Convert training and evaluation as a pair using
   `references/pytorch-client-api-conversion.md`: initialize FLARE, receive an
   `FLModel`, load `params`, evaluate the received global model, train, and
   send an `FLModel` with updated `params` and `metrics`. Adapt the user's
   evaluation code into the packaged evaluation template; if evaluation is
   required but missing, ask or fail closed.
6. Add or update `job.py` with the selected recipe: explicit model config
   `{"class_path": ..., "args": ...}` (never a live model instance), custom
   aggregator wiring through `aggregator=` when requested, and
   `enable_tensor_disk_offload=True` when the recipe exposes it.
7. Validate in a ladder per `../nvflare-shared/references/validation-evidence.md`: compile
   checks, recipe construction, local simulation, then export per
   `../nvflare-shared/references/conversion-workflow.md` ("Export"); use
   `references/job-validation.md` for PyTorch-specific checks. Stop at the
   first failed rung and report the product error. First execution of
   source-derived code follows the shared execution trust gate.
8. Report per the shared contract, using
   `../nvflare-shared/references/metrics-and-artifact-reporting.md` for metric and artifact
   evidence.

## Requirements

- Must audit model constructor arguments before writing `job.py` by reading the
  model module's `__init__` and the selected recipe's `model` parameter from
  `nvflare recipe show <recipe-name> --format json`, not by reading NVFLARE
  library source. Emit explicit recipe model config with `class_path` and
  `args` only when the values are statically clear per
  `../nvflare-shared/references/conversion-workflow.md`; otherwise ask in interactive mode or
  fail closed in unattended mode.
- Must keep outbound PyTorch model weights as `torch.Tensor` values in
  `FLModel(params=...)` when using `PTInProcessClientAPIExecutor`; load
  `../nvflare-shared/references/pytorch-model-exchange.md` and
  `references/pytorch-client-api-conversion.md` for the exact send pattern.
- Must convert source evaluation alongside training and return metrics through
  `FLModel.metrics`; must not synthesize metric semantics without source
  evidence.
- Must load checkpoints with `torch.load(..., weights_only=True)`; a
  checkpoint that needs full unpickling is ask/fail, per
  `references/pytorch-client-api-conversion.md`.
- Custom aggregation must use the recipe `aggregator=` hook with a
  `ModelAggregator` subclass in `aggregators.py` per
  `../nvflare-shared/references/conversion-workflow.md`; algorithms needing new client/server
  exchange semantics also need the matching client transformation, or ask/fail.
- Must follow the Source Of Truth Boundary in
  `../nvflare-shared/references/conversion-workflow.md`: public checks can stop the skill path;
  they cannot license a source-discovered replacement.
- Must not make TensorFlow or other non-PyTorch skills load
  `../nvflare-shared/references/pytorch-model-exchange.md`; that reference is only for
  PyTorch-family model/state-dict exchange.

## Agent Responsibilities

- Run static project inspection and recipe discovery before selecting a recipe.
- Explain the selected recipe when the user's algorithm intent is ambiguous.
- Convert PyTorch Client API model exchange and generate or update `job.py`.
- Keep conversion choices, validation blockers, recipe comparisons, and
  data-prep decisions within this skill, its references, and the shared
  conversion guidance.
- Report PyTorch-specific blockers such as non-`state_dict` model state,
  checkpoints requiring unsafe deserialization, unsupported metric
  serialization, or data loaders that cannot be parameterized per site.

## User Input And Approval

- Ask the user to clarify FL workflow intent when recipe selection is
  uncertain; in unattended mode fail closed on high-impact ambiguity.
- Follow the shared approval boundary in `../nvflare-shared/references/conversion-workflow.md`
  for overwriting files, installing dependencies, fetching repo-supplied URLs,
  downloading data, and first execution of source-derived code. POC or
  production submission is outside conversion scope.

Load only the shared references needed for the current phase:
`../nvflare-shared/references/conversion-workflow.md` for every conversion,
`../nvflare-shared/references/dependency-install.md` before Python import/introspection commands,
`../nvflare-shared/references/runtime-output-guidance.md` before choosing runtime/export
locations, `../nvflare-shared/references/validation-evidence.md` before validation, and
`../nvflare-shared/references/metrics-and-artifact-reporting.md` before final reporting. Load
`../nvflare-shared/references/pytorch-model-exchange.md` only for PyTorch-family model/state-dict
exchange.

Load the smallest PyTorch-specific reference needed for the current phase:
`references/recipe-selection.md` before selecting or constructing a recipe,
`references/pytorch-client-api-conversion.md` when converting training and
evaluation code to Client API model exchange, and
`references/job-validation.md` before validation, export, or debugging
PyTorch-specific validation failures. Do not load every reference
preemptively, and do not depend on NVFLARE repository examples being present
in the user's environment.
