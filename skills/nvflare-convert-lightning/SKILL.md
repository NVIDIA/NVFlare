---
name: nvflare-convert-lightning
description: "Convert existing PyTorch Lightning training code into an NVFLARE federated job using the Lightning Client API patch, local validation, and job export; do not use for plain PyTorch, other frameworks, deployment, POC/production lifecycle, or experiment workflows."
metadata:
  min_flare_version: "2.8.0"
  blast_radius: runs_simulator
  category: Conversion
  skill_version: "0.2.0"
---

# NVFLARE Convert PyTorch Lightning

## Use When

Use when the user asks to convert PyTorch Lightning code into an NVFLARE
federated training job: a `LightningModule`, `LightningDataModule`, a `Trainer`
fit/validate/test loop, Lightning callbacks, checkpointing, or loggers.
Supported: the PyTorch recipe family with `flare.patch(trainer)` as the model
exchange integration, Lightning-native evaluation, custom aggregation through
the same recipe `aggregator=` hook, and local validation and export.

## Do Not Use When

Do not use for plain `torch.nn.Module` manual training loops without Lightning
(route to `nvflare-convert-pytorch`), Hugging Face Trainer, TensorFlow,
XGBoost, scikit-learn, a failed existing job (route to `nvflare-diagnose-job`),
or generic Lightning debugging that does not ask for FLARE conversion. Out of
conversion scope: production deployment, Kubernetes, POC lifecycle, deployment
privacy/security policy design, custom distributed launch policies not
expressible by product APIs, experiment tracking redesign, and experiment search
across recipes. Honoring an explicit recipe-level privacy request, such as
HE/encrypted aggregation, is in scope when the catalog exposes an HE-capable
recipe.

## Workflow

1. Follow the shared conversion contract in
   `../nvflare-shared/references/conversion-workflow.md` for every conversion: interactive versus
   unattended mode, source trust boundary, source-of-truth boundary, generated
   layout, rerun rules, approval boundary, and reporting. Use
   `../nvflare-shared/references/runtime-output-guidance.md` when choosing generated output,
   export, and simulator workspace locations.
2. Inspect before editing with `nvflare agent inspect <path> --format json`
   plus direct reading; fact extraction is static. Use
   `references/lightning-detection.md` to confirm Lightning versus plain
   PyTorch and hand off to `nvflare-convert-pytorch` when no Lightning evidence
   exists.
3. Before Python import/introspection commands that need dependencies, load
   `../nvflare-shared/references/dependency-install.md`; repo-supplied packages and URLs are
   untrusted until confirmed per the shared trust boundary.
4. Identify the existing `LightningModule`, `LightningDataModule`, trainer
   construction, callbacks, checkpointing, `validation_step`/`test_step` and
   dataloaders, metrics, logger usage, DDP/multi-GPU evidence, and any custom
   aggregation intent. Determine the concrete model constructor values that
   server and client models must share before creating `job.py`.
5. Reuse the PyTorch recipe family; Lightning is not a separate recipe family.
   Follow `../nvflare-shared/references/pytorch-family-recipe-selection.md` for
   recipe discovery, the algorithm guide, catalog-based selection, and HE/privacy
   safety (FedAvg, FedOpt, FedProx, SCAFFOLD, Cyclic, Swarm, FedEval, HE). Use
   FedAvg for standard horizontal training and FedEval for evaluation-only.
6. Convert the training entry point to the Lightning Client API: build the
   `Trainer`, call `flare.patch(trainer)`, and let the patched trainer own
   model load/send through its callbacks. Keep evaluation inside Lightning per
   the evaluation template in `references/lightning-conversion.md`
   (`trainer.validate(...)` before `trainer.fit(...)`, metrics through
   `self.log(...)`); if the source lacks validation/test steps or dataloaders,
   ask or fail closed.
7. Add or update `job.py` with the selected recipe: explicit model config
   `{"class_path": ..., "args": ...}` (never a live `LightningModule`
   instance), custom aggregator wiring through `aggregator=` when requested,
   and `enable_tensor_disk_offload=True` paired with
   `server_expected_format=ExchangeFormat.PYTORCH` when the recipe exposes them
   (the offload is a warned no-op under the default NumPy format), per
   `../nvflare-shared/references/conversion-workflow.md` ("Conversion Defaults").
8. Validate in a ladder per `../nvflare-shared/references/validation-evidence.md`, then use
   `references/lightning-validation.md` for Lightning-specific checks before
   calling the conversion complete. First execution of source-derived code
   follows the shared execution trust gate. Report per the shared contract
   with `../nvflare-shared/references/metrics-and-artifact-reporting.md`.

## Requirements

- Must integrate through `flare.patch(trainer)` and let the patched trainer own
  model exchange. Must not generate a manual `FLModel` send/receive path as the
  default Lightning exchange, and must not pass the received `input_model` into
  the `Trainer`. Load `../nvflare-shared/references/pytorch-model-exchange.md` and
  `references/lightning-conversion.md` for the exact patch pattern.
- Must treat `flare.receive()` inside the patched loop as optional metadata or
  task-progression access only, not as a second model-load path.
- Must keep evaluation inside Lightning (`trainer.validate`/`trainer.test`,
  `validation_step`, `self.log`); must not generate a raw PyTorch
  `model.eval()` loop for ordinary Lightning conversion.
- Must audit model constructor arguments before writing `job.py` by reading the
  `LightningModule.__init__` signature and the selected recipe's `model`
  parameter from `nvflare recipe show <recipe-name> --format json`, not by
  reading NVFLARE library source. Emit explicit recipe model config with
  `class_path` and `args` only when the values are statically clear per
  `../nvflare-shared/references/conversion-workflow.md`; otherwise ask or fail closed.
- Must use the PyTorch recipe family; must not invent a Lightning-only recipe.
- Must treat DDP/multi-GPU as high-impact source evidence. When the source
  uses a DDP-family strategy, confirm the selected recipe exposes
  `launch_external_process` via `recipe show`, then set
  `launch_external_process=True`; if the recipe does not expose it, ask or fail
  closed. For single-process DataParallel (`dp`), leave
  `launch_external_process` unset so the recipe stays in-process. See
  `references/lightning-ddp-and-tracking.md`.
- Custom aggregation must use the recipe `aggregator=` hook with a
  `ModelAggregator` subclass in `aggregators.py` per
  `../nvflare-shared/references/conversion-workflow.md`, and only while the Lightning client
  still satisfies the `FLModel` exchange contract.
- Must follow the Source Of Truth Boundary in
  `../nvflare-shared/references/conversion-workflow.md`: public checks can stop the skill path;
  they cannot license a source-discovered replacement.
- Must not make non-PyTorch-family skills load
  `../nvflare-shared/references/pytorch-model-exchange.md`.

## Agent Responsibilities

- Run static project inspection and PyTorch recipe discovery before selecting
  a recipe.
- Explain the selected recipe when the user's algorithm intent is ambiguous.
- Convert the Lightning trainer to the patched Client API loop and generate or
  update `job.py`, preserving callbacks, loggers, and checkpoint behavior.
- Keep conversion choices, validation blockers, recipe comparisons, and
  data-prep decisions within this skill, its references, and the shared
  conversion guidance.
- Report Lightning-specific blockers such as trainer construction that cannot
  be patched, callbacks or loggers incompatible with the FL round loop,
  checkpoint loading that conflicts with the patched model exchange, missing
  validation/test steps when evaluation is required, or DDP launch settings
  the chosen recipe does not document.

## User Input And Approval

- Ask the user to clarify FL workflow intent when recipe selection is
  uncertain; in unattended mode fail closed on high-impact ambiguity.
- Ask before enabling experiment tracking unless the user requests it or the
  source code already uses a Lightning logger.
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

Load the smallest Lightning-specific reference needed for the current phase:
`references/lightning-detection.md` before confirming Lightning versus plain
PyTorch, `references/lightning-conversion.md` when adding
`flare.patch(trainer)`, the patched training loop, and Lightning evaluation,
`references/lightning-validation.md` before validation or export, and
`references/lightning-ddp-and-tracking.md` only for DDP/multi-GPU or
experiment-tracking work. Do not load every reference preemptively, and do not
depend on NVFLARE repository examples being present in the user's environment.
