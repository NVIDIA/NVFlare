---
name: nvflare-convert-lightning
description: "Convert existing PyTorch Lightning training code into an NVFLARE federated job using the Lightning Client API patch, local validation, and job export; do not use for plain PyTorch, other frameworks, deployment, POC/production lifecycle, or experiment workflows."
license: Apache-2.0
metadata:
  author: "NVIDIA FLARE Team <federatedlearning@nvidia.com>"
  min_flare_version: "2.9.0"
  blast_radius: runs_simulator
  category: Conversion
  version: "0.1.0"
  tags: "nvflare, federated-learning, pytorch-lightning, conversion"
  languages: "python"
  frameworks: "pytorch-lightning, pytorch, nvflare"
  domain: ml
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
(route to `nvflare-convert-pytorch`), Hugging Face Trainer (route to `nvflare-convert-huggingface`), TensorFlow,
XGBoost, scikit-learn, a failed job (route to `nvflare-diagnose-job`),
federated statistics without training (route to `nvflare-fed-stats`), or
generic Lightning debugging without FLARE intent; when the inspected project
actively contains both Lightning and Hugging Face Trainer entrypoints, route to
`nvflare-orient`. Out of conversion scope: production deployment, Kubernetes,
POC lifecycle, deployment privacy/security policy design, custom distributed
launch policies not expressible by product APIs, experiment tracking redesign,
and experiment search across recipes. Privacy-protection requests — homomorphic encryption (HE) /
encrypted aggregation, differential privacy, and privacy filters — are not
supported: they require provisioning or deployment policy beyond conversion
scope, so report such a request as unsupported and route it to
provisioning/deployment, never substituting an unprotected recipe or disclaimer.
If a request combines federated statistics and model-training conversion,
treat it as two independent jobs and workflows: do not merge or automatically
chain them, do not route the combination to `nvflare-orient`, and ask which
workflow to run first before generating or running either job. Recommend
`nvflare-fed-stats` first only when the user's purpose is to understand data
distribution; handle conversion later as a separate request.

## Workflow

1. Load `../nvflare-shared/references/conversion-common.md` and apply it for the
   whole conversion; this SKILL.md states only the framework-specific deltas.
   Load `../nvflare-shared/references/conversion-workflow.md` only for a non-standard
   case that needs its detailed rerun, data-location, authorization, or
   missing-semantics guidance.
2. Inspect before editing with `nvflare agent inspect source <path> --format json`
   plus direct reading; fact extraction is static. Use
   `references/lightning-detection.md` to confirm Lightning versus plain
   PyTorch and hand off to `nvflare-convert-pytorch` when no Lightning evidence
   exists. If inspection recommends `nvflare-orient` for active Lightning and
   Hugging Face Trainer owners, stop and hand off before editing.
3. Apply the dependency-install ordering rule in `../nvflare-shared/references/conversion-common.md` before
   any Python command imports user, Lightning, NVFLARE, or declared dependency
   modules.
4. Identify the existing `LightningModule`, `LightningDataModule`, trainer
   construction, callbacks, checkpointing, `validation_step`/`test_step` and
   dataloaders, metrics, logger usage, source partition evidence, distributed
   process-spawning evidence, custom aggregation intent, and the concrete model
   constructor values that server and clients must share.
5. Reuse the PyTorch recipe family; Lightning is not a separate recipe family.
   For the standard case — the user explicitly requests FedAvg and inspection
   identifies Lightning — run `nvflare recipe show fedavg-pt --format json`
   directly and construct it. Use the returned module, class, and parameters;
   for `fedavg-pt`, import `FedAvgRecipe` from
   `nvflare.app_opt.pt.recipes.fedavg`, never from `nvflare.recipe`. Load
   `../nvflare-shared/references/pytorch-family-recipe-selection.md` (discovery,
   algorithm guide, catalog-based selection, HE-not-supported rule; FedAvg,
   FedOpt, FedProx, SCAFFOLD, Cyclic, Swarm, FedEval) only for ambiguous or
   non-FedAvg algorithms, reserving `nvflare recipe list` for those cases. Use
   FedEval for evaluation-only. After every `recipe show`, load
   `../nvflare-shared/references/pytorch-family-recipe-construction.md` and
   derive the recipe's construction capabilities.
6. Convert the training entry point to the Lightning Client API: build the
   `Trainer`, call `flare.patch(trainer)`, and let the patched trainer own
   model exchange. Keep evaluation inside Lightning per
   `references/lightning-conversion.md`: validate before fit and use `self.log`.
   When server metrics are required, follow that reference to preserve scalar
   results under `MetaKey.INITIAL_METRICS`; calling `trainer.validate(...)`
   alone does not prove delivery. Ask or fail closed when validation semantics
   are missing. Partition site data per the "Site Data Partitioning" rule in
   `../nvflare-shared/references/conversion-common.md`.
7. Add or update `job.py` under the shared "Recipe Model Config" policy. A
   direct instance, when allowed by that policy, must be the complete
   `LightningModule`, not its inner network. Add the requested `aggregator=`
   wiring and the metric, tensor-transport, server
   offload, and execution settings derived from the shared PyTorch-family
   construction profile.
8. Validate in a ladder per `../nvflare-shared/references/validation-evidence.md`:
   compile checks, recipe construction, one final full-run path chosen by the
   artifact being validated, and export inspection; then use
   `references/lightning-validation.md` for Lightning-specific checks before
   calling the conversion complete. Use the environment and permission
   mechanisms supplied by the agent host; do not inspect or enforce its security
   boundary. Report the recipe, changed files, validation status, metrics, and
   exact artifact paths. Load
   `../nvflare-shared/references/metrics-and-artifact-reporting.md` only when
   normal metric artifacts are absent or inconsistent.

## Requirements

- Must integrate through `flare.patch(trainer)` and let the patched trainer own
  model exchange. Must not generate a manual `FLModel` send/receive path as the
  default Lightning exchange, and must not pass the received `input_model` into
  the `Trainer`. Load `../nvflare-shared/references/pytorch-model-exchange.md`
  and `references/lightning-conversion.md` for the patch pattern.
- Must treat `flare.receive()` inside the patched loop as optional metadata or
  task-progression access only, not as a second model-load path.
- Must keep evaluation inside Lightning (`trainer.validate`/`trainer.test`,
  `validation_step`, `self.log`); must not generate a raw PyTorch
  `model.eval()` loop for ordinary Lightning conversion.
- When training promises server metrics, must preserve finite scalar pre-fit
  validation results through `model.__fl_meta__[MetaKey.INITIAL_METRICS]` per
  `references/lightning-conversion.md` and `assets/lightning_client.py`; this is
  patched-exchange metadata, not a second manual `flare.send(...)`.
- Must audit model constructor arguments before writing `job.py` by reading the
  `LightningModule.__init__` signature and the selected recipe's `model`
  parameter from `nvflare recipe show <recipe-name> --format json`, not by
  reading NVFLARE library source. The shared "Recipe Model Config" policy
  governs whether to emit `class_path`/`args` config or a direct
  `LightningModule`; required values must be statically clear from literal
  source, configuration, or supplied metadata. Otherwise ask one semantic
  question when an answer channel exists or fail closed on that missing value.
- Must use the PyTorch recipe family; must not invent a Lightning-only recipe.
- Must apply
  `../nvflare-shared/references/pytorch-family-recipe-construction.md` after
  `recipe show`; it is the canonical policy for optional recipe parameters,
  model selection, tensor transport, server disk offload, and execution mode.
  For Lightning DDP details see
  `references/lightning-ddp-and-tracking.md`.
- Must preserve local-only callbacks and logger behavior where safe. Existing
  network-connected tracking, upload callbacks, and custom/unknown loggers are
  evidence, not a user request: keep them disabled during validation unless
  explicitly requested, and do not ask solely to enable them. This narrows
  `references/lightning-conversion.md`.
- Must not make non-PyTorch-family skills load
  `../nvflare-shared/references/pytorch-model-exchange.md`.
- Site partitioning, custom aggregation, the Source Of Truth Boundary, and user
  input/authorization follow `../nvflare-shared/references/conversion-common.md`.

Always read this converter SKILL.md together with
`../nvflare-shared/references/conversion-common.md`. Load detailed references
only at their named phase:
`../nvflare-shared/references/conversion-workflow.md` for non-standard cases;
`../nvflare-shared/references/pytorch-family-recipe-selection.md` only for ambiguous or non-FedAvg algorithms;
`../nvflare-shared/references/pytorch-family-recipe-construction.md` after every `recipe show`;
`../nvflare-shared/references/dependency-install.md` only when an install is needed;
`../nvflare-shared/references/runtime-output-guidance.md` only for read-only source roots or chosen outputs;
`../nvflare-shared/references/metrics-and-artifact-reporting.md` only when metrics are absent or inconsistent;
`../nvflare-shared/references/validation-evidence.md` before validation;
`../nvflare-shared/references/pytorch-model-exchange.md` for PyTorch-family exchange.
For Lightning work load `references/lightning-detection.md`, `references/lightning-conversion.md`,
`references/lightning-validation.md`, or `references/lightning-ddp-and-tracking.md` only as needed.
Do not depend on NVFLARE repository examples being present.
