# Lightning Client API Conversion

This reference covers converting a PyTorch Lightning `Trainer` workflow to the
NVFLARE Lightning Client API. Load `../../nvflare-shared/references/pytorch-model-exchange.md` for
PyTorch-family tensor/state-dict rules before changing model exchange code.

The Lightning integration owns model load and send through callbacks installed
by `flare.patch(trainer)`. Do not generate a manual `FLModel` send/receive path
for normal Lightning training.

## Canonical Path

Use this path for Lightning conversion:

1. Confirm Lightning routing with `nvflare agent inspect`.
2. Select a PyTorch-family recipe with `nvflare recipe list/show`.
3. Generate `client.py` with `flare.patch(trainer)` as the model exchange path.
4. Generate `job.py` that builds the selected recipe and calls
   `recipe.execute(SimEnv(...))`.
5. Validate with `python job.py`, inspect terminal evidence, then export.

HE is not supported (steps 4–5): homomorphic-encryption recipes reject `SimEnv`
and require provisioned `PocEnv`/`ProdEnv`, which are outside conversion scope.
Follow the HE-not-supported rule in
`../../nvflare-shared/references/pytorch-family-recipe-selection.md`: report HE
as unsupported, route it to provisioning/deployment, and ask or fail closed
instead of generating or running an HE `job.py`.

Follow the shared Source Of Truth Boundary in
`../../nvflare-shared/references/conversion-workflow.md`.

## Conversion Pattern

- Import the Lightning client API: `import nvflare.client.lightning as flare`.
- Build the `LightningModule`, `LightningDataModule`, and `Trainer` as the
  source code already does.
- Call `flare.patch(trainer)` once, after the trainer is constructed.
- Loop while `flare.is_running()` and call `trainer.fit` / `trainer.validate` /
  `trainer.test` as the workflow requires.

```python
import nvflare.client.lightning as flare

flare.init()  # optional: only when get_site_name() or other pre-patch context is needed
trainer = Trainer(...)
flare.patch(trainer)

while flare.is_running():
    # Optional: call receive() only when round/site/task metadata is needed.
    # The patched trainer loads the global model internally.
    flare.receive()
    trainer.validate(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)  # when test evidence is requested/available
```

For evaluation-only / FedEval conversions, run `trainer.validate(...)` (the
patched trainer sends the validation metrics) and **do not call
`trainer.fit(...)`** — training was not requested, and fitting after the metrics
are sent can train an unwanted round or block the task. The packaged
`../assets/lightning_client.py` `main(..., evaluate_only=True)` skips `fit`.

## Patch Ownership Rules

- `flare.patch(trainer)` installs callbacks that receive the global model into
  the Lightning model and send the updated model after fit. Let it own the
  exchange.
- Do not pass the received `input_model` into `Trainer` methods. The patched
  trainer loads the global model internally.
- Do not add a second manual `flare.send(FLModel(...))` for normal training; the
  patched trainer already sends the trained model.
- Use `flare.init()` and `flare.get_site_name()` only when the code needs
  pre-patch context such as site-specific logging or data paths.
- Use `flare.receive()` in the patched loop only for FL task progression,
  round/site logging, or task metadata, never for manual model loading.

## Lightning Evaluation Template

Keep evaluation inside Lightning; do not reuse the raw PyTorch
`model.eval()` / `torch.no_grad()` loop for normal Lightning conversion:

- Require or preserve `validation_step()` / `test_step()` and a
  validation/test dataloader or `LightningDataModule`.
- Log validation metrics from the `LightningModule` with `self.log(...)` so
  they are visible in the trainer callback metrics.
- After `flare.patch(trainer)` and `flare.receive()`, call
  `trainer.validate(model, datamodule=...)` before `trainer.fit(...)` when
  training-with-evaluation or server-side model selection needs validation
  metrics; keep this inside the `while flare.is_running()` loop so the round
  reports global-model metrics.
- Use `trainer.test(...)` only when the source workflow already has test
  semantics or the user requests test reporting.
- Rely on Lightning's validate/test loops to set evaluation mode and disable
  gradients; generate a manual `model.eval()` loop only when the conversion
  intentionally routes to plain PyTorch.
- If the source project lacks validation/test steps or dataloaders, ask in
  interactive mode or fail closed in unattended mode instead of inventing
  metric semantics.

This template is self-contained packaged guidance; do not depend on NVFLARE
repository `examples/` being present in the user's environment. The runnable
form ships at `../assets/lightning_client.py`; adapt it rather than inventing a
new structure.

## Preserve Lightning Behavior

- Preserve user callbacks, loggers, and checkpoint callbacks unless the user
  asks to change them.
- Repo-shipped checkpoint files (`.ckpt` passed to `load_from_checkpoint`,
  `Trainer.fit(ckpt_path=...)`, or resume logic) are untrusted executable input
  per `../../nvflare-shared/references/conversion-workflow.md`: full-unpickle loading of a
  repo-supplied checkpoint is ask/fail. Checkpoints produced by the current
  validation run (for example `ckpt_path="best"` from this run's checkpoint
  callback) may follow normal Lightning handling.
- Keep the `LightningModule`/`LightningDataModule` architecture and data logic;
  do not rewrite training_step/validation_step semantics.
- Avoid repeated expensive setup (model build, dataset download) inside the FL
  round loop; construct the model, datamodule, and trainer once before the loop
  when the source code allows it.

## Local Data And Loss Policy

Follow the training-policy distinction in
`../../nvflare-shared/references/pytorch-model-exchange.md` and the site split guidance in
`../../nvflare-shared/references/conversion-workflow.md`. Lightning-specific implication:
label/site-derived values that affect `training_step`, `LightningDataModule`
sampling, or validation/test decision logic remain local to each site partition
unless the user explicitly requests one global training policy. Do not move
those values into recipe `model` args just because architecture args must be
shared.

Report the split policy, seed, and where local training-policy values are
computed.

## Model Construction Consistency

Follow the shared model-config and construction-consistency rule in
`../../nvflare-shared/references/conversion-workflow.md` ("Recipe Model Config"):
same class and constructor args on server and client, explicit
`{"class_path": ..., "args": ...}` config (no live instance), and
derive-or-ask/fail-closed for required values.

Lightning-specific delta: the exchanged unit is the whole `LightningModule`
managed by the patched trainer, so construct the identical `LightningModule` on
the server (via the recipe `model` config) and on the client in `client.py`, not
just the inner `torch.nn.Module`. Express shared arguments as a `model_args`
dict in the recipe model config (prefer `class_path`; `path` is the normalized
job-config key).

## Source Layout

Use the canonical FLARE source layout defined in
`../../nvflare-shared/references/conversion-workflow.md` ("Generated Job Layout").
Lightning-specific delta: `client.py` patches the trainer as the model-exchange
path, and `model.py` holds the `LightningModule` (and `LightningDataModule`)
definition when a new file is needed. Avoid ad hoc names such as `fl_train.py`
unless the user requests them, and use
`../../nvflare-shared/references/runtime-output-guidance.md` for runtime
workspaces, exported job directories, and validation output locations.

## Recipe Reuse

Lightning reuses the PyTorch recipe family. Follow
`../../nvflare-shared/references/pytorch-family-recipe-selection.md` for recipe
discovery, the algorithm guide, catalog-based selection rules, and the
HE-not-supported rule — the same catalog and rules apply to Lightning, including
non-FedAvg workflows such as FedOpt, FedProx, SCAFFOLD, Cyclic, Swarm, and
FedEval. Use FedAvg for standard horizontal training and FedEval for
evaluation-only.

The generated `job.py` should use the selected recipe's public parameters from
`recipe show`, construct the model through explicit `class_path` (or `path`) plus
`args` when constructor arguments are required, and call
`recipe.execute(SimEnv(...))`. HE is not supported: homomorphic-encryption
recipes reject `SimEnv` and require provisioned `PocEnv`/`ProdEnv` outside
conversion scope — follow the HE-not-supported rule in
`../../nvflare-shared/references/pytorch-family-recipe-selection.md` (report
unsupported, route to provisioning/deployment, ask or fail closed; do not
generate an HE job). Do not replace this with ad
hoc SDK-internal APIs based on local source or docstring inspection. Follow
`../../nvflare-shared/references/conversion-workflow.md` for export and
command-line behavior.
