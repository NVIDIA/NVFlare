# Lightning Client API Conversion

This reference covers converting a PyTorch Lightning `Trainer` workflow to the
NVFLARE Lightning Client API. Load `../../_shared/pytorch-model-exchange.md` for
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

Follow the Source Of Truth Boundary in
`../../_shared/conversion-workflow.md`: public checks can stop the skill path;
they cannot license a source-discovered replacement.

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
form ships at `templates/lightning_client.py`; adapt it rather than inventing a
new structure.

## Preserve Lightning Behavior

- Preserve user callbacks, loggers, and checkpoint callbacks unless the user
  asks to change them.
- Repo-shipped checkpoint files (`.ckpt` passed to `load_from_checkpoint`,
  `Trainer.fit(ckpt_path=...)`, or resume logic) are untrusted executable input
  per `../../_shared/conversion-workflow.md`: full-unpickle loading of a
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
`../../_shared/pytorch-model-exchange.md` and the site split guidance in
`../../_shared/conversion-workflow.md`. Lightning-specific implication:
label/site-derived values that affect `training_step`, `LightningDataModule`
sampling, or validation/test decision logic remain local to each site partition
unless the user explicitly requests one global training policy. Do not move
those values into recipe `model` args just because architecture args must be
shared.

Report the split policy, seed, and where local training-policy values are
computed.

## Model Construction Consistency

Follow the shared state-dict and constructor rules in
`../../_shared/pytorch-model-exchange.md`. The Lightning-specific point is that
the exchanged unit is the whole `LightningModule` managed by the patched
trainer, so construct the identical `LightningModule` on the server through the
recipe `model` config and on the client in `client.py`, not just the inner
`torch.nn.Module`. Express shared arguments with a `model_args` dict, an
explicit `{"class_path": "model.LitClass", "args": model_args}` recipe model
config (prefer `class_path`; `path` is the normalized job-config key), or
explicit `train_args`. Do not pass a live `LightningModule` instance as the
recipe model input; when constructor args are not statically clear per
`../../_shared/conversion-workflow.md`, ask in interactive mode or fail closed
in unattended mode.

## Source Layout

Generated Lightning job source should normally contain:

- `client.py`: Lightning Client API entry point that patches the trainer;
- `job.py`: recipe or FedJob builder, simulation entry point, and export entry
  point;
- `model.py`: the `LightningModule` (and `LightningDataModule`) definition when
  a new file is needed;
- `aggregators.py`: only when the conversion includes custom aggregation (see
  `../../_shared/conversion-workflow.md`, "Custom Aggregation");
- `prepare_data.py` / `download_data.py`: only when the conversion generates
  data setup code;
- `requirements.txt` only when dependencies differ from the source project.

Avoid ad hoc names such as `fl_train.py` unless the user requests them. Use
`../../_shared/runtime-output-guidance.md` for runtime workspaces, exported job
directories, and validation output locations.

## Recipe Reuse

Lightning reuses the PyTorch recipe family. Select the recipe from the user's FL
intent with `nvflare recipe list --framework pytorch --format json` and
`nvflare recipe show <recipe-name> --format json`. Use FedAvg for standard
horizontal training and FedEval for evaluation-only. Follow
`../../_shared/conversion-workflow.md` for export and command-line behavior.

The generated `job.py` should use the selected recipe's public parameters from
`recipe show`, construct the model through explicit `class_path` (or `path`) plus
`args` when constructor arguments are required, and call
`recipe.execute(SimEnv(...))`. Do not replace this with ad hoc SDK-internal
APIs based on local source or docstring inspection.
