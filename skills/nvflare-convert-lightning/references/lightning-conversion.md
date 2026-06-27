# Lightning Client API Conversion

This reference covers converting a PyTorch Lightning `Trainer` workflow to the
NVFLARE Lightning Client API. Load `../../_shared/pytorch-model-exchange.md` for
PyTorch-family tensor/state-dict rules before changing model exchange code.

The Lightning integration owns model load and send through callbacks installed
by `flare.patch(trainer)`. Do not generate a manual `FLModel` send/receive path
for normal Lightning training.

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
    # Do not pass input_model to trainer; the patched trainer loads the
    # global model internally.
    input_model = flare.receive()
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

## Validation Before Training

When server-side model selection needs validation metrics, call
`trainer.validate(...)` on the received global model before `trainer.fit(...)`
so the round reports global-model metrics. Keep this inside the
`while flare.is_running()` loop.

## Preserve Lightning Behavior

- Preserve user callbacks, loggers, and checkpoint callbacks unless the user
  asks to change them.
- Keep the `LightningModule`/`LightningDataModule` architecture and data logic;
  do not rewrite training_step/validation_step semantics.
- Avoid repeated expensive setup (model build, dataset download) inside the FL
  round loop; construct the model, datamodule, and trainer once before the loop
  when the source code allows it.

## Model Construction Consistency

Follow the shared state-dict and constructor rules in
`../../_shared/pytorch-model-exchange.md`. The Lightning-specific point is that
the exchanged unit is the whole `LightningModule` managed by the patched
trainer, so construct the identical `LightningModule` on the server through the
recipe `model` config and on the client in `client.py`, not just the inner
`torch.nn.Module`. Express shared arguments with a `model_args` dict, an
explicit `{"path": "model.LitClass", "args": model_args}` recipe model config,
or explicit `train_args`.

## Source Layout

Generated Lightning job source should normally contain:

- `client.py`: Lightning Client API entry point that patches the trainer;
- `job.py`: recipe or FedJob builder, simulation entry point, and export entry
  point;
- `model.py`: the `LightningModule` (and `LightningDataModule`) definition when
  a new file is needed;
- `requirements.txt` only when dependencies differ from the source project.

Avoid ad hoc names such as `fl_train.py` unless the user requests them. Use
`../../_shared/runtime-output-guidance.md` for runtime workspaces, exported job
directories, and validation output locations.

## Recipe Reuse

Lightning reuses the PyTorch recipe family. Select the recipe from the user's FL
intent with `nvflare recipe list --framework pytorch --format json` and
`nvflare recipe show <recipe-name> --format json`. Use FedAvg for standard
horizontal training and FedEval for evaluation-only. Follow
`../../_shared/nvflare-job-lifecycle.md` for export and command-line behavior.
