# Lightning Detection

Use this reference to confirm that the project is a PyTorch Lightning project
before applying the Lightning conversion pattern, and to hand off to
`nvflare-convert-pytorch` when no Lightning evidence exists.

## Positive Lightning Evidence

Treat the project as Lightning when any of the following are present:

- imports of `pytorch_lightning` or `lightning.pytorch` (including aliased
  imports such as `import lightning as L` or `import pytorch_lightning as pl`);
- a class that subclasses `LightningModule` or `LightningDataModule`;
- a `Trainer(...)` construction followed by `trainer.fit`, `trainer.validate`,
  or `trainer.test`;
- Lightning callbacks, `LightningModule.training_step`/`validation_step`, or
  Lightning loggers such as `TensorBoardLogger` or `MLFlowLogger`.

`nvflare agent inspect <path> --format json` reports Lightning framework
evidence and routing; use it to confirm before editing.

## Plain PyTorch Versus Lightning

Route to `nvflare-convert-pytorch` instead of this skill when the project uses a
plain `torch.nn.Module` with a manual training loop (an explicit
`for batch in loader:` loop calling `loss.backward()` and `optimizer.step()`)
and there is no `LightningModule`, `LightningDataModule`, or `Trainer`.

A project may contain both a plain `nn.Module` and a `LightningModule`. Choose
the skill from the training entry point the user asks to federate:

- if the user federates the `Trainer` fit/validate/test loop, use this skill;
- if the user federates a manual PyTorch loop and only borrows an `nn.Module`,
  use `nvflare-convert-pytorch`.

## Lightning Trainer Wrappers

Some ecosystems build the trainer through a wrapper or factory, for example
`nl.Trainer(...)` from `nemo.lightning`. Do not treat wrapper imports alone as
PyTorch Lightning evidence. Use this skill only when the user explicitly asks
for Lightning conversion, canonical PyTorch Lightning evidence is also present,
or an existing/verified `nvflare.client.lightning.patch(trainer)` compatibility
signal shows that the wrapper trainer can use the Lightning Client API. The
definitive converted-state signal is a patched trainer, not the exact
constructor module.

## Negative Handoff

When no Lightning evidence exists, do not force the Lightning patch pattern.
State that the project is not Lightning and route to `nvflare-convert-pytorch`
for plain PyTorch loops, or to the appropriate framework conversion skill for
TensorFlow, XGBoost, scikit-learn, or Hugging Face code.
