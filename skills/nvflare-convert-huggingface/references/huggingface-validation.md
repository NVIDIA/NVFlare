# Hugging Face Conversion Validation

Run these checks after the shared validation ladder. Stop at the first failure.

## Static Checks

- Compile generated `client.py`, `model.py`, and `job.py`.
- Confirm one `flare.patch(trainer)` call and one rank-symmetric
  `flare.is_running()` loop.
- Confirm no manual model `receive()`/`send()` path was added.
- Confirm model, Trainer, datasets, and tokenizer are constructed outside the
  FL round loop.
- Confirm generated `train_args` quote configurable model and data paths.
- Confirm site data remains external to the exported job.

## Import And Contract Checks

- Import the generated model and job modules after declared dependencies are
  installed.
- Construct the selected recipe and let product validation reject unsupported
  arguments.
- Construct the Trainer with a minimized local model/dataset when practical;
  do not replace the user's final model merely to make the generated job pass.
- Verify `flare.patch()` accepts the Trainer configuration.
- Verify DeepSpeed, FSDP, best-model-at-end, save-only-model, and prebuilt
  optimizer/checkpoint constraints before simulation.

## Parameter Checks

- Extract server and Trainer exchange key sets without training.
- Require exact full-model or adapter-key agreement after the documented
  prefix transformation.
- For PEFT, verify both sides use the same adapter configuration and that only
  adapter parameters are exchanged.
- Confirm PyTorch exchange format for BF16/FP16 or other dtype-sensitive
  models.

## Execution Checks

- Run a bounded simulation with the requested site count when model/data
  artifacts and resources are available.
- Require terminal completion evidence, positive per-round training step
  counts, and finite source-backed evaluation metrics.
- For more than one round, verify metrics advance and checkpoint behavior
  matches `restore_state`.
- For DDP, run a reduced two-process test when available; otherwise report that
  distributed execution was not validated.
- Inspect the exported job for `client.py`, `model.py`, dependencies, quoted
  arguments, and absence of private data.

## Report

Report commands, exit codes, terminal status, metric keys/values, parameter
scope, checkpoint mode, process/rank count, exact workspace/export/result paths,
and skipped checks with concrete reasons.
