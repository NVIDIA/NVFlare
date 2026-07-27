# Hugging Face Conversion Validation

Run these checks after the shared validation ladder. Stop at the first failure.

## Static Checks

- Compile generated `client.py`, `model.py`, and `job.py`.
- Confirm one `flare.patch(trainer)` call and one rank-symmetric
  `flare.is_running()` loop.
- Confirm the generated FL client cannot bypass `flare.init()` or
  `flare.patch(trainer)` through launch-environment detection; standalone mode,
  when preserved, must use an explicit entry-point parameter.
- Confirm no manual model `receive()`/`send()` path was added.
- Confirm model, Trainer, datasets, and tokenizer are constructed outside the
  FL round loop.
- Confirm generated `train_args` quote configurable model and data paths.
- Pass the final generated `train_args` through the client entry's actual
  `HfArgumentParser` in parse-only mode. Do not start simulation unless every
  argument is consumed. Do not inject a `TrainingArguments` field that is absent
  from the installed Transformers version.
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

- Before launching multiple CPU clients, estimate exchanged `state_dict` bytes
  and available container memory. If the real model/data workload is too large
  for the host, keep the generated job's default model unchanged and label any
  reduced-checkpoint run as validation-only.
- First run a one-round topology smoke test with the requested site count,
  minimal samples, and an explicitly labelled reduced checkpoint only when the
  real model is too large for the host. Treat this as FL-wiring validation only.
- Run the real model only when the resource estimate says it is feasible. If it
  is not feasible, report "full-model validation blocked by host capacity" and
  leave the job as a draft; do not call the conversion fully validated.
- Require terminal completion evidence, positive per-round training step counts,
  and finite source-backed evaluation metrics for whichever validation stage is
  being claimed.
- After a partial aggregation, unexplained client disconnect, or exit `-9`,
  inspect logs and either reduce the model/data workload with a changed causal
  factor or report a resource blocker. Do not retry the same large payload with
  guessed concurrency settings. Make at most one expensive real-model retry.
- For more than one round, verify metrics advance and checkpoint behavior
  matches `restore_state`.
- For DDP, run a reduced two-process test when available; otherwise report that
  distributed execution was not validated.
- Inspect the exported job for `client.py`, `model.py`, dependencies, quoted
  arguments, and absence of private data. If `model.py` is server-only, confirm
  it is still packaged despite being referenced by `job.py` rather than called
  from the Trainer path.

## Report

Report commands, exit codes, terminal status, metric keys/values, parameter
scope, checkpoint mode, process/rank count, exact workspace/export/result paths,
and unresolved validation blockers with concrete reasons.
