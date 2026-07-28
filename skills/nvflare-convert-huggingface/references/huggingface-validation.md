# Hugging Face Conversion Validation

Run these checks after the shared validation ladder. Stop at the first failure.

## Static Checks

- Compile generated `client.py`, `model.py`, and `job.py`.
- Verify executable structure with Python AST nodes, not raw source-string
  counts. Comments, docstrings, and examples do not count as calls.
- Confirm exactly one executable `flare.patch(trainer)` call and one
  rank-symmetric `while flare.is_running():` loop.
- Inspect `While.test` separately from `While.body`; `ast.walk(while_node)`
  includes the loop condition. Confirm required patched Trainer calls occur in
  the source-prescribed order. Do not require the loop body's complete call set
  to equal `{"evaluate", "train"}`; source-preserved logging and bookkeeping
  are allowed.
- Confirm the generated FL client cannot bypass `flare.init()` or
  `flare.patch(trainer)` through launch-environment detection; standalone mode,
  when preserved, must use an explicit entry-point parameter.
- Confirm no manual model `receive()`/`send()` path was added.
- Confirm model, Trainer, datasets, and tokenizer are constructed outside the
  FL round loop.
- Confirm generated `train_args` quote configurable model and data paths.
- If generated `job.py` defines local options, confirm it imports the recipe API
  before parsing, constructs `ArgumentParser(allow_abbrev=False)`, and calls
  strict `parse_args()`. Using the shared assertion-wrapper rule, verify both an
  unknown typo and a unique-prefix abbreviation fail; `parse_known_args()` must
  not hide either.
- Pass the final generated `train_args` through the client entry's actual
  argument mechanism and actual dataclass types in parse-only mode; reject every
  unused argument. If the generated client uses `HfArgumentParser`, parse with
  the same project and framework dataclass types that the client constructs; if
  it preserves `argparse` or another parser, use that parser instead. Check the
  installed Transformers/TRL version only for fields claimed to belong to a
  framework `TrainingArguments` or `SFTConfig` base class. Do not reject or
  remove a project-defined subclass field merely because the base class lacks
  it; preserve the field when its source definition is verified and the actual
  parser accepts it.
- Confirm site data remains external to the exported job.

## Import And Contract Checks

- Import the generated model and job modules after declared dependencies are
  installed.
- Construct the selected recipe and let product validation reject unsupported
  arguments.
- Before constructing a model, tokenizer/processor, or dataset, resolve its
  configured identifier to an existing local path or verify that every required
  file is present in the intended cache. Probe remote-style identifiers with
  `local_files_only=True` or the existing offline environment. A Hugging Face
  error that mentions `https://huggingface.co` during an offline probe can mean
  only that the local cache entry is missing; it is not evidence that a network
  request occurred.
- Never recover from an offline/cache-only miss by removing
  `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`, dropping `local_files_only=True`, or
  rerunning the loader online unless the user explicitly requested the
  model/data download. Do not silently substitute a different local checkpoint,
  including a fine-tuned output, because that changes starting semantics. When
  no compatible local artifact exists, keep the generated job as a draft and
  report full-model validation blocked by the missing artifact.
- Pass the same resolved local model path and cache configuration to the server
  model and every client. Do not validate with a cached Hub identifier and then
  export a job that depends on an unverified online lookup.
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

- Before launching multiple CPU clients, estimate server exchange/offload memory
  plus a conservative per-client training-memory bound multiplied by actual
  worker concurrency. Include model copies, gradients, optimizer state,
  activations, dataloaders/data, and framework overhead when they can be bounded.
  If optimizer, activation, or data memory cannot be bounded, report the
  full-model rung as capacity-unverified rather than treating `state_dict` fit
  as sufficient.
- First run a one-round topology smoke test with the requested site count,
  minimal samples, and an explicitly labelled reduced checkpoint only when the
  real model is too large for the host. Treat this as FL-wiring validation only.
- Run the real model only when the complete resource estimate says it is
  feasible. If it is not feasible, report "full-model validation blocked by host
  capacity" and leave the job as a draft; do not call the conversion fully
  validated.
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
- For exported-job validation, use the supported Recipe interface
  `python job.py --export --export-dir <dir>`. Reject generated job-local export
  aliases such as `--export_only` and manual `recipe.export()` branches that only
  run for private flags.
- Inspect the exported job for `client.py`, `model.py`, dependencies, quoted
  arguments, and absence of private data. If `model.py` is server-only, confirm
  it is still packaged despite being referenced by `job.py` rather than called
  from the Trainer path.

## Report

Report commands, exit codes, terminal status, metric keys/values, parameter
scope, checkpoint mode, process/rank count, exact workspace/export/result paths,
and unresolved validation blockers with concrete reasons.
