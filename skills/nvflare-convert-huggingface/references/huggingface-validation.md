# Hugging Face Conversion Validation

Use `../../nvflare-shared/references/validation-evidence.md` for compilation,
Recipe construction, export, package inspection, simulation, terminal evidence,
and reporting. Run these Trainer-specific checks only after the shared ladder
reaches the applicable phase, and stop at the first failure.

## Generated Trainer Contract

- Adapt `../assets/client_with_eval.py`, `../assets/server_model.py`, and
  `../assets/job.py` rather than drafting replacements. Review the generated
  differences and exercise the asset functions directly; do not write one-off
  AST, class-loader, persistor, Recipe-source, or simulator-source probes.
- Confirm the generated client keeps one persistent Trainer, one
  `flare.patch(trainer)` call, no manual model receive/send path, and the
  source-prescribed evaluate/train sequence.
- Keep model, tokenizer/processor, datasets, collator, Trainer arguments, and
  callbacks outside the FL round loop. The generated FL client must always
  reach `flare.init()` and `flare.patch(trainer)`; preserve standalone behavior
  only behind an explicit non-client entry parameter.
- Pass the final generated `train_args` through the client entry's actual
  argument mechanism and actual dataclass types in parse-only mode; reject every
  unused argument. When the generated client uses `HfArgumentParser`, parse with
  the same project and framework dataclass types. When it preserves `argparse`
  or another parser, use that parser instead.
- Run intentional typo and abbreviation rejection cases through the shared
  assertion-wrapper rule. `HfArgumentParser.parse_args_into_dataclasses()` may
  raise `ValueError` for unused arguments; accept it only when its diagnostic
  names the rejected argument and the wrapper itself exits zero.
- Version-check only fields claimed to belong to a framework
  `TrainingArguments` or `SFTConfig` base class. Preserve a project-defined
  subclass field when its source definition is verified and the actual parser
  accepts it.
- For the default in-process executor, require unquoted whitespace-free
  `train_args` values. Do not assume shell parsing or call internal command
  splitters. If a required value contains whitespace and no documented
  structured argument surface exists, fail closed.

## Hugging Face Artifacts And Compatibility

- Before constructing a model, tokenizer/processor, or dataset, resolve its
  configured identifier to an existing local path or verify that every required
  file is present in the intended cache. Probe remote-style identifiers with
  `local_files_only=True` or the existing offline environment. An error that
  mentions `https://huggingface.co` during an offline probe can mean only that
  the local cache entry is missing; it is not evidence of a network request.
- Never recover from an offline/cache-only miss by removing
  `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`, dropping `local_files_only=True`, or
  rerunning online unless the user explicitly requested the download. Do not
  substitute another local checkpoint. Keep the job as a draft and report
  full-model validation blocked when the required artifact is unavailable.
- Pass the same resolved local model path and cache configuration to the server
  model and every client. Do not validate with a cached Hub identifier and
  export a job that depends on an unverified online lookup.
- Verify `flare.patch()` accepts the actual Trainer configuration. Reject or
  report DeepSpeed, FSDP, best-model-at-end, save-only-model, prebuilt
  optimizer/scheduler, and checkpoint/restore combinations identified in
  `huggingface-state-and-distributed.md`.

## Parameter Checks

- Instantiate the adapted `server_model.ServerModel` and the source Trainer
  model factory with the same local constructor values, then compare their
  state-dict key sets and shapes without training. Do not inspect NVFLARE
  persistors or class loaders; the exported job and final simulation validate
  product-side construction.
- Require exact full-model key agreement after any documented prefix
  transformation. For PEFT or auxiliary trainable models, load
  `huggingface-state-and-distributed.md` and run its adapter/ownership checks.
- Confirm tensor-native exchange only when selected by the capability profile.

## Standard Execution Checks

- Follow the shared compile, construction, export, package-inspection, and one
  final foreground simulation path. Do not add separate Recipe-method,
  argument-transport, simulator-process, or export-dispatch probes when the
  maintained job asset compiles and the public validation rungs pass.
- Require terminal completion evidence, positive per-round training step
  counts, and finite source-backed evaluation metrics for the claimed stage.
- Confirm the requested budget appears once, metrics advance across requested
  rounds, and server artifacts contain the selected metric.

## Conditional Execution Checks

- When the real model may exceed host capacity, estimate server and concurrent
  client memory before simulation. Capability-check optional diagnostics such
  as `free` or `nvidia-smi`; missing diagnostics mean capacity evidence is
  unavailable, not conversion failure. Use a reduced one-round topology smoke
  only when the real model cannot fit, keep the generated default unchanged,
  and report full-model validation blocked.
- After partial aggregation, unexplained disconnect, or exit `-9`, inspect logs
  and either reduce workload with a changed causal factor or report a resource
  blocker. Do not retry the same payload with guessed concurrency settings, and
  make at most one expensive real-model retry.
- For non-default checkpoint/restore behavior, PEFT, auxiliary trainable
  models, or DDP, load `huggingface-state-and-distributed.md` and run only its
  applicable checks.

## Report

Report commands, exit codes, terminal status, metric keys/values, parameter
scope, checkpoint mode, process/rank count, exact workspace/export/result paths,
and unresolved validation blockers with concrete reasons.
