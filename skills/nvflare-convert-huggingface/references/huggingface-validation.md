# Hugging Face Conversion Validation

Use `../../nvflare-shared/references/validation-evidence.md` for compilation,
Recipe construction, export, package inspection, simulation, terminal evidence,
and reporting. Run these Trainer-specific checks only after the shared ladder
reaches the applicable phase, and stop at the first failure.

## Generated Trainer Contract

- Adapt `../assets/client_with_eval.py` rather than drafting a new patch/round
  loop. Confirm by direct review that the generated client keeps one persistent
  Trainer, one `flare.patch(trainer)` call, no manual model receive/send path,
  and the source-prescribed evaluate/train sequence. Do not write a one-off AST
  or Recipe-source introspection program to re-prove the maintained template.
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

- Extract server and Trainer exchange key sets without training.
- Require exact full-model or adapter-key agreement after the documented prefix
  transformation.
- For PEFT, verify both sides use the same adapter configuration and only
  adapter parameters are exchanged.
- Confirm PyTorch exchange format for BF16/FP16 or other dtype-sensitive models.

## Execution Checks

- Before launching multiple CPU clients, estimate server exchange/offload
  memory plus a conservative per-client training-memory bound multiplied by
  actual worker concurrency. Include model copies, gradients, optimizer state,
  activations, dataloaders/data, and framework overhead when they can be
  bounded. If optimizer, activation, or data memory cannot be bounded, report
  the full-model rung as capacity-unverified rather than treating `state_dict`
  fit as sufficient.
- Capability-check optional host diagnostics such as `free` or `nvidia-smi`
  before invoking them, for example with `shutil.which()` or `command -v`. Run
  them separately from correctness checks, or guard their absence. A missing
  diagnostic means capacity evidence is unavailable, not conversion failure.
- Use a one-round topology smoke test with the requested site count, minimal
  samples, and an explicitly labelled reduced checkpoint only when the real
  model is too large for the host. This validates FL wiring only; leave the
  generated job's default model unchanged. Otherwise run the requested real
  model and round count once.
- Require terminal completion evidence, positive per-round training step
  counts, and finite source-backed evaluation metrics for the claimed stage.
- Calculate the expected per-round optimizer-step delta from `max_steps` or
  from `num_train_epochs`, real dataloader length, and gradient accumulation.
  Confirm the budget is not duplicated in patch `local_steps`/`local_epochs`.
  Verify `NUM_STEPS_CURRENT_ROUND` or equivalent round evidence against that
  delta. Under `restore_state=True`, a progress denominator or `args.max_steps`
  equal to the per-round delta times total FL rounds is the expected cumulative
  scheduler target, not multiplied local work.
- After partial aggregation, unexplained disconnect, or exit `-9`, inspect logs
  and either reduce workload with a changed causal factor or report a resource
  blocker. Do not retry the same payload with guessed concurrency settings, and
  make at most one expensive real-model retry.
- For multiple rounds, verify metrics advance and checkpoint behavior matches
  `restore_state`. For DDP, run a reduced two-process test when available;
  otherwise report that distributed execution was not validated.

## Report

Report commands, exit codes, terminal status, metric keys/values, parameter
scope, checkpoint mode, process/rank count, exact workspace/export/result paths,
and unresolved validation blockers with concrete reasons.
