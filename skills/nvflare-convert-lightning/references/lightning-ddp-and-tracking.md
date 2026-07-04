# Lightning DDP And Experiment Tracking

Use this reference only for Lightning DDP/multi-GPU work or experiment-tracking
work. Single-GPU or single-process Lightning training does not need it.

## DDP Execution Model

Treat Lightning DDP/multi-process evidence as a high-impact runtime fact, not
as a skill-owned mapping to a recipe argument. Run
`nvflare recipe show <recipe-name> --format json` and inspect the public
parameter descriptions or capability metadata for an option explicitly
documented for the detected Lightning launch model. The presence of an
external-process-looking parameter name alone is insufficient evidence.

Use the documented value only when that public contract establishes the
mapping. If the recipe does not expose such a capability, or its public
description is ambiguous, ask in interactive mode or fail closed in unattended
mode and report the source DDP evidence plus the missing product surface. Apply
the same public-contract check to single-process multi-GPU strategies; do not
encode a `dp` exception table in this reference.

## Rank-Synchronized Round Loop

When the product-documented path launches multiple ranks and defines rank 0 as
the FL server communicator, all ranks must agree on whether to continue.
Broadcast `flare.is_running()` from rank 0 before each round, and let the
patched trainer callback receive and broadcast the model during
`validate`/`fit`:

```python
flare.patch(trainer)

while True:
    is_running = flare.is_running()
    is_running = trainer.strategy.broadcast(is_running, src=0)
    if not is_running:
        break

    trainer.validate(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)  # when test evidence is requested/available
```

Do not insert an unguarded `flare.receive()` on every rank for metadata. Outside
the patched callback, rank 0 can fetch the cached `FLModel`; non-zero ranks do
not get the task/model object. If user code outside the patched trainer needs
round/task metadata on all ranks, derive a small serializable value on rank 0 and
broadcast that value before using it:

```python
round_info = {}
if trainer.global_rank == 0:
    input_model = flare.receive()
    round_info = {
        "current_round": input_model.current_round if input_model else None,
        "meta": dict(input_model.meta or {}) if input_model and input_model.meta else {},
    }
round_info = trainer.strategy.broadcast(round_info, src=0)
```

As with single-process training, the patched trainer owns model exchange. Do not
pass `input_model` into `Trainer` methods and do not add a manual `flare.send`.
In DDP, rank-0 communication with the FL server is either handled by the patched
callback path or by explicit rank-0 guarded metadata code like the snippet above;
non-zero ranks should read broadcast values, not `input_model`.

### DDP validation metric capability

Do not infer server-side metric delivery from a local `trainer.validate(...)`
call or from executor internals. Check the recipe's public capability metadata
and validation evidence for the selected launch path. Claim server-side
per-round metrics only when that public surface and an observed validation
artifact establish delivery; otherwise report local metrics separately and
surface server-side delivery as an unverified limitation (or a blocker when the
request requires it in unattended mode).

## GPU/CPU Fallback

- Keep the user's `accelerator`/`devices` settings; do not silently force CPU.
- When the validation environment has no GPU, report the limitation and validate
  conversion structure on CPU or a reduced device count instead of changing the
  user's training intent.

## Experiment Tracking

Enable remote tracking only when the user explicitly asks for it or explicitly
approves the external effects and destinations. Existing source logger or
callback configuration is evidence of intent, not approval.

- Preserve local-only loggers such as a local `TensorBoardLogger` when their
  output stays in the recorded private run directory.
- Treat remote `MLFlowLogger`, WandB/Comet-style clients, upload callbacks, and
  custom or unknown loggers as network-capable. Keep them disabled during
  validation unless the user explicitly approves them. In unattended mode,
  disable them and retain denied network egress; if the trainer cannot run
  without them, report validation as blocked rather than opening egress.
- Hand metrics to FLARE through `add_experiment_tracking` or the FLARE client
  logger when the workflow needs server-side or streamed tracking. The canonical
  client-facing shortcut is `flare.logger()` (with
  `import nvflare.client.lightning as flare`); the class is
  `nvflare.app_opt.lightning.loggers.client_logger.ClientLogger`. Do not import
  it as `nvflare.app_opt.lightning.loggers.ClientLogger`, which is not exported.
- The client logger streams metrics through the FL client; it is not a full
  replacement for a standalone tracking server. State this limitation rather
  than promising parity with a dedicated tracking backend.

Follow `../../nvflare-shared/references/metrics-and-artifact-reporting.md` for reporting metric
and artifact paths and for missing-evidence reporting.
