# Lightning DDP And Experiment Tracking

Use this reference only for Lightning DDP/multi-GPU work or experiment-tracking
work. Single-GPU or single-process Lightning training does not need it.

## DDP Execution Model

Lightning DDP maps to the external-process executor exactly like plain-PyTorch
DDP. Lightning's process-spawning strategies — `ddp` and its variants such as
`ddp_spawn`, and any other strategy that launches one worker process per device
through a torchrun-style / `torch.distributed` launch — run the training script
as multiple worker processes, and distributed workers cannot run inside an
in-process executor. So DDP/multi-process evidence maps to
`launch_external_process=True`.

The one exception is single-process multi-GPU DataParallel (`dp`), which runs in
a single process and stays in-process like single-GPU training; leave
`launch_external_process` unset so the recipe applies its own default.

When the source shows a DDP-family strategy, confirm the selected recipe exposes
`launch_external_process` with `nvflare recipe show <recipe-name> --format json`,
then set it to `True`. If the recipe does not expose it, ask in interactive mode
or fail closed in unattended mode, reporting the DDP evidence and the missing
product surface.

## Rank-Synchronized Round Loop

Under DDP, only rank 0 communicates with the FL server, so all ranks must agree
on whether to continue. Broadcast `flare.is_running()` from rank 0 before each
round:

```python
flare.patch(trainer)

while True:
    is_running = flare.is_running()
    is_running = trainer.strategy.broadcast(is_running, src=0)
    if not is_running:
        break

    input_model = flare.receive()  # optional, only when round/task metadata is needed
    trainer.validate(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)  # when test evidence is requested/available
```

As with single-process training, the patched trainer owns model exchange. Do not
pass `input_model` into `Trainer` methods and do not add a manual `flare.send`.
The `flare.receive()` above only exposes round/task metadata and progression; it
is not model exchange, so calling it on every rank is safe. Rank-0-only
communication with the FL server is handled by the patched trainer, not by
guarding `flare.receive()`.

### DDP validation metrics are not delivered to the server by default

DDP requires the external-process launch (`launch_external_process=True`), which
runs the script under `PTClientAPILauncherExecutor`. That executor defaults to
`train_with_evaluation=False`, and the recipe's `ScriptRunner` does not expose a
switch to change it, so the pre-`fit` `trainer.validate(...)` metrics are **not**
attached to the training result. The `validate` call above still runs locally
(useful for local logging), but server-side model selection and per-round metric
reporting do **not** receive those metrics under the default DDP path.

Do not promise per-round server-side validation metrics for a DDP conversion.
Report this as a recipe limitation, and only claim server-side round metrics when
the user opts into an advanced, non-`ScriptRunner` configuration that constructs
the launcher executor with `train_with_evaluation=True`. Otherwise surface the
limitation (or a blocker in unattended mode) instead of promising metrics the
default recipe path cannot deliver.

## GPU/CPU Fallback

- Keep the user's `accelerator`/`devices` settings; do not silently force CPU.
- When the validation environment has no GPU, report the limitation and validate
  conversion structure on CPU or a reduced device count instead of changing the
  user's training intent.

## Experiment Tracking

Enable tracking only when the user asks for it or the source code already uses a
Lightning logger.

- Preserve existing Lightning loggers such as `TensorBoardLogger` or
  `MLFlowLogger`.
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
