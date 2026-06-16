# Lightning DDP And Experiment Tracking

Use this reference only for Lightning DDP/multi-GPU work or experiment-tracking
work. Single-GPU or single-process Lightning training does not need it.

## DDP Execution Model

Lightning DDP and multi-GPU training spawns multiple worker processes. Do not
run distributed workers inside an in-process executor. Use external process
launch such as `launch_external_process=True` on the selected PyTorch recipe so
each site launches its own distributed training process. This external-process
rule is shared PyTorch-family guidance.

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

Follow `../../_shared/metrics-and-artifact-reporting.md` for reporting metric
and artifact paths and for missing-evidence reporting.
