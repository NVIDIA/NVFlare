# Lightning Job Validation Notes

Use `../../_shared/validation-evidence.md` for generic validation status,
commands, blockers, and evidence reporting. Use
`../../_shared/metrics-and-artifact-reporting.md` for final metrics, round
metrics, model artifact paths, and missing-evidence reporting. This file only
covers Lightning-specific validation checks.

## Validate In Order

1. Install dependencies first through `../../_shared/dependency-install.md`,
   using `uv pip` when available, before importing the user's Lightning code.
2. Run local SimEnv validation with `python job.py`; follow
   `../../_shared/runtime-output-guidance.md` for workspace location.
3. Wait for terminal completion according to
   `../../_shared/validation-evidence.md`; Lightning simulations can run longer
   than plain PyTorch, so scheduled wakeups or progress logs are not success
   evidence. If the run times out, report it as blocked or timed out with the
   current server/client log evidence.
4. Validate export through the shared lifecycle when export is in scope.
5. Report the declared primary/global metric scalar when one exists.

## Lightning-Specific Checks

- Confirm `flare.patch(trainer)` is applied to the same `Trainer` instance used
  for fit/validate/test, and that the patched trainer, not manual `FLModel`
  code, performs model exchange.
- Confirm no `input_model` returned by `flare.receive()` is passed into
  `Trainer` methods.
- Confirm the `LightningModule` constructed on the client matches the recipe's
  server-side model constructor arguments and state-dict shapes.
- Confirm callbacks, loggers, and checkpoint callbacks still run after patching
  and do not break the FL round loop.
- Confirm validation metrics are exposed as scalars (for example through
  `self.log(...)` in the `LightningModule`) so aggregation recipes can write
  server-side metric artifacts.
- For data-prep changes, confirm the `LightningDataModule` receives the
  generated per-site path or arguments rather than hard-coded global paths.

## Known SimEnv Limitations

- SimEnv runs sites in a single local environment; multi-GPU and DDP behavior is
  validated separately (see `lightning-ddp-and-tracking.md`). A single-process
  SimEnv run validates conversion structure, not distributed scaling.
- Treat synthetic or smoke-test data runs as structural validation, not as
  meaningful accuracy evidence, unless the user supplies expected metrics.
- Report Lightning-specific blockers such as a trainer that cannot be patched, a
  callback or logger that fails inside the round loop, checkpoint loading that
  conflicts with the patched model exchange, or metrics that are not logged as
  scalars.
