# PyTorch-Family Recipe Parameters

Use this reference when writing `job.py` for any PyTorch-family converter:
plain PyTorch, PyTorch Lightning, or Hugging Face Trainer. The training
framework controls the client-side loop; these recipe parameters have the same
meaning across the family.

## Model Selection Metric

`FedAvgRecipe.key_metric` configures the automatic model selector. The selected
metric must be the exact key sent by the client metrics path, and larger values
must mean a better model.

- Plain PyTorch clients send `FLModel.metrics`.
- Lightning clients expose metrics logged through Lightning, usually
  `self.log(...)` and `trainer.callback_metrics`.
- Hugging Face clients expose the keys returned by `trainer.evaluate()`. Prefer
  preserving source metric names when the generated evaluation call can emit
  them; if Trainer prefixes a metric such as `accuracy` to `eval_accuracy`,
  configure the server to the exact emitted key and report the mapping.

For source-backed lower-is-better metrics that the converted client controls,
emit an explicitly negated companion metric and select that higher-is-better
key, for example `neg_loss` or `eval_neg_wer`. Preserve the original metric too
when it is useful for reporting.

Do not select a raw loss as `key_metric`. If best-model selection is required
but the only available metric is a framework-generated loss that cannot be
source-backed and safely transformed by the conversion, ask for a selection
metric or fail closed. Use `key_metric=""` only when best-model selection is not
requested; it omits the automatic model selector.

## External Process Launch

Leave `launch_external_process` unset for single-process training so the recipe
uses its default in-process executor. Set `launch_external_process=True` only
when the source requires a separate worker process, such as DDP/torchrun-style
multi-process execution, a scheduler-launched script, or an explicit user
request for external-process execution.

Before setting it, confirm the selected recipe exposes
`launch_external_process` with `nvflare recipe show <recipe-name> --format
json`. If external launch is required but the recipe does not expose it, ask in
interactive mode or fail closed in unattended mode.

Single-process multi-GPU `DataParallel` (`dp`) stays in-process; leave
`launch_external_process` unset.

## Simulator Concurrency

Do not reduce simulator `num_threads` below the requested client count as a
speculative memory workaround for an in-process Client API executor with tensor
offload. Use a lower thread count only when the selected execution mode
documents that behavior and a bounded smoke test verifies it. After partial
aggregation, unexplained client disconnect, or process loss, inspect logs and
reduce the model/data workload or report a resource blocker. Do not retry the
same large payload with guessed concurrency settings.

## Tensor Disk Offload

Set `enable_tensor_disk_offload=True` when the selected recipe exposes it and
the job exchanges PyTorch tensors. When the recipe also exposes
`server_expected_format`, pair disk offload with
`server_expected_format=ExchangeFormat.PYTORCH`.

Do not enable disk offload with a NumPy exchange format. Under NumPy exchange it
is a warned no-op, not a memory-saving feature.
