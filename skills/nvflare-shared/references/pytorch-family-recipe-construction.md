# PyTorch-Family Recipe Construction

Use this reference after selecting a PyTorch-family recipe and running
`nvflare recipe show <recipe-name> --format json`. It is the canonical owner of
recipe-capability handling for both plain PyTorch and PyTorch Lightning.

## Derive A Capability Profile

The JSON returned by `recipe show` contains `data.parameters`, a list of
parameter objects whose `name` field is the public constructor keyword. Build
the exposed-name set from those fields. Treat a missing or malformed
`data.parameters` value as a failed capability check rather than guessing from
another recipe.

Only pass a recipe keyword when its name is in the exposed-name set. Do not copy
the FedAvg constructor shape to Cyclic, FedEval, Swarm, or another recipe.

## Tensor-Native Transport

When `server_expected_format` is exposed, pass
`server_expected_format=ExchangeFormat.PYTORCH`. This selects tensor-native
transport. When tensor-native transport was selected, call
`recipe.add_decomposers(["nvflare.app_opt.pt.decomposers.TensorDecomposer"])`
after any `set_per_site_config(...)` call and before export or execution.

If `server_expected_format` is absent, preserve the recipe's documented
transport default and do not infer tensor-native transport from its framework
name. Register `TensorDecomposer` only when another public recipe surface
explicitly selects tensor-native transport. If the user requires tensor-native
transport and the selected recipe has no public way to select it, report a
recipe-capability gap instead of passing unsupported keywords or switching
recipes.

`pytorch-model-exchange.md` is the canonical owner of framework payload and
state-dict rules.

## Server Tensor Disk Offload

Disk offload is a server memory optimization, not a model-exchange format. It
causes incoming streamed tensors to be downloaded to server-side temporary files
and materialized lazily during aggregation instead of being deserialized into
memory immediately, reducing peak server memory pressure and OOM risk.

When tensor-native transport was selected and `enable_tensor_disk_offload` is
exposed, pass `enable_tensor_disk_offload=True`. NVFLARE activates this
optimization only with `server_expected_format=ExchangeFormat.PYTORCH`;
otherwise it warns and treats the setting as a no-op. Never pass the keyword to
a recipe that does not expose it.

If the user requires disk offload and the selected recipe cannot expose both
tensor-native transport and `enable_tensor_disk_offload`, report a
recipe-capability gap instead of claiming that offload is enabled.

## Transfer Mode

When `params_transfer_type` is exposed, choose the mode that matches the user's
intent: `FULL` sends whole models and `DIFF` sends model differences. Do not pass
it when absent or infer a transfer mode from another recipe.

## Execution Mode Is Process-Based

Select execution mode from process-spawning or distributed-launch evidence, not
from GPU count:

- CPU, single-GPU, and single-process multi-GPU `torch.nn.DataParallel` stay
  in-process; leave `launch_external_process` unset.
- `DistributedDataParallel`, `torchrun`, `torch.distributed`, Lightning DDP
  strategies, or another source launch that creates distributed worker
  processes requires `launch_external_process=True`.

Before setting it, confirm `launch_external_process` is exposed. Also confirm
the recipe exposes `command`, `launcher`, or another public launch surface that
can preserve required distributed-launch arguments. If either capability is
missing, ask or fail closed rather than dropping the source process model.
Multiple requested GPUs without evidence of how the source uses them is not
enough to choose a process model; inspect the source, then ask or fail closed if
it remains ambiguous.

## Best-Model Filename

When customizing a best-model artifact name, pass `best_model_filename` only
when the selected recipe exposes it. Do not also pass `save_filename`; it is a
deprecated alias, and conflicting values make recipe construction fail. Omit
both when the default artifact name is acceptable. If customization is required
but `best_model_filename` is absent, report the capability gap.

## Best-Model Metric

Only configure a source-derived `key_metric` when the recipe exposes it and the
selected execution path delivers that exact metric to the server. A local
evaluation call, `self.log`, or client log does not by itself prove server
delivery. If delivery is unavailable or unverified, do not pass a source-derived
`key_metric` or claim server-side best-model selection; report the execution-path
limitation instead. The Lightning conversion guidance documents the explicit
training-result metric bridge and its DDP validation requirements.

When both preconditions hold, select a source-backed metric whose larger value
means a better model. Its name must exactly match one key delivered by the
client in `FLModel.metrics` (or by the Lightning integration). For example, a
delivered client metric named `f1` uses `key_metric="f1"`.

For a lower-is-better source metric such as loss, send its negated value under a
clear key such as `metrics={"neg_loss": -loss}` and use
`key_metric="neg_loss"`. Do not rely on a recipe default unless the client emits
that exact metric, and ask or fail closed when the metric direction is unclear.
Do not pass `key_metric` when the recipe does not expose it.
