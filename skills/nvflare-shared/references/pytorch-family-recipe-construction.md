# PyTorch-Family Recipe Construction

Use this reference after selecting a PyTorch-family recipe and running
`nvflare recipe show <recipe-name> --format json`. It is the canonical owner of
recipe-capability handling for both plain PyTorch and PyTorch Lightning.

## Derive A Capability Profile

Build a set of parameter names from the returned `data.parameters` entries.
Only pass a recipe keyword when its name is in that set. Do not copy the FedAvg
constructor shape to Cyclic, FedEval, Swarm, or another recipe.

Apply the tensor profile in this order:

1. When `server_expected_format` is exposed, pass
   `server_expected_format=ExchangeFormat.PYTORCH`. This selects tensor-native
   exchange.
2. When `enable_tensor_disk_offload` is exposed, pass
   `enable_tensor_disk_offload=True`. Never pass it to a recipe that does not
   expose it.
3. When tensor-native exchange was selected, call
   `recipe.add_decomposers(["nvflare.app_opt.pt.decomposers.TensorDecomposer"])`
   after any `set_per_site_config(...)` call and before export or execution.

If `server_expected_format` is absent, preserve the recipe's documented
exchange default and do not infer tensor-native exchange from its framework
name. Register `TensorDecomposer` only when another public recipe surface
explicitly selects a tensor-native exchange. If the user requires tensor-native
exchange and the selected recipe has no public way to select it, report a
recipe-capability gap instead of passing unsupported keywords or switching
recipes.

Keep outbound plain-PyTorch model state as `torch.Tensor` values; never add a
manual NumPy conversion to compensate for a missing recipe capability.
`pytorch-model-exchange.md` owns the framework payload and state-dict rules.

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
