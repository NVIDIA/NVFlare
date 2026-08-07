# PyTorch-Family Recipe Construction

Use this reference after selecting a PyTorch-family recipe and running
`nvflare recipe show <recipe-name> --format json`. It is the canonical owner of
recipe-capability handling for plain PyTorch, PyTorch Lightning, and Hugging
Face Trainer conversions.

## Derive A Capability Profile

The JSON returned by `recipe show` contains `data.parameters`, a list of
parameter objects whose `name` field is the public constructor keyword. Build
the exposed-name set from those fields. Treat a missing or malformed
`data.parameters` value as a failed capability check rather than guessing from
another recipe.

Only pass a recipe keyword when its name is in the exposed-name set. Do not copy
the FedAvg constructor shape to Cyclic, FedEval, Swarm, or another recipe.

Every recipe construction or construction preflight must include one model
source accepted by the selected recipe, such as `model`, `initial_ckpt`, or
`model_persistor`, and that keyword must be exposed by its capability profile.
For an ordinary conversion, use the explicit model `class_path` and `args`
mapping. Never instantiate an incomplete `FedAvgRecipe` merely to inspect
attributes or discover constructor behavior; use `recipe show` and non-raising
module/class attribute checks for capability discovery.

## Recipe API Boundary

`recipe show` validates only the selected recipe's module, class, and listed
constructor parameters. It does not advertise other symbols from
`nvflare.recipe` or define a separate export environment. For supported local
conversion validation, use `SimEnv` with `recipe.execute(...)`; export through
`python job.py --export --export-dir <dir>`.

Do not guess or directly import additional recipe-adjacent symbols. When the
selected public workflow genuinely requires another symbol, first use a
non-failing module attribute check such as `hasattr`; if it is absent, report a
version or skill-contract gap. Do not replace a failed local public check with
web search or SDK-source discovery.

Validate a recipe model's `class_path` through the public recipe construction,
export inspection, and bounded execution path. Do not import internal class
loader helpers, guess helper names, or inspect implementation source to build a
parallel validation path.

## Client Argument Transport

Treat `train_args` as a recipe-owned argument string, not automatically as a
POSIX shell command. Encode it according to the selected public execution
surface:

- For the default in-process Client API executor, pass whitespace-free values
  as unquoted tokens. Ordinary quote characters may be preserved literally.
- Use shell quoting such as `shlex.quote()` only when the selected documented
  launcher explicitly uses POSIX command tokenization.
- If a required path or identifier contains whitespace and the selected recipe
  exposes no documented structured or per-site argument surface, ask or fail
  closed rather than switching launch modes or inventing an encoding.

Do not import or call internal command-splitting helpers to predict argument
delivery. Validate the exact generated `train_args` end to end through the
selected recipe path and the generated client's actual parser.

## Tensor-Native Transport

Use the workflow-side format keyword exposed by the selected recipe:

- When `aggregation_format` is exposed, pass
  `aggregation_format=ExchangeFormat.PYTORCH`. This names the representation
  consumed by a client-side aggregation workflow such as Swarm.
- Otherwise, when `server_expected_format` is exposed, pass
  `server_expected_format=ExchangeFormat.PYTORCH`. This names the representation
  consumed by a server-side workflow.

Either form selects tensor-native transport. When tensor-native transport was selected, call
`recipe.add_decomposers(["nvflare.app_opt.pt.decomposers.TensorDecomposer"])`
after any `set_per_site_config(...)` call and before export or execution.

If neither workflow-side format keyword is exposed, preserve the recipe's
documented transport default and do not infer tensor-native transport from its
framework name. Register `TensorDecomposer` only when another public recipe
surface explicitly selects tensor-native transport. If the user requires
tensor-native transport and the selected recipe has no public way to select it,
report a recipe-capability gap instead of passing unsupported keywords or
switching recipes.

`pytorch-model-exchange.md` is the canonical owner of framework payload and
state-dict rules.

## Workflow Tensor Disk Offload

Disk offload is an aggregation-workflow memory optimization, not a
model-exchange format. It causes incoming streamed tensors to be downloaded to
temporary files on the aggregation host and materialized lazily during
aggregation instead of being deserialized into memory immediately, reducing
peak memory pressure and OOM risk.

When tensor-native transport was selected and `enable_tensor_disk_offload` is
exposed, pass `enable_tensor_disk_offload=True`. NVFLARE activates this
optimization only when the exposed workflow-side format is
`ExchangeFormat.PYTORCH`; otherwise it warns and treats the setting as a no-op.
Never pass the keyword to a recipe that does not expose it.

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

`launch_once` is the companion Client API knob and is framework-neutral: it
selects whether the external client process is launched once for the whole job
and reused for every task, or relaunched for each task/round. It belongs to the
recipe, not to any framework integration, and applies only when
`launch_external_process=True`. Confirm it is exposed before passing it and keep
the recipe default (`True`) unless the source genuinely requires a fresh process
per task. A framework integration may constrain this value — an integration that
keeps one persistent trainer alive across rounds cannot use `launch_once=False`
— so check the framework skill's delta before overriding the default.

## Simulator Concurrency

Do not reduce simulator `num_threads` below the requested client count as a
speculative memory workaround for an in-process Client API executor with tensor
offload. Use a lower thread count only when the selected execution mode
documents that behavior and a bounded smoke test verifies it. After partial
aggregation, unexplained client disconnect, or process loss, inspect logs and
reduce the model/data workload or report a resource blocker. Do not retry the
same large payload with guessed concurrency settings.

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

When best-model selection is requested for a lower-is-better source metric, send
its negated value under a clear key and select that key. This applies to loss in
every framework, whether the client computes it or the framework's own
evaluation call emits it: send `metrics={"neg_loss": -loss}` and use
`key_metric="neg_loss"`. When the loss is produced by a framework integration
rather than user code, add the negated companion through that framework's
documented metric hook before the value is delivered — the framework skill
names the hook. Never select a raw loss key as though larger values were better,
and do not fail closed merely because loss is the only available metric.

Do not rely on a recipe default when explicitly selecting a source metric unless
the client emits that exact metric, and ask or fail closed when the metric
direction is unclear — not merely because the only available metric is a loss.
Do not pass `key_metric` when the recipe does not expose it.

Resolve model selection to exactly one state before constructing the recipe:

- **Disabled:** Best-model selection is not requested. When the recipe exposes
  `key_metric` and documents empty-string disabling, pass `key_metric=""`.
  Omitting the argument is not disabling when the recipe has a non-empty
  default. If the selected recipe cannot disable selection, report the
  capability gap.
- **Metric:** Best-model selection is requested and an exact higher-is-better
  client metric is available. Pass that non-empty key.
- **Recipe default:** Accept the documented default only deliberately, when the
  client delivers that exact key. Omit `key_metric` and report the resolved
  default; never use this state as a fallback for an unavailable metric.

After export, inspect the server configuration. The disabled state must contain
no active model-selector component. The metric and recipe-default states must
contain a selector with the resolved key. Treat a mismatch, or missing-metric
warnings from a supposedly disabled job, as validation failure.
