# Tensor Disk Offload

## Objective

Reduce aggregation peak memory for large PyTorch model updates by materializing
incoming streamed tensor payloads on disk and resolving them lazily during
aggregation.

## Scope

- Applies to streamed **PyTorch tensor** payloads handled by `TensorDecomposer`.
- Controlled by `enable_tensor_disk_offload` in a supported receiving workflow/controller config:
  FedAvg (including FedProx), FedOpt, SCAFFOLD, or Swarm.
- Default is `False` (legacy in-memory behavior).
- If model updates are converted to NumPy before transport, tensor disk offload is not engaged.

Tensor streaming, ref pass-through, and tensor disk offload are separate
behaviors. The sender's `TensorDownloadable` remains memory-backed while it
serves a transport ref. An intermediate process may preserve that ref without
downloading it. Disk offload applies when the receiving aggregation workflow
terminates the ref and downloads its tensor chunks. It does not spool the
trainer's model or training result to disk at the source, and it does not reduce
the memory required to load or train the model in the trainer.

## How To Enable

FedAvg:

- `nvflare/recipe/fedavg.py` -> `FedAvgRecipe(..., enable_tensor_disk_offload=True)`
- `nvflare/app_opt/pt/recipes/fedavg.py` -> PT recipe forwards the same flag
- `nvflare/app_common/workflows/fedavg.py` -> `FedAvg(..., enable_tensor_disk_offload=True)`
- PyTorch FedProx uses the same FedAvg recipe and server aggregation path

Other server-controlled workflows:

- `nvflare/app_opt/pt/recipes/fedopt.py` -> `FedOptRecipe(..., enable_tensor_disk_offload=True)`
- `nvflare/app_opt/pt/recipes/scaffold.py` -> `ScaffoldRecipe(..., enable_tensor_disk_offload=True)`
- both require `server_expected_format=ExchangeFormat.PYTORCH`; custom configurations can enable
  the same setting directly on `ScatterAndGather` or `Scaffold`

Swarm/CCWF:

- `nvflare/app_opt/pt/recipes/swarm.py` -> use
  `SwarmLearningRecipe(aggregation_format=ExchangeFormat.PYTORCH, enable_tensor_disk_offload=True)`
- `nvflare/app_common/ccwf/ccwf_job.py` -> use
  `SwarmClientConfig(..., enable_tensor_disk_offload=True)` for custom Job API configurations
- `nvflare/app_common/ccwf/swarm_client_ctl.py` owns an offload root on each
  eligible client because the aggregation role moves between clients, but
  enables disk materialization only for terminal aggregation downloads

If no active Cell is available, the offload context is not enabled and the runtime falls back to in-memory download.

## Data Flow

```
TensorDownloadable chunks
        |
        v
TensorDecomposer.download()
  - enable_tensor_disk_offload=False -> deserialize in memory
  - enable_tensor_disk_offload=True  -> write safetensors temp files
        |
        v
LazyTensorDict
        |
        v
ViaDownloaderDecomposer.recompose()
        |
        v
Lazy refs in payload tree
        |
        +--> aggregator consumes lazy refs (materialize on demand)
```

Server-controlled FedAvg/FedProx, FedOpt, and SCAFFOLD workflows install their
disk-offload setting on the receiving server Cell. Swarm does not enable disk
offload globally on its client Cells because the same Cells also receive
learner tasks and final-result broadcasts. Instead, the selected aggregation
controller puts the setting and its job-scoped root in the terminal result
download's FOBS decode context. The destination therefore does not depend on
mutable Cell-wide state, and non-aggregation deliveries remain ordinary
in-memory tensors.

## Runtime Behavior

### FedAvg

In `nvflare/app_common/workflows/fedavg.py`:

- custom aggregators receive `result.params` as-is
- with `enable_tensor_disk_offload=True`, lazy refs are passed through directly
- built-in weighted aggregation materializes per tensor inside `WeightedAggregationHelper.add()`
  and relies on lazy-ref object lifetime / GC for temp-resource cleanup

The built-in weighted path remains lazy-friendly and memory-efficient.

### Swarm/CCWF

`ClientAPIExecutor` preserves the Cell/FOBS large-payload references between an
external trainer and the CCWF controller. When a training result is sent to a
remote aggregation client, `SwarmClientController` requests PASS_THROUGH on that
message. The aggregation controller therefore receives refs instead of
materializing the tensors inside the Cell receive callback, and explicitly
resolves them with its job-scoped disk root. A result returned by a local
external trainer also crosses a Cell boundary as a ref and is resolved by the
local aggregation controller with the same disk root. In both cases tensor disk
offload yields lazy tensor refs, and the built-in
`InTimeAccumulateWeightedAggregator` materializes one tensor at a time.

The Swarm controller owns the decision to preserve or resolve transport refs.
On a non-aggregation client it keeps refs when the local learn executor is
`ClientAPIExecutor(execution_mode="external_process")`, making the external
trainer their single consumer. If that external-process site is also the selected
aggregator, the controller resolves the task once for its aggregation base model
and passes the same in-memory payload to the trainer. This avoids two consumers
racing a one-receiver download transaction. The controller also resolves tasks
into memory for `in_process`, `attach`, and non-`ClientAPIExecutor` learners.
This conservative fallback supports jobs where sites use different learner
execution modes. Disk-backed aggregation refs remain local to the aggregation
client and are never passed to a learner.

The external trainer process's Cell is not configured as a disk-offload
receiver. It is the terminal consumer of an incoming learn task, and the Client
API materializes ordinary in-memory tensors for model loading. For an outgoing
training result, `TensorDownloadable` still holds the tensors in trainer memory
while serving a transport ref. `ClientAPIExecutor` and the client job preserve
and route that ref; the selected aggregation controller decides whether its
terminal download is materialized on disk. Source-side tensor spooling is not
part of this feature.

If an in-process learner is also the local aggregation client, its own result is
already an in-memory object with no transport ref to preserve. That one local
contribution remains in memory; remote contributions still use terminal
aggregation-CJ disk offload.

`SwarmLearningRecipe` defaults to NumPy exchange for compatibility. Disk offload
therefore requires `aggregation_format=ExchangeFormat.PYTORCH`; streamed
NumPy arrays are not handled by `TensorDecomposer`.

## Custom Aggregator Contract

When a custom aggregator is used, payload params may contain lazy refs (duck-typed object with `materialize()`).

Custom aggregators are responsible for:

1. materializing refs when tensor math is required
2. releasing lazy-ref object references after use so temp resources can be reclaimed

## Temp File Lifecycle

- Each workflow creates a job-scoped offload root (`nvflare_tensor_offload_<job>_*`),
  with safetensors download directories beneath it.
- Temp dir selection follows Python `tempfile` behavior (`TMPDIR` / OS default, typically `/tmp`).
- In containerized deployments, `/tmp` may be tmpfs (RAM-backed); set `TMPDIR` to a disk-backed mount to realize memory offload benefits.
- `LazyTensorDict` owns a shared `_TempDirRef`; each lazy ref keeps this reference alive.
- Lazy download directories are reclaimed when their refs are released, with GC as
  a fallback.
- FedAvg-style workflows restore the prior FOBS context and remove their
  job-scoped root when the workflow exits.
- Swarm keeps its job-scoped root and tensor-forwarding route through workflow
  finalization. At job `END_RUN`, it gives the controller-owned learning and
  aggregation threads a bounded drain window, then removes the root and its
  remaining contents.

## Failure Behavior

- Download failures trigger `DiskTensorConsumer.download_failed(...)`, which removes the temp dir.
- Invalid safetensors payload/header parsing fails fast and bubbles up as a download-consume error.
- Existing in-memory download path remains unchanged when offload is disabled.

## Design-Relevant Files

- `nvflare/app_opt/pt/decomposers.py`
- `nvflare/app_opt/pt/lazy_tensor_dict.py`
- `nvflare/app_opt/pt/tensor_downloader.py`
- `nvflare/fuel/utils/fobs/decomposers/via_downloader.py`
- `nvflare/app_common/workflows/fedavg.py`
- `nvflare/app_common/ccwf/swarm_client_ctl.py`
- `nvflare/app_common/ccwf/ccwf_job.py`
- `nvflare/recipe/fedavg.py`
- `nvflare/app_opt/pt/recipes/swarm.py`

## Test Coverage

- `tests/unit_test/app_common/workflow/fedavg_test.py`
- `tests/unit_test/app_common/ccwf/test_swarm_tensor_disk_offload.py`
- `tests/unit_test/recipe/swarm_recipe_test.py`
- `tests/unit_test/app_opt/pt/test_lazy_tensor_dict.py`
- `tests/unit_test/app_opt/pt/test_disk_tensor_consumer.py`
- `tests/unit_test/app_common/aggregators/weighted_aggregation_helper_test.py`
- `tests/unit_test/private/fed/server/server_runner_test.py`
- `tests/stress_test/fedavg_large_model/fedavg_stress_test.py`
