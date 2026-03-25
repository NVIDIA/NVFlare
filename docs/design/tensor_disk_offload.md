# Tensor Disk Offload

## Objective

Reduce server peak memory for large PyTorch FedAvg model updates by streaming tensor payloads to disk and resolving tensors lazily end-to-end.

## Scope

- Applies to streamed **PyTorch tensor** payloads handled by `TensorDecomposer`.
- Controlled by `enable_tensor_disk_offload` in the server-side FedAvg workflow/controller config.
- Default is `False` (legacy in-memory behavior).
- If model updates are converted to NumPy before transport, tensor disk offload is not engaged.

## How To Enable

FedAvg:

- `nvflare/recipe/fedavg.py` -> `FedAvgRecipe(..., enable_tensor_disk_offload=True)`
- `nvflare/app_opt/pt/recipes/fedavg.py` -> PT recipe forwards the same flag
- `nvflare/app_common/workflows/fedavg.py` -> `FedAvg(..., enable_tensor_disk_offload=True)`

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

`tensor_disk_offload_context` checks `run_manager.cell` directly when available, then falls back to `engine.get_cell()`. This applies workflow-level FOBS context updates to the active job runner cell in the normal run path.

## Runtime Behavior

In `nvflare/app_common/workflows/fedavg.py`:

- custom aggregators receive `result.params` as-is
- with `enable_tensor_disk_offload=True`, lazy refs are passed through directly
- built-in weighted aggregation materializes per tensor inside `WeightedAggregationHelper.add()`
  and relies on lazy-ref object lifetime / GC for temp-resource cleanup

The built-in weighted path remains lazy-friendly and memory-efficient.

## Custom Aggregator Contract

When a custom aggregator is used, payload params may contain lazy refs (duck-typed object with `materialize()`).

Custom aggregators are responsible for:

1. materializing refs when tensor math is required
2. releasing lazy-ref object references after use so temp resources can be reclaimed

## Temp File Lifecycle

- Disk offload writes safetensors chunks under a temp dir (`nvflare_tensors_*`).
- Temp dir selection follows Python `tempfile` behavior (`TMPDIR` / OS default, typically `/tmp`).
- In containerized deployments, `/tmp` may be tmpfs (RAM-backed); set `TMPDIR` to a disk-backed mount to realize memory offload benefits.
- `LazyTensorDict` owns a shared `_TempDirRef`; each lazy ref keeps this reference alive.
- Runtime cleanup relies on GC: when lazy refs are released, `_TempDirRef.__del__` removes the temp dir.

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
- `nvflare/recipe/fedavg.py`

## Test Coverage

- `tests/unit_test/app_common/workflow/fedavg_test.py`
- `tests/unit_test/app_opt/pt/test_lazy_tensor_dict.py`
- `tests/unit_test/app_opt/pt/test_disk_tensor_consumer.py`
- `tests/unit_test/app_common/aggregators/weighted_aggregation_helper_test.py`
- `tests/unit_test/private/fed/server/server_runner_test.py`
- `tests/stress_test/fedavg_large_model/fedavg_stress_test.py`
