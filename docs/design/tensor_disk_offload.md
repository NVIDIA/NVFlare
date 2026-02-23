# Tensor Disk Offload

## Objective

Reduce server peak memory for large PyTorch model updates by streaming tensor payloads to disk and resolving tensors lazily end-to-end.

## Problem

Without disk streaming, each incoming client model is deserialized directly into memory during FOBS recompose. For large models and multiple concurrent submissions, peak server RSS grows with the number of in-flight updates.

## Scope

- Applies to streamed **PyTorch tensor** payloads handled by `TensorDecomposer`.
- Controlled by `enable_tensor_disk_offload` in server-side workflow/controller config.
- Default is `False` (legacy in-memory behavior).
- If model updates are converted to NumPy before transport, tensor disk offload is not engaged.

## How To Enable

FedAvg:

- `nvflare/recipe/fedavg.py` -> `FedAvgRecipe(..., enable_tensor_disk_offload=True)`
- `nvflare/app_opt/pt/recipes/fedavg.py` -> PT recipe forwards the same flag
- `nvflare/app_common/workflows/fedavg.py` -> `FedAvg(..., enable_tensor_disk_offload=True)`

Swarm/CCWF:

- `nvflare/app_common/ccwf/ccwf_job.py` -> `SwarmClientConfig(..., enable_tensor_disk_offload=True)`
- `nvflare/app_common/ccwf/swarm_client_ctl.py` applies the flag to the active Cell FOBS context at run start

If no active Cell is available, FedAvg logs a warning and falls back to in-memory download.

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

`ServerEngine.get_cell()` now prefers the run manager cell when available, so workflow-level FOBS context updates are applied to the active job runner cell.

## Runtime Behavior

### FedAvg

In `nvflare/app_common/workflows/fedavg.py`:

- custom aggregators receive `result.params` as-is
- with `enable_tensor_disk_offload=True`, lazy refs are passed through directly
- built-in weighted aggregation materializes per tensor inside `WeightedAggregationHelper.add()`
  and relies on lazy-ref object lifetime / GC for temp-resource cleanup

The built-in weighted path remains lazy-friendly and memory-efficient.

### Custom Aggregator Contract (Important)

When a custom aggregator is used, payload params may contain lazy refs (duck-typed object with `materialize()`).

Custom aggregators are responsible for:

1. materializing refs when tensor math is required
2. cleaning temporary resources after use (for example by calling `cleanup()` on lazy refs)

### Swarm/CCWF

In `nvflare/app_common/ccwf/swarm_client_ctl.py` gather path:

- shareables are passed to the aggregator as-is
- with `enable_tensor_disk_offload=True`, aggregator inputs include lazy refs

## Temp File Lifecycle

- Disk offload writes safetensors chunks under a temp dir (`nvflare_tensors_*`).
- `LazyTensorDict` owns a shared `_TempDirRef`; each lazy ref keeps this reference alive.
- Cleanup paths:
  1. explicit cleanup (`_LazyRef.cleanup()`/`LazyTensorDict.cleanup()`, or equivalent app-level traversal that calls `cleanup()`)
  2. fallback GC cleanup via `_TempDirRef.__del__`

## Cleanup Semantics

Cleanup is done through:

1. explicit cleanup hooks (`cleanup()`)
2. `_TempDirRef` lifetime fallback on GC

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
- `nvflare/private/fed/server/server_engine.py`
- `nvflare/recipe/fedavg.py`

## Test Coverage

- `tests/unit_test/app_common/workflow/fedavg_test.py`
- `tests/unit_test/app_common/ccwf/test_swarm_lazy_payload.py`
- `tests/unit_test/app_opt/pt/test_lazy_tensor_dict.py`
- `tests/unit_test/app_opt/pt/test_disk_tensor_consumer.py`
- `tests/unit_test/app_common/aggregators/weighted_aggregation_helper_test.py`
- `tests/unit_test/private/fed/server/server_runner_test.py`
- `tests/stress_test/fedavg_large_model/fedavg_stress_test.py`
