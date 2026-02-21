# Disk-Streamed Tensor Aggregation

## Objective

Reduce server peak memory for large PyTorch model updates by streaming tensor payloads to disk and resolving tensors lazily end-to-end.

## Problem

Without disk streaming, each incoming client model is deserialized directly into memory during FOBS recompose. For large models and multiple concurrent submissions, peak server RSS grows with the number of in-flight updates.


## Data Flow

```
TensorDownloadable chunks
        |
        v
TensorDecomposer.download()
  - stream_to_disk=False -> deserialize in memory
  - stream_to_disk=True  -> write safetensors temp files
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

## Runtime Behavior

### FedAvg

In `nvflare/app_common/workflows/fedavg.py`:

- custom aggregators receive `result.params` as-is
- with `stream_to_disk=True`, this means lazy refs are passed through directly
- built-in weighted aggregation materializes per tensor inside `WeightedAggregationHelper.add()`
  and runs `cleanup_inplace(result.params)` in a `finally` block

The built-in weighted path remains lazy-friendly and memory-efficient.

### Swarm/CCWF

In `nvflare/app_common/ccwf/swarm_client_ctl.py` gather path:

- shareables are passed to the aggregator as-is
- with `stream_to_disk=True`, aggregator inputs include lazy refs

## Lazy Payload Utilities

`nvflare/app_common/utils/lazy_payload.py` provides duck-typed helpers so `app_common` does not depend on PyTorch lazy internals:

- `cleanup_inplace(data)`

Lazy resolution itself happens where tensor math is performed (for example in weighted aggregation).

## FOBS Context Scope

`Cell.get_fobs_context()` is process-local runtime state (`CoreCell.fobs_ctx`) used by FOBS encode/decode behavior.

Implications:

- scope is per-process, per-cell runtime
- affects subsequent FOBS operations in that process
- fits run-scoped policy

## Cleanup Semantics

Cleanup is done through:

1. explicit cleanup hooks (`cleanup_inplace`, DXO cleanup)
2. `_TempDirRef` lifetime fallback on GC


## Design-Relevant Files

- `nvflare/app_opt/pt/decomposers.py`
- `nvflare/app_opt/pt/lazy_tensor_dict.py`
- `nvflare/app_opt/pt/tensor_downloader.py`
- `nvflare/fuel/utils/fobs/decomposers/via_downloader.py`
- `nvflare/app_common/utils/lazy_payload.py`
- `nvflare/app_common/workflows/fedavg.py`
- `nvflare/app_common/ccwf/swarm_client_ctl.py`
- `nvflare/private/fed/server/server_engine.py`
- `nvflare/recipe/fedavg.py`

## Test Coverage

- `tests/unit_test/app_common/workflow/fedavg_test.py`
- `tests/unit_test/app_common/ccwf/test_swarm_lazy_payload.py`
- `tests/unit_test/app_common/utils/lazy_payload_test.py`
- `tests/unit_test/private/fed/server/server_runner_test.py`
- `tests/stress_test/fedavg_large_model/fedavg_stress_test.py`
