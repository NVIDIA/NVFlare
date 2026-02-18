# Disk-Streamed Tensor Aggregation

## Problem

During federated learning, the server receives model updates (PyTorch state dicts) from multiple clients via FOBS streaming. The existing path deserializes every tensor into GPU/CPU memory immediately upon receipt. For large models (multi-GB), receiving updates from N clients simultaneously causes peak memory usage of ~N× model size on the server — often exceeding available RAM.

## Design

### Core Idea

Write incoming safetensors bytes directly to disk during download instead of deserializing into memory. During FOBS recomposition, return lightweight `_LazyRef` placeholders instead of tensors. Tensors are loaded from disk one-at-a-time only when the aggregation helper needs them.

### Data Flow

```
Client sends tensors (safetensors chunks)
        │
        ▼
  DiskTensorConsumer          ← writes raw bytes to temp files (no deserialization)
        │
        ▼
  LazyTensorDict              ← dict-like: item_id → (file_path, key)
        │
        ▼
  ViaDownloaderDecomposer.recompose()
        │
        ▼
  make_lazy_ref(item_id)      ← returns _LazyRef placeholder (~100 bytes)
        │
        ▼
  consumer receives dict of _LazyRef objects
        │
        ├─ aggregation path: helper.add() → resolve() per tensor, one-at-a-time
        │
        └─ training path: trainer resolves all refs before use
        │
        ▼
  _TempDirRef.__del__()       ← cleanup when all refs are GC'd
```

### Key Components

**`DiskTensorConsumer`** (`tensor_downloader.py`) — `ItemConsumer` subclass that receives raw safetensors byte chunks from the streaming layer and writes them directly to temp files. Parses only the safetensors header (8-byte length + JSON) to extract key names — never calls `load()` on tensor data. On download failure, cleans up the temp directory.

**`LazyTensorDict`** (`lazy_tensor_dict.py`) — Dict-like object mapping tensor names to `(file_path, safetensors_key)` pairs. `__getitem__` loads tensors via `safe_open`. `make_lazy_ref(key)` creates `_LazyRef` placeholders. Shares a `_TempDirRef` with all refs it creates.

**`_LazyRef`** (`lazy_tensor_dict.py`) — Placeholder carrying `file_path`, `key`, and a shared `_TempDirRef`. `resolve()` opens mmap, copies tensor data out, closes mmap. Does NOT behave like a tensor — `isinstance(t, torch.Tensor)` returns False.

**`_TempDirRef`** (`lazy_tensor_dict.py`) — Reference-counted sentinel for a temp directory. Shared between `LazyTensorDict` and all `_LazyRef` instances. The directory is deleted only when ALL holders are garbage collected, preventing premature cleanup when the FOBS context is GC'd but lazy refs still exist.

**`cleanup_lazy_refs(data)`** (`lazy_tensor_dict.py`) — Explicit cleanup: finds any `_TempDirRef` in the dict values and deletes the temp directory.

### Integration Points

**`TensorDecomposer.download()`** (`decomposers.py`) — Checks `download_to_disk` config var. If true, calls `download_tensors_to_disk()` returning a `LazyTensorDict`.

**`ViaDownloaderDecomposer.recompose()`** (`via_downloader.py`) — When items is a `LazyTensorDict`, returns `_LazyRef` via `make_lazy_ref()`. Otherwise returns the item directly.

**`WeightedAggregationHelper.add()`** (`weighted_aggregation_helper.py`) — Resolves lazy refs via duck typing (`hasattr(v, 'resolve')`) before tensor math. Each tensor is loaded, accumulated, then freed at the end of the loop iteration.

**`DXOAggregator.accept()`** (`dxo_aggregator.py`) — Calls `data.cleanup()` in a `finally` block after `add()`. Deterministic cleanup for the DXO aggregation path.

**`FedAvg._aggregate_one_result()`** (`fedavg.py`) — Built-in InTime aggregation path calls `cleanup_lazy_refs(result.params)` after `add()`. Custom aggregator path does not force cleanup and relies on `_TempDirRef` lifetime or aggregator-managed cleanup.

**`ConfigVarName.DOWNLOAD_TO_DISK`** (`fl_constant.py`) — Config variable controlling disk download. Defaults to `False`. Enable via env var `NVFLARE_TENSOR_DOWNLOAD_TO_DISK=true`.

**`get_bool_var()`** (`app_config_utils.py`) — Utility matching existing `get_int_var` pattern.

## Client-Side Behavior

When disk streaming is enabled, clients also download tensors to disk and receive `_LazyRef` objects. Trainers must resolve all refs before use:

```python
for k, v in list(dxo.data.items()):
    if hasattr(v, "resolve"):
        dxo.data[k] = v.resolve()
```

This is a one-time bulk resolve. After this, the trainer sees normal `torch.Tensor` objects. Peak memory is 1× model size (same as without disk streaming), but the download phase benefits from not deserializing into memory.

## Memory Analysis

Measured with swarm stress test (3 clients, 1 round, CCWF/DXO path).

| Model size | Disk streaming | No disk streaming | Saved |
|------------|---------------|-------------------|-------|
| 2 GB | 6.32 GB | 10.40 GB | 4.08 GB (39%) |
| 4 GB | 10.04 GB | 17.32 GB | 7.28 GB (42%) |

Savings scale linearly with model size.

### Why it works

Without disk streaming, tensors are deserialized into memory during FOBS recomposition. The full model for each client exists as real tensors in `dxo.data`. During `add()`, the running total plus the client's full tensor dict are both in memory = 2× per client being aggregated.

With disk streaming, `dxo.data` contains `_LazyRef` placeholders (~100 bytes each). During `add()`, each tensor is loaded from disk, accumulated into the running total, and freed at the end of the loop iteration (Python refcounting). Peak = running total + one tensor ≈ 1× model size.

### safe_open and mmap

`_LazyRef.resolve()` calls `safe_open(path)` which mmaps the file. `get_tensor(key)` copies data from the mmap region into a new `torch.Tensor`. The `with` block closes the mmap. The returned tensor is an independent allocation — no persistent mmap.

## Cleanup

Temp files are cleaned up via two mechanisms:

1. **Explicit** — `cleanup_lazy_refs()` / `data.cleanup()` in built-in aggregation paths
2. **Automatic** — `_TempDirRef.__del__()` when all `_LazyRef` holders are GC'd

| API Surface | Cleanup site | Paths covered |
|-------------|-------------|---------------|
| **FLModel path (built-in InTime)** | `FedAvg._aggregate_one_result()` → `cleanup_lazy_refs(result.params)` | FedAvg, FedProx, FedOpt without custom `ModelAggregator` |
| **FLModel path (custom aggregator)** | No forced cleanup in `FedAvg` custom-aggregator branch | Custom `ModelAggregator` implementations; cleanup via `_TempDirRef` GC or aggregator logic |
| **DXO path** | `DXOAggregator.accept()` → `data.cleanup()` | ScatterAndGather, CCWF/Swarm |

## Persistors

Persistors are orthogonal to disk streaming. In the CCWF path, the aggregated result from `WeightedAggregationHelper.get_result()` contains real tensors (the running totals), not `_LazyRef` objects. `save_model()` sees fully materialized weights.

**Exception**: if a persistor receives weights that flowed through FOBS deserialization without aggregation (e.g., `shareable_to_learnable()` in some paths), it may see `_LazyRef` objects and must resolve them before saving.

## Design Principles

1. **Disk-first download (when enabled)** — Tensor downloads go to disk, saving memory during the streaming phase.

2. **Lazy resolution** — Tensors are loaded one-at-a-time during aggregation. Peak memory ≈ 1× model size.

3. **Reference-counted cleanup** — `_TempDirRef` prevents premature deletion. Temp files live as long as any `_LazyRef` references them.

4. **Duck typing** — `hasattr(v, 'resolve')` avoids coupling `app_common` to `app_opt/pt`.

5. **Opt-in via config** — `download_to_disk` config variable defaults to `false`. Set `NVFLARE_TENSOR_DOWNLOAD_TO_DISK=true` to enable disk streaming.

## Files Changed

| File | Change |
|------|--------|
| `nvflare/app_opt/pt/lazy_tensor_dict.py` | **New.** `_LazyRef`, `_TempDirRef`, `LazyTensorDict`, `cleanup_lazy_refs()` |
| `nvflare/app_opt/pt/tensor_downloader.py` | `DiskTensorConsumer`, `download_tensors_to_disk()`, `_extract_safetensors_keys()` |
| `nvflare/app_opt/pt/decomposers.py` | `TensorDecomposer.download()` dispatches to disk path |
| `nvflare/fuel/utils/fobs/decomposers/via_downloader.py` | `recompose()` returns `_LazyRef` via `make_lazy_ref()` |
| `nvflare/app_common/aggregators/weighted_aggregation_helper.py` | `add()`: resolve lazy refs per tensor |
| `nvflare/app_common/aggregators/dxo_aggregator.py` | `accept()`: cleanup after `add()` |
| `nvflare/app_common/workflows/fedavg.py` | `_aggregate_one_result()`: cleanup via `cleanup_lazy_refs()` |
| `nvflare/apis/fl_constant.py` | `DOWNLOAD_TO_DISK` config var |
| `nvflare/fuel/utils/app_config_utils.py` | `get_bool_var()` utility |

## Tests

- `tests/unit_test/app_opt/pt/test_disk_tensor_consumer.py` — safetensors header parsing, disk write, failure cleanup
- `tests/unit_test/app_opt/pt/test_lazy_tensor_dict.py` — `LazyTensorDict`, `_LazyRef`, `_TempDirRef` shared lifetime, `cleanup_lazy_refs()`, `WeightedAggregationHelper` integration
- `tests/stress_test/swarm_large_model/swarm_stress_test.py` — end-to-end swarm with large PT models, `--compare` validates identical results, `--no-disk-streaming` for baseline

## Open Questions

- Custom `ModelAggregator` implementations that defer param consumption past `accept_model()` must handle lazy ref lifetime themselves (or rely on `_TempDirRef` GC).
- Trainer transparency — trainers must explicitly resolve lazy refs when disk streaming is enabled. Auto-resolve in `shareable_to_learnable()` could cover standard paths but not custom consumers.
