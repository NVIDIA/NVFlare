# Async recipe

`AsyncRecipe` runs buffered asynchronous training using CollabAPI calls and
serialization. The default accepts actual arrivals, with one outstanding call
per physical client. NumPy and PyTorch examples use this same recipe:
[NumPy](../../examples/advanced/collab/numpy_async/README.md) and
[CIFAR-10](../../examples/advanced/collab/pt_async_cifar10/README.md).

## Use and deployment

```python
from nvflare.collab import AsyncRecipe, collab
from nvflare.collab.async_models.numpy import NumpyAsyncModelAdapter

recipe = AsyncRecipe(
    job_name="buffered_training",
    client=MyClient(), application=MyApplication(),
    model_adapter=NumpyAsyncModelAdapter(),
    num_updates=100, num_active_jobs=8, buffer_size=4, min_open_slots=1,
    weight_metric="num_steps",
)
```

Application/client classes are user supplied. Use `recipe.execute(SimEnv(...))`,
`PocEnv(...)`, or `ProdEnv(...)` from `nvflare.recipe` for the target environment.
The recipe uses the existing Collab controller/executor; it introduces no
simulator-specific training path. Configure site data, admission and call
timeouts for the deployment. Other runtime options pass to `CollabRecipe`.
Default `min_clients` is K, or the custom scheduler's `required_workers(K)`.

Components must be importable and reconstructible from constructor arguments
for export. Supporting application modules travel through existing source
packaging and `add_server_file`/`add_client_file`. For example, CIFAR explicitly
bundles its model helper. Export a job with `recipe.export(path)`.

Clients publish `train(assignment_ids, model_version, global_model,
logical_assignments)` and return `(model, metrics, metadata)`. Metadata contains
`assignment_id`, `model_version`, and `logical_name`, selected using
`collab.site_name`. Set `metadata["is_delta"] = True` for an assignment-relative
delta; otherwise return a full model. The server validates identity and version.
The default logical name is the physical site name. `AsyncResult` and assignment
objects remain server-local; the wire uses the standard Collab tuple.

## Scheduling and arithmetic

- K=`num_active_jobs`: active assignment capacity. B=`buffer_size`: accepted
  contributions per publication. O=`min_open_slots`: free slots before refill.
- Version zero is initial; `num_updates` counts global publications. Dispatch
  stops at the exact contribution budget `num_updates * B`.
- Publication precedes refill when both boundaries are reached. Refill precedes
  the publication hook so artifact work does not delay admission.
- K=B=O gives a synchronous boundary; B=O=1 gives per-arrival updates. B may
  exceed K; O must be between one and K. Default K cannot exceed physical clients.
- Full models are differenced against their exact assignment base, then the
  buffered average is applied to the current model with `server_lr`.
- Weights are equal unless `weight_metric` selects a finite, positive metric.
  CIFAR uses `num_steps`. The engine reuses `WeightedAggregationHelper` for
  running sums; it does not retain B full uploaded models.

Model adapters own snapshot/validation/difference/apply operations and leave
inputs unchanged. Supplied adapters validate keys, shapes, dtypes and finite
values, average floating entries, and preserve non-floating global buffers.
PyTorch snapshots are detached CPU tensors. Basic examples use float32.
Optional `prepare_for_aggregation(delta)` widens float16/bfloat16 before
weighting; publication restores the model dtype. There is no overflow recovery
policy. A different buffer policy belongs in a custom adapter.

Default stores use RAM, sharing one immutable base per active version. Retained
assignment/result state is bounded by K plus the current model/accumulator.
Application checkpoints/evaluation histories have their own storage costs.

## Application and update hooks

| Hook | Contract |
| --- | --- |
| `initialize()` | Required: return the initial model. |
| `on_model(model, version)` | Initial state and each published model. |
| `on_update(update)` | Inspect a validated update before accumulation. |
| `on_event(event)` | Lightweight dispatch/accept identity/version metadata. |
| `on_finish(model, version)` | Normal completion. |
| `close()` | Always called, including partial initialization/failure. |

Hooks run on the workflow thread. Borrowed states are read-only. A callable
`update_transform(AcceptedUpdate)` returns a replacement delta. Its optional
`initialize()`/`close()` manage resources. The context exposes the assignment
base, current model/version, original client delta, weight, metrics and metadata.
`staleness` is current minus base version. `returned_model` provides the full
upload; for delta uploads it is reconstructed only on demand and cached. A
delta-only consumer incurs no full-model reconstruction. An unrepresentable
full state may raise on explicit access even when a scaled delta is valid.
Copy any teacher/model state retained beyond the hook.

## Extension contracts

Core contains small policy/storage ports; research implementations live in the
[async research example](../../examples/advanced/collab/pt_async_cifar10_reproducible/README.md).
The production recipe imports no replay or RNG helper from that example.
Its research composition retains all base snapshots, results, model versions
and traces on disk, avoiding in-run artifact reclamation and reference counts.

| Option | Interface |
| --- | --- |
| `scheduler` | `AsyncScheduler` in `nvflare.collab.async_scheduling`; default `ArrivalOrderScheduler`. |
| `train_kwargs` | Callable taking `ScheduledCall` and returning extra train keyword arguments; cannot override the four standard arguments. |
| `snapshot_store` | `SnapshotStore` in `nvflare.collab.async_storage`; default `MemorySnapshotStore`. |
| `result_store` | `ResultStore` in the same module; default `MemoryResultStore`. |
| `update_transform` | Callable taking `AcceptedUpdate` and returning an adapter-compatible delta. |

A scheduler implements `required_workers(K)`, `initialize(engine, names)`,
`dispatch()`, `on_result(call)`, `next_result()`, `on_accept(assignment)`,
`finished`, and `close()`. `dispatch()` returns `ScheduledCall` objects bound to
idle physical workers. Reserve via `engine.dispatch(names)` or
`engine.reserve(logical_names, metadata=None)`; both enforce K/O and the remaining
budget. `engine.active` is keyed by assignment ID. Only the workflow accepts
results. `on_result` observes stored physical completion; `next_result` selects
a completed ID for acceptance or returns `None`. If a policy unlocks work while
returning `None`, the workflow retries dispatch before waiting. `finished` can
end a run before the publication budget. All scheduling methods run serially.

Snapshot stores implement `retain(version, model, adapter) -> handle`,
`load(handle)`, `release(handle)`, and `close()`. Each assignment retains one
reference; `assignment.base_model` borrows it until acceptance/cleanup. Handles
stay server-local. Result stores implement `put(id, result)`, `take(id)`,
`release(result)`, and `close()`. `put` takes ownership; `take` transfers it back.
The workflow releases results after consumption or failure. Store `close()`
must be idempotent and safe after partial initialization. An optional
`initialize()` runs inside the Collab context. Result `release()` may also run
on a reader thread after abort/close and must be thread-safe.

## Failures and operational limits

Transport errors/timeouts, absent or invalid results, invalid weights, and hook
exceptions stop the workflow without retry. Failure/abort discards partial
buffers. Result readers/outcome queues are bounded by physical worker count.
Outstanding waits use existing Collab timeouts; local cleanup does not guarantee
remote computation has stopped.

Checkpoints do not restore assignments, partial buffers or client optimizer
state. Existing Collab result context can supply starting weights, but starts
a fresh version-zero run. Export and simulator checks do not qualify a specific
production deployment.

Actual arrival order depends on RPC and training timing. Fixed training seeds
alone do not make it reproducible. Deterministic replay and isolated seed streams
are optional research components, documented with the research example.
Edge's selection capacity/publication buffer/refill threshold align conceptually
with K/B/O; edge retains its separate runtime and hierarchy.
