# Async Recipe Contracts

`AsyncRecipe` provides actual-arrival buffered training over Collab. Its consumers are `examples/advanced/collab/numpy_async`
(NumPy) and `examples/advanced/collab/pt_async_cifar10` (PyTorch).

## Public imports

```python
from nvflare.collab import AsyncRecipe, collab
from nvflare.collab.async_workflow import (
    AcceptedUpdate, Assignment, AsyncEngine, AsyncModelAdapter, AsyncResult,
)
from nvflare.collab.async_models.numpy import NumpyAsyncModelAdapter

# Optional, for PyTorch applications:
from nvflare.collab.async_models.pytorch import PTAsyncModelAdapter
```

The engine/result types work independently of a recipe. The recipe and NumPy
adapter do not import PyTorch, Kornia, or KD. Imports allocate no models.

## Composition and client protocol

```python
recipe = AsyncRecipe(
    job_name="buffered_training",
    client=MyClient(),
    application=MyApplication(),
    model_adapter=NumpyAsyncModelAdapter(),
    num_updates=100, num_active_jobs=8, buffer_size=4, min_open_slots=1,
    weight_metric="num_steps",
)
```

The application/client are user-defined importable classes. Job export requires
components reconstructible from their constructor arguments. Use importable
callable classes for transformations; direct engine use can also use functions.
Existing Collab source packaging and explicit `add_server_file`/`add_client_file`
calls supply supporting modules. Other runtime options pass to `CollabRecipe`.
Default `min_clients` is K; an override must still leave K clients available when
the workflow starts.

Clients publish `train(assignment_ids, model_version, global_model,
logical_assignments)`. Return a tuple `(model, metrics, metadata)` where
metadata contains `assignment_id`, `model_version`, and `logical_name`.
Select the ID using `assignment_ids[collab.site_name]`. CollabAPI performs normal
serialization; no additional wire type, serializer, or registration is introduced.
`AsyncResult` is used only by the local engine interface.
Set `metadata["is_delta"] = True` for a delta relative to the assignment base;
otherwise return a full state. Server-owned identities and versions are validated.
Default logical identity equals physical site identity, matching the
CIFAR example. General logical-population virtualization is not implemented yet.

## Scheduling and arithmetic

- K=`num_active_jobs`: active assignment capacity. B=`buffer_size`: contributions
  per publication. O=`min_open_slots`: free slots required before refill.
- Version zero is initial. Stop after `num_updates` global publications. Dispatch
  is capped at the exact remaining contribution budget, avoiding speculative work.
- Publication precedes refill when both boundaries are reached. Refill precedes
  the publication hook, so artifact work does not delay dispatch.
- K=B=O gives a synchronous boundary; B=O=1 gives per-arrival updates. B may exceed
  K; O must be between one and K. Idle clients are selected in queue order.
- Full returned models are differenced against their exact assignment base.
  Apply the buffered average to the current global model with `server_lr`.
- Default weights are equal. A configured `weight_metric` requires finite, strictly
  positive metric values. CIFAR uses `num_steps`, independently of sample counts.

The engine reuses `WeightedAggregationHelper` for running weighted sums, retaining
one accumulator rather than B full models. Floating-point summation follows
acceptance order; another arrival order need not be bitwise identical.

Adapters provide snapshot, validation, difference, and apply without mutating
inputs. Supplied adapters require identical full-state keys/shapes/dtypes and
finite values. They average floating entries and preserve non-floating buffers
from the current model. PyTorch snapshots are detached CPU tensors. Supply
another adapter for different buffer rules.

Snapshots are shared within dispatch batches and released when assignments no
longer reference them. Model storage is bounded by K active assignments plus
the current model/accumulator, not run length. This implementation uses RAM;
there is no disk snapshot store.

## Optional update transformation

A callable receives `AcceptedUpdate`: assignment, current model/version, delta,
weight, and metrics. Return a replacement delta with matching keys/shapes/dtypes.
Borrowed model states are read-only. Transformation, acceptance, and publication
run serially. Staleness is measured at acceptance as current minus base version.
Transform rejection and weight changes are not part of this initial contract.

A transformation may combine deltas with different reference versions, such as
a client delta relative to its assignment base and a distillation delta relative
to the current model. It must preserve this distinction explicitly.

## Application hooks

| Hook | Contract |
| --- | --- |
| `initialize()` | Required: construct resources and return initial model. |
| `on_model(model, version)` | Optional: initial and each published model; borrowed states must not be mutated. |
| `on_finish(model, version)` | Optional: successful completion only, after outstanding work settles at the result-protocol level. |
| `close()` | Optional: always called, including partial initialization/failure/abort. |

CIFAR retains its ModerateCNN, augmentation, SGD, persistent client optimizer and
cosine scheduler. It writes version snapshots during training, then evaluates
every version on the same centralized test set after training. Round checkpoints,
final weights, accuracy history, and TensorBoard remain application concerns.

## Failures and limits

Transport errors/timeouts, absent results, malformed metadata/model values,
invalid weights, and transform exceptions stop the workflow. There are no
automatic retries: timeout does not prove remote training has stopped. Obsolete
terminal events cannot retire newer assignments. Failure/abort discards partial
buffers rather than publishing them.

The adapter has a K-sized outcome queue and at most K result readers. Abort is
checked while waiting; cleanup closes admission and stops readers from publishing
further outcomes. Outstanding result waits use existing Collab timeouts. The
public group API offers no remote training cancellation, so local cleanup does
not prove remote compute termination.

Checkpoints do not implement automatic distributed workflow recovery. Assignment
state, partial buffers, client optimizer state, and version continuity are not
restored. Existing Collab result context may supply initial weights but starts
a fresh version-zero run.

## Edge alignment

K/B/O correspond conceptually to edge's `device_selection_size`,
`num_updates_for_model`, and `min_hole_to_fill`. Hierarchical device reports,
contribution counts, availability, and stale-version policies remain edge concerns.
There is no edge migration, adapter, or configuration interchange in this work.
