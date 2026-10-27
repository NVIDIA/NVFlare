# Async Recipe Contracts

`AsyncRecipe` provides buffered asynchronous training over Collab, using actual
arrival order by default and an optional scheduler for controlled experiments. Its consumers are `examples/advanced/collab/numpy_async`
(NumPy) and `examples/advanced/collab/pt_async_cifar10` (PyTorch).

## Public imports

```python
from nvflare.collab import AsyncRecipe, collab
from nvflare.collab.async_workflow import (
    AcceptedUpdate, Assignment, AsyncEngine, AsyncModelAdapter, AsyncResult,
)
from nvflare.collab.async_scheduling import (
    AsyncEvent, AsyncScheduler, ArrivalOrderScheduler, ReplayScheduler, ScheduledCall,
)
from nvflare.collab.async_storage import (
    SnapshotStore, ResultStore, MemorySnapshotStore, MemoryResultStore,
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
Default `min_clients` is `scheduler.required_workers(K)`: K for actual-arrival
and physical-site replay, one for logical replay. An explicit override is passed
to CollabRecipe; the scheduler validates its worker requirements at startup.

Clients publish `train(assignment_ids, model_version, global_model,
logical_assignments)`. Return a tuple `(model, metrics, metadata)` where
metadata contains `assignment_id`, `model_version`, and `logical_name`.
Select the ID using `assignment_ids[collab.site_name]`. CollabAPI performs normal
serialization; no additional wire type, serializer, or registration is introduced.
`AsyncResult` is used only by the local engine interface.
Set `metadata["is_delta"] = True` for a delta relative to the assignment base;
otherwise return a full state. Server-owned identities and versions are validated.
Default logical identity equals physical site identity, matching the CIFAR
example. A custom scheduler or logical replay can reserve logical assignments
before binding them to physical sites. The four standard arguments are always
sent; `train_kwargs(call)` supplies additional keyword arguments when needed.

## Scheduling and arithmetic

- K=`num_active_jobs`: active assignment capacity. B=`buffer_size`: contributions
  per publication. O=`min_open_slots`: free slots required before refill.
- Version zero is initial. Stop after `num_updates` global publications. Dispatch
  is capped at the exact remaining contribution budget, avoiding speculative work.
- Publication precedes refill when both boundaries are reached. Refill precedes
  the publication hook, so artifact work does not delay dispatch.
- K=B=O gives a synchronous boundary; B=O=1 gives per-arrival updates. B may exceed
  K; O must be between one and K. In actual-arrival mode, workers start in sorted
  site-name order and are refilled in acceptance order.
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

Adapters may optionally implement `prepare_for_aggregation(delta)` to select
an accumulator dtype after input/transform validation. The supplied adapters
widen float16 (and PyTorch bfloat16) deltas to float32 before weighting and
summation, then restore the original model dtype when publishing. Custom
adapters without this method retain their own delta dtype.

The default snapshot store shares one immutable state per active base version,
including across dispatch batches. Assignment references end at acceptance or
workflow cleanup. The default model/result storage is bounded by K assignments
plus the current model/accumulator, not run length. Defaults use RAM; custom
snapshot/result stores can provide disk offload without replacing the workflow.

## Optional update transformation

A callable receives `AcceptedUpdate`: assignment, current model/version, client
delta, weight, metrics, full `returned_model`, and raw result `metadata`. For a
delta payload, `returned_model` is reconstructed against the exact assignment
base using the adapter's buffer rules. For full uploads it is the original model.
Return a replacement delta with matching keys/shapes/dtypes.
Borrowed model states are read-only. Transformation, acceptance, and publication
run serially. Staleness is measured at acceptance as current minus base version.
An optional `initialize()` runs after application initialization; `close()` always
runs, including initialization failure. These hooks let a transform own teacher,
proxy, generator and KD resources. Transform rejection and weight changes are
not part of the contract.

A transformation may combine deltas with different reference versions, such as
a client delta relative to its assignment base and a distillation delta relative
to the current model. It must preserve this distinction explicitly.

## Application hooks

| Hook | Contract |
| --- | --- |
| `initialize()` | Required: construct resources and return initial model. |
| `on_model(model, version)` | Optional: initial and each published model; borrowed states must not be mutated. |
| `on_update(update)` | Optional: inspect the validated client update after the transform returns, before accumulation. `update.delta` is the original client delta; `returned_model` and metadata support teacher/proxy diagnostics. |
| `on_event(event)` | Optional: observe a lightweight dispatch/accept event, including identity, base/current versions, assignment metadata and optional scheduler time. |
| `on_finish(model, version)` | Optional: normal completion at the version budget or scheduler horizon. Outstanding jobs at a horizon are discarded locally. |
| `close()` | Optional: always called, including partial initialization/failure/abort. |

CIFAR retains its ModerateCNN, augmentation, SGD, persistent client optimizer and
cosine scheduler. It writes version snapshots during training, then evaluates
every version on the same centralized test set after training. Round checkpoints,
final weights, accuracy history, and TensorBoard remain application concerns.

## Explicit extension points on AsyncRecipe

Pass components through `scheduler`, `train_kwargs`, `snapshot_store`,
`result_store`, and `update_transform`. They are constructor arguments on both
`AsyncRecipe` and its exported `AsyncWorkflow`, rather than monkey patches or
application-owned scheduling loops. Omitting a component selects default
behavior; none of these imports requires a tensor framework or KD package.

### Predefined arrival sequences and replay

An arrival sequence contains **assignment IDs**, not only client names. IDs are
unique across repeated participation by one logical client. The sequence must
permute all `num_updates * buffer_size` contributions. At each acceptance, replay
waits for the specified result while retaining early completions. Failures and
transport timeouts are reported immediately, even for an out-of-order result.
An order requiring an assignment that K/O has not yet allowed to be reserved
raises an error instead of hanging.

```python
recipe = AsyncRecipe(
    job_name="replay", client=MyClient(), application=MyApplication(),
    model_adapter=NumpyAsyncModelAdapter(),
    num_updates=4, num_active_jobs=2, buffer_size=1,
    scheduler=ReplayScheduler(arrival_sequence=[1, 0, 3, 2]),
)
```

This fixes acceptance order **and subsequent dispatch/base-version decisions**.
It preserves the ordinary physical-site assignment policy, including client-local
state. K must fit the physical worker count in this mode.

For experiments with more logical jobs than physical workers:

```python
class TrainingSeeds:
    def __call__(self, call):
        # These seeds belong to logical assignments, independent of host timing.
        return {"train_seeds": {
            call.client_name: call.assignment.metadata["train_seed"],
        }}

recipe = AsyncRecipe(
    job_name="logical_replay", client=LogicalClient(), application=MyApplication(),
    model_adapter=NumpyAsyncModelAdapter(),
    num_updates=4, num_active_jobs=3, buffer_size=1,
    scheduler=ReplayScheduler(
        arrival_sequence=[2, 0, 1, 3],
        logical_sequence=["alice", "bob", "alice", "carol"],
        assignment_metadata=[{"train_seed": 100 + i} for i in range(4)],
    ),
    train_kwargs=TrainingSeeds(),
    min_clients=1,
)
```

`logical_sequence` names assignments in reservation/ID order. It may contain
repeated clients, including overlapping uploads. Jobs receive an immutable base
when reserved, **before** any physical worker becomes available. Workers are
reusable immediately after their RPC completes; a result can remain held for
later acceptance. Logical K/O/B decisions do not depend on this reuse. Clients
must manage any persistent training state by logical identity; FedRevive's fresh
optimizer per assignment also works with this model.

`LogicalClient.train` must declare the extra `train_seeds` argument. The ordinary
four-argument client needs no change when `train_kwargs` is omitted. Extra options
must not override the four standard arguments. Training seeds, data splits,
augmentation, and deterministic model operations remain application concerns;
replay controls scheduling, not all numerical nondeterminism.

Record a run through `application.on_event(event)`, for example by writing
`dataclasses.asdict(event)` to JSONL. Events contain no model tensors. For replay,
collect accepted assignment IDs in trace order; collect logical names and
assignment metadata in ID order. Save the K/B/O configuration and seeds with the
trace. `current_version` on an acceptance event is the version **before** that
update, so staleness is `current_version - base_version`. A physical dispatch
event can occur after logical reservation and reports both versions explicitly.

### Custom scheduling and simulated time

Implement `AsyncScheduler` for sampled delay profiles, phase events, or another
selection policy. It has these contracts, all on the workflow thread:

| Method/property | Responsibility |
| --- | --- |
| `required_workers(K)` | Minimum physical workers used as the default Collab admission count. |
| `initialize(engine, client_names)` | Start a fresh run; physical names arrive in sorted order. |
| `dispatch()` | Reserve jobs, route pending jobs, and return `ScheduledCall(assignment, physical_name)` objects for idle workers. |
| `on_result(call)` | Observe physical completion; a result is already in ResultStore. A virtual scheduler can release the worker here. |
| `next_result()` | Return a completed assignment ID to accept, or `None` to wait. May advance logical phase events and a simulated clock. |
| `on_accept(assignment)` | Observe logical retirement after the engine consumes a contribution. |
| `finished` | Stop at a policy horizon independently of the engine's version ceiling. |
| `close()` | Clear runtime state, including after partial initialization. |

`engine.dispatch(physical_names)` reserves one outstanding job per supplied
site. `engine.reserve(logical_names, metadata=None)` reserves unbound logical
jobs, allowing repeated logical names. Both enforce K, O and the remaining
contribution budget. `engine.active` is keyed by assignment ID. An unbound
assignment has `client_name=None`; its `ScheduledCall.client_name` supplies the
physical destination. Only the workflow calls `engine.accept`.

A phase-based scheduler can release a logical client at its simulated training
completion, sample an upload delay, and later accept that assignment at its
upload event. This permits the client's next job to overlap the prior upload.
Expose an optional numeric `time` attribute to include simulated time in traces;
set `finished` at the time horizon. K counts all reserved/unaccepted jobs,
including pending execution and completed results. Physical completions must
not advance the simulated clock or change reservation/eligibility decisions.
Order equal-time events with an explicit deterministic tie breaker. After a
`next_result()` phase advance returns `None`, the workflow retries `dispatch()`
before waiting. If this produces no physical work and no call is outstanding,
the scheduler cannot make progress and the run fails explicitly. `next_result`
should process internal phase events until it can yield a result, unlock
physical work, reach a horizon, or needs an outstanding physical result.

ReplayScheduler supplies fixed order replay, **not** FedRevive's sampled
multi-phase delay simulator. That policy can be added through this interface
without replacing Collab transport, engine aggregation, or the simulator.

### Snapshot and result storage

`SnapshotStore` and `ResultStore` are independently importable protocols. Their
memory implementations are defaults; offload implementations remain optional.
Both own one run and must support idempotent cleanup after partial initialization.
An optional `initialize()` runs inside the Collab context before scheduling.

SnapshotStore implements `retain(version, model, adapter) -> handle`, `load(handle)
-> model`, `release(handle)`, and `close()`. Each retained assignment acquires one
reference. A store may share a version file; the last release reclaims it.
`assignment.base_model` loads the exact base on demand and is valid only while
that assignment is active. A transform must copy any teacher/model state it
retains beyond its call. Store handles remain server-local.

ResultStore implements `put(assignment_id, result)`, `take(assignment_id)`,
`release(result)`, and `close()`. `put` takes ownership of the raw Collab tuple;
`take` removes it and may materialize it. The workflow releases a taken result
whether normalization/transformation succeeds or fails. `close` releases stored
results and any run-owned offload context. After abort, a result reader can call
`release` for a late result, including after `close`; this method must be
thread-safe. Storing an early result must not require accepting its update.

These ports allow future tensor disk offload and reference-counted mmap bases
while retaining the standard Collab tuple/serialization protocol. The NumPy
extension tests exercise separate file-backed stores through these contracts;
they do not introduce another transport or simulator.

## FedRevive extension coverage

The source workflow needs the following independently switchable pieces:

| FedRevive requirement | AsyncRecipe extension |
| --- | --- |
| K/B/O, exact stale base, running aggregation | Shared AsyncEngine and model adapter. |
| Fixed participation and arrival order | ReplayScheduler, logical_sequence and assignment metadata. |
| K logical jobs over a small physical pool | engine.reserve plus ScheduledCall worker binding. |
| Seeded download/train/upload delays, logical release before upload | Custom AsyncScheduler phase events and optional `time`. |
| Simulated-time budget with a version safety limit | Scheduler `finished` plus recipe `num_updates`; partial buffers are discarded. |
| Per-assignment training seeds | Assignment metadata plus train_kwargs. |
| Exact bases and early results offloaded to disk | SnapshotStore and ResultStore. |
| Full uploaded teachers, two-upload class proxies | AcceptedUpdate.returned_model, metadata and assignment.logical_name. |
| Teacher buffer, generator cadence, KD warmup and cosine stale mixing | Stateful update_transform using current_version/staleness; optional initialize/close. |
| Distilled delta relative to current global state | Transform sees current_model separately from assignment.base_model/client delta. |
| Algorithm-specific buffer/reset rules | Custom model adapter. |
| Schedule traces, staleness metrics, accuracy/artifacts | Application on_event, on_update, on_model and on_finish. |

The async layer implements the scheduling/storage **interfaces** and fixed
replay. The sampled FedRevive scheduler, production tensor offload stores,
teacher/proxy/KD components, and migration of the research experiment remain
later implementation stages. They should compose these ports rather than add
FedRevive imports or flags to the generic recipe.

## Failures and limits

Transport errors/timeouts, absent results, malformed metadata/model values,
invalid weights, and transform exceptions stop the workflow. There are no
automatic retries: timeout does not prove remote training has stopped. Obsolete
terminal events cannot retire newer assignments. Failure/abort discards partial
buffers rather than publishing them.

The workflow bounds the outcome queue and result readers by physical worker
count; submitted, pending and completed logical work is bounded by K. Abort is
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
