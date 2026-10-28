# Async recipe

`AsyncRecipe` runs buffered, actual-arrival training over CollabAPI. It uses the
existing controller, executor, serialization and execution environments.
[NumPy](../../examples/advanced/collab/numpy_async/README.md) and
[CIFAR-10](../../examples/advanced/collab/pt_async_cifar10/README.md) demonstrate
the same core workflow with example-owned model helpers.

## Component scope

| Component | Owns | Responsibilities supplied elsewhere |
| --- | --- | --- |
| [async_recipe.py](../../nvflare/collab/async_recipe.py): `AsyncRecipe` and `AsyncWorkflow` | Recipe configuration/export; Collab client calls; actual-arrival dispatch/result delivery; bounded readers/queues; timeout/abort cleanup; invoking application hooks. | Numerical acceptance/publication belongs to the engine. Data, model operations and artifacts are supplied by the application. |
| [async_engine.py](../../nvflare/collab/async_engine.py): `AsyncEngine` and server-local records/interfaces | Assignment IDs, active bases and versions; K/B/O capacity and contribution budget; result/weight validation; exact-base deltas; buffered publication; update-transform context. | Collab RPCs, execution environments, result arrival timing, framework-specific arithmetic and research policies. |
| Example model helpers: [NumPy](../../examples/advanced/collab/numpy_async/job.py) and [PyTorch](../../examples/advanced/collab/pt_async_cifar10/model.py) | Model snapshots, shape/dtype/finite checks, differences, update application, and non-floating buffer rules. | Assignment scheduling, transport, training and experiment storage. |
| Application and client | Dataset/shards, model construction, local training and optimizer state; evaluation, metrics and artifacts through application hooks. | Core assignment/version accounting and buffered aggregation. |
| [Research workflow](../../examples/advanced/collab/pt_async_cifar10_reproducible/workflow.py) and its example helpers | `ResearchRecipe`, logical reservations, predefined replay, worker reuse, seed streams, held results and append-only disk checkpoints. | Shared acceptance/aggregation math and Collab call/lifecycle implementation come from core. Sampled FedRevive phase scheduling remains follow-up work. |

`AsyncRecipe` builds `AsyncWorkflow`, which invokes `AsyncEngine`. The actual
Collab workflow lives with the recipe; the engine is independent of Collab
execution and tensor frameworks. Both production and research use the same
numerical acceptance/publication path. Research may choose when to submit or
accept work and where to keep bases/results; core never imports the examples.

The engine module also defines `Assignment`, `AsyncResult`, `AcceptedUpdate`,
`AsyncEvent`, and the `AsyncModelAdapter` interface. They describe server-local
state and hook inputs, and introduce no additional wire format. Normal consumers
import `AsyncRecipe` from `nvflare.collab`; alternative orchestration can import
`AsyncEngine` from `nvflare.collab.async_engine`. Protected workflow methods are
implementation reuse points rather than a public scheduler/storage plugin API.

## Use and deployment

```python
from nvflare.collab import AsyncRecipe

recipe = AsyncRecipe(
    job_name="buffered_training", client=MyClient(), application=MyApplication(),
    model_adapter=MyModelAdapter(), num_updates=100, num_active_jobs=8,
    buffer_size=4, min_open_slots=1, weight_metric="num_steps",
)
```

Application/client classes and model arithmetic are supplied by the caller.
Use `recipe.execute(SimEnv(...))`, `PocEnv(...)`, or `ProdEnv(...)` from
`nvflare.recipe` after preparing the target environment and site data. Runtime
options pass to `CollabRecipe`; `min_clients` defaults to K. Export with
`recipe.export(path)`. Components must be importable and reconstructible from
constructor arguments. Bundle supporting code with the existing source
packaging and `add_server_file`/`add_client_file` methods.

Clients publish `train(assignment_ids, model_version, global_model,
logical_assignments)` and return `(model, metrics, metadata)`. Metadata contains
`assignment_id`, `model_version`, and `logical_name`, selected using
`collab.site_name`. Core logical names equal physical client names. Set
`metadata["is_delta"] = True` for an assignment-relative delta; otherwise return
a full state. The server validates identity/version. Engine objects remain
server-local; the wire uses the normal Collab tuple.

## Scheduling, arithmetic and ownership

- K=`num_active_jobs`: outstanding physical client capacity. B=`buffer_size`:
  contributions per publication. O=`min_open_slots`: free slots before refill.
- Version zero is initial. Stop after `num_updates` global publications, issuing
  exactly the remaining contribution budget `num_updates * B`.
- Accept in actual arrival order, with one outstanding call per client. K must
  fit the available physical clients; O must be between one and K. B may exceed K.
- K=B=O gives a synchronous boundary; B=O=1 gives per-arrival updates.
- Publish before refill when both boundaries are reached; refill before invoking
  the publication hook so evaluation/artifact work does not delay dispatch.
- Difference full returned models against their exact assignment base. Apply
  the weighted average to the current global model with `server_lr`.
- Weights are equal unless `weight_metric` selects a finite, positive metric.
  CIFAR uses `num_steps`. `WeightedAggregationHelper` maintains running sums.

`AsyncEngine` owns the current model. Each publication creates a new state;
active assignments directly reference their immutable base, shared by version.
Removing an assignment releases the engine's reference. There is no snapshot
store, reference counter, result store or selectable scheduler in core. Queued
results and readers are bounded by physical worker count; retained bases by K,
plus the current model/accumulator. Applications own any evaluation histories.
Do not mutate borrowed model states, including in client training; create a local
model/copy before in-place updates.

`AsyncModelAdapter` is a small structural interface in `async_engine.py`:
`snapshot`, `validate`, `difference`, and `apply`. Operations leave inputs
unchanged; `apply` must return a new state. NumPy's concrete helper lives in
`numpy_async/job.py`; PyTorch's lives in `pt_async_cifar10/model.py`. They validate
keys/shapes/dtypes/finite values, aggregate floating entries, and preserve
non-floating global buffers. PT snapshots are detached CPU tensors. The async
examples use float32 and expose no precision setting. Aggregation uses the supplied
dtype through the existing weighted helper; core adds no precision conversion or
overflow-recovery policy and has no mandatory tensor-framework dependency.

## Application and update hooks

| Hook | Contract |
| --- | --- |
| `initialize()` | Required: return initial model. |
| `on_model(model, version)` | Initial and each published model. |
| `on_update(update)` | Inspect a validated update before accumulation. |
| `on_event(event)` | Dispatch/accept identity/version metadata; no models. |
| `on_finish(model, version)` | Normal completion. |
| `close()` | Always called, including partial initialization/failure. |

An optional `update_transform(AcceptedUpdate)` returns an adapter-compatible
replacement delta. Optional `initialize()`/`close()` manage its resources. The
context distinguishes the assignment base, current model/version, original
client delta, weight, metrics and metadata. `staleness` is current minus base
version. For delta uploads, full `returned_model` is reconstructed only on demand
and cached; delta-only consumers incur no reconstruction. Explicit full-state
access can fail if the unscaled state is unrepresentable, even when a scaled
delta is valid. Copy any state retained beyond a hook. Hooks run serially on the
workflow thread and must leave borrowed states unchanged.

## Research boundary and limits

The [reproducible example](../../examples/advanced/collab/pt_async_cifar10_reproducible/README.md)
owns `ResearchRecipe`, its workflow, replay, logical reservations, seed helpers,
and append-only disk checkpoints. It reuses core numerical acceptance and
protected engine/call/run implementation methods; these are not a general
scheduler/storage plugin API. Core imports no research module. Model-specific
helpers and the research API remain examples, outside the installed SDK.

Errors/timeouts, missing or invalid results, invalid weights and hook exceptions
stop the workflow without retry. Abort/failure discards partial buffers. Reader
cleanup drains outstanding Collab calls to their configured timeout and discards
late replies; it cannot guarantee remote computation has stopped. Checkpoints do
not restore assignments, buffers or client optimizer state. Existing Collab result
context can supply starting weights but starts a fresh version-zero run.

Actual arrivals depend on training/RPC timing; fixed seeds alone do not make them
reproducible. Export/simulator checks do not qualify a particular production
deployment. Edge's selection capacity/publication buffer/refill threshold align
conceptually with K/B/O; edge retains its separate runtime and hierarchy.
