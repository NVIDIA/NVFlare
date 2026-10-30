.. _async_recipe:

Async Training Recipe
=====================

``AsyncRecipe`` runs buffered, actual-arrival training over the
:ref:`Collaboration API <collab_api>`. It uses the existing controller, executor,
serialization and execution environments. The
:github_nvflare_link:`NumPy example <examples/advanced/collab/numpy_async>` and
:github_nvflare_link:`CIFAR-10 example <examples/advanced/collab/pt_async_cifar10>`
demonstrate the same workflow with example-owned model helpers.

Use and Deployment
------------------

Supply your application, client and model adapter classes:

.. code-block:: python

   from nvflare.collab import AsyncRecipe
   from nvflare.recipe import SimEnv

   recipe = AsyncRecipe(
       job_name="buffered_training",
       client=MyClient(),
       application=MyApplication(),
       model_adapter=MyModelAdapter(),
       num_updates=100,
       num_active_jobs=8,
       buffer_size=4,
       min_open_slots=1,
       weight_metric="num_steps",
   )
   run = recipe.execute(SimEnv(num_clients=8))

Use ``SimEnv``, ``PocEnv`` or ``ProdEnv`` from ``nvflare.recipe`` after preparing
the target environment and site data. Runtime options pass to ``CollabRecipe``;
``min_clients`` defaults to ``num_active_jobs``. Export with
``recipe.export(path)``. Components must be importable and reconstructible from
constructor arguments. Bundle supporting code with the existing source
packaging and ``add_server_file``/``add_client_file`` methods.

Client Contract
---------------

Clients publish ``train(assignment_ids, model_version, global_model,
logical_assignments)`` and return ``(model, metrics, metadata)``. Metadata contains
``assignment_id``, ``model_version`` and ``logical_name``, selected using
``collab.site_name``. Core logical names equal physical client names. Set
``metadata["is_delta"] = True`` for an assignment-relative delta; otherwise return
a full state. The server validates identity/version. Engine objects remain
server-local; the wire uses the normal Collab tuple.

Scheduling and Aggregation
--------------------------

* K = ``num_active_jobs``: outstanding physical client capacity.
* B = ``buffer_size``: contributions per publication.
* O = ``min_open_slots``: free slots before refill.

Version zero is initial. The workflow stops after ``num_updates`` global
publications, issuing exactly the remaining contribution budget
``num_updates * B``. It accepts results in actual arrival order, with one
outstanding call per client. Successful Collab callbacks enqueue replies directly
into one shared server queue; admission to that queue defines receipt order.
Per-call handles track completion/errors without retaining model payloads or
using result-reader threads. K must fit the available physical clients; O must
be between one and K. B may exceed K. K = B = O gives a synchronous boundary;
B = O = 1 gives per-arrival updates.

The workflow publishes before refill when both boundaries are reached, and
refills before invoking the publication hook so evaluation/artifact work does
not delay dispatch. Full returned models are differenced against their exact
assignment base. The weighted average is applied to the current global model
with ``server_lr``. Weights are equal unless ``weight_metric`` selects a finite,
positive metric. CIFAR uses ``num_steps``. ``WeightedAggregationHelper`` maintains
running sums.

Model Operations and Ownership
------------------------------

Supply a model adapter with four methods: ``snapshot``, ``validate``,
``difference`` and ``apply``. Operations leave inputs unchanged; ``apply`` must
return a new state. The concrete helpers live in the
:github_nvflare_link:`NumPy example <examples/advanced/collab/numpy_async/job.py>`
and :github_nvflare_link:`PyTorch example <examples/advanced/collab/pt_async_cifar10/model.py>`.
They validate keys/shapes/dtypes/finite values, aggregate floating entries and
preserve non-floating global buffers. PyTorch snapshots are detached CPU tensors.
The async examples use float32 and expose no precision setting. Aggregation uses
the supplied dtype through the existing weighted helper; core adds no precision
conversion or overflow-recovery policy and has no mandatory tensor-framework
dependency.

``AsyncEngine`` owns the current model. Each publication creates a new state;
active assignments reference their immutable base, shared by version. Removing
an assignment releases the engine's reference. The result queue and pending
call records are bounded by physical worker count; retained bases by K, plus the
current model/accumulator. Applications own any evaluation histories. Do not mutate
borrowed model states, including in client training; create a local model/copy
before in-place updates.

Application and Update Hooks
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Hook
     - Contract
   * - ``initialize()``
     - Required: return initial model.
   * - ``on_model(model, version)``
     - Initial and each published model.
   * - ``on_update(update)``
     - Inspect a validated update before accumulation.
   * - ``on_event(event)``
     - Dispatch/accept identity/version metadata; no models.
   * - ``on_finish(model, version)``
     - Normal completion.
   * - ``close()``
     - Always called, including partial initialization/failure.

An optional ``update_transform(AcceptedUpdate)`` returns an adapter-compatible
replacement delta. Optional ``initialize()``/``close()`` manage its resources.
The context distinguishes the assignment base, current model/version, original
client delta, weight, metrics and metadata. ``staleness`` is current minus base
version. For delta uploads, full ``returned_model`` is reconstructed only on
demand and cached; delta-only consumers incur no reconstruction. Explicit
full-state access can fail if the unscaled state is unrepresentable, even when
a scaled delta is valid. Copy any state retained beyond a hook. Hooks run
serially on the workflow thread and must leave borrowed states unchanged.

.. _async_recipe_component_scope:

Component Scope
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Component
     - Owns
     - Responsibilities supplied elsewhere
   * - :github_nvflare_link:`async_recipe.py <nvflare/collab/async_recipe.py>`:
       ``AsyncRecipe`` and ``AsyncWorkflow``
     - Configuration/export; Collab calls; actual-arrival dispatch/result
       delivery through one bounded result queue; timeout/abort cleanup;
       application hooks.
     - Numerical acceptance/publication belongs to the engine. The application
       supplies data, model operations and artifacts.
   * - :github_nvflare_link:`async_engine.py <nvflare/collab/async_engine.py>`:
       ``AsyncEngine`` and server-local records/interfaces
     - Assignment IDs, active bases and versions; K/B/O capacity and contribution
       budget; result/weight validation; exact-base deltas; buffered publication;
       update-transform context.
     - Collab RPCs, execution environments, result arrival timing,
       framework-specific arithmetic and research policies.
   * - Example model helpers
     - Snapshots, shape/dtype/finite checks, differences, update application and
       non-floating buffer rules.
     - Assignment scheduling, transport, training and experiment storage.
   * - Application and client
     - Dataset/shards, model construction, training and optimizer state;
       evaluation, metrics and artifacts through application hooks.
     - Core assignment/version accounting and buffered aggregation.
   * - :github_nvflare_link:`Research workflow
       <examples/advanced/collab/pt_async_cifar10_reproducible/workflow.py>`
     - ``ResearchRecipe``, logical reservations, predefined replay, worker reuse,
       seed streams, held results and append-only disk checkpoints.
     - Core supplies shared acceptance/aggregation math and Collab call/lifecycle
       implementation. Sampled FedRevive phases remain follow-up work.

``AsyncRecipe`` builds ``AsyncWorkflow``, which invokes ``AsyncEngine``. The
engine is independent of Collab execution and tensor frameworks. Both production
and research use the same numerical acceptance/publication path. Research may
choose when to submit or accept work and where to keep bases/results; core never
imports the examples.

The engine module defines ``Assignment``, ``AsyncResult``, ``AcceptedUpdate``,
``AsyncEvent`` and the ``AsyncModelAdapter`` interface. They describe server-local
state and hook inputs, and introduce no additional wire format. Normal consumers
import ``AsyncRecipe`` from ``nvflare.collab``; alternative orchestration can
import ``AsyncEngine`` from ``nvflare.collab.async_engine``. Protected workflow
methods are implementation reuse points rather than a public scheduler/storage
plugin API.

Reproducible Research
---------------------

Actual arrivals depend on training/RPC timing; fixed seeds alone do not make them
reproducible. The
:github_nvflare_link:`reproducible example <examples/advanced/collab/pt_async_cifar10_reproducible>`
explains fixed replay, seed helpers and retained experiment files. Its
``ResearchRecipe`` and workflow reuse core numerical acceptance and the protected
``_engine_class``/``_run`` implementation points while sharing engine construction,
Collab calls and lifecycle. The research API and concrete model helpers remain
examples, outside the installed SDK.

Failure and Recovery Limits
---------------------------

Errors/timeouts, missing or invalid results, invalid weights and hook exceptions
stop the workflow without retry. Abort/failure discards partial buffers and
stops queuing late replies. Submitted Collab calls finish or reach their configured
timeout; cleanup does not cancel remote computation. Checkpoints do
not restore assignments, buffers or client optimizer state. Existing Collab
result context can supply starting weights but starts a fresh version-zero run.

Export/simulator checks do not qualify a particular production deployment.
Edge's selection capacity/publication buffer/refill threshold align conceptually
with K/B/O; edge retains its separate runtime and hierarchy.
