.. _reproducible_simulation:

Reproducible Simulation Plans
=============================

``SimEnv(reprod=True, plan=...)`` controls logical asynchronous execution for
cooperating Collab workflows without changing a recipe or its exported job.
The same job can run in ``ProdEnv`` with live scheduling. A plan is a JSON
dictionary or a path to a JSON file; it is simulator input and is never
packaged in the job.

Buffered Asynchronous Execution
-------------------------------

The :github_nvflare_link:`runnable NumPy example
<examples/advanced/collab/numpy_async_reproducible>` constructs a
``FedBuffRecipe`` and runs it from ``examples/advanced/collab`` with:

.. code-block:: bash

   python -m numpy_async_reproducible.job --num-clients 2 --run-seed 10

The :github_nvflare_link:`CIFAR-10 replay example
<examples/advanced/collab/pt_async_cifar10_reproducible>` uses the same
``FedBuffRecipe`` and simulator port with a seeded PyTorch trainer and
logical CIFAR data shards. It resets client optimizer and loader state per
assignment so a logical client's result does not depend on its physical worker.

Its ``make_recipe()`` returns a regular ``FedBuffRecipe`` and ``make_plan()``
supplies only the simulator. The equivalent import path is:

.. code-block:: python

   from numpy_async_reproducible.job import make_plan, make_recipe
   from nvflare.recipe import SimEnv

   recipe = make_recipe(run_seed=10)
   run = recipe.execute(SimEnv(num_clients=2, reprod=True, plan=make_plan(10)))

``FedBuffRecipe`` consumes an ``async_execution`` plan through its shared
execution port. ``logical_sequence`` names assignments in reservation order.
``arrival_sequence`` lists assignment IDs in acceptance order, independent of
physical completion order:

.. code-block:: python

   plan = {
       "format_version": 1,
       "async_execution": {
           "logical_sequence": ["a", "b", "c", "d"],
           "arrival_sequence": [1, 0, 2, 3],
           "train_seeds": [11, 12, 13, 14],
       },
   }
   run = recipe.execute(SimEnv(num_clients=2, reprod=True, plan=plan))

The physical sites execute the logical assignments; their names do not
determine the assignment base or acceptance order. An arrival-sequence or start-time plan must have
``num_rounds * buffer_size`` assignments, and its acceptance order must be
compatible with the workflow's concurrency and refill threshold. A logical
client cannot be reserved again while an earlier assignment for that client
remains active, including across separate reservation batches. The runnable
example uses ``["alice", "bob", "bob", "alice"]`` with acceptance order
``[1, 0, 2, 3]`` to satisfy this rule.

An optional ``physical_sites`` list pins each assignment to a named physical
site; without it, available sites take pending work in FIFO order. The pinned
site names must exist in the ``SimEnv`` pool. ``dispatch.jsonl`` records the
actual assignment-to-site mapping in either mode.

``train_seeds`` are passed to the client's published ``train`` method as a
``{physical_site: seed}`` mapping. The client must apply them to its own
training/data RNG. A reused physical worker must reset or separate its data
and optimizer state by logical identity.

For virtual-time execution, replace ``arrival_sequence`` with ``timing``.
Starts are explicit, and delay distributions are sampled independently for
each assignment and phase from the plan seed:

.. code-block:: python

   constant = {"distribution": "constant", "value": 0}
   profile = {
       "download": constant,
       "train": {"distribution": "exponential", "mean": 2},
       "upload": {"distribution": "uniform", "min": 0.1, "max": 0.3},
   }
   plan["async_execution"].pop("arrival_sequence")
   plan["async_execution"]["timing"] = {
       "starts": [0, 0, 5, 8],
       "seed": 7,
       "profiles": {name: profile for name in ("a", "b", "c", "d")},
       "horizon": 20,
   }

The virtual clock advances through start, download, train and upload events.
Assignments bind to the published model at their start event, even if all
physical workers are busy. Upload events release stored results for acceptance.
Events at or after the horizon are not accepted. Starts at one time are
reserved as a batch; at equal times, starts precede phase completions. The
plan must leave enough free workflow slots at each start. A virtual deadline
limits logical progress; it is not a wall-clock timeout for client calls.

An external scheduler can supply one concrete duration per phase and
assignment in ``timing.delays`` in place of ``seed`` and ``profiles``.
For example, ``{"download": 0.1, "train": 2.0, "upload": 0.3}`` specifies
the three phase durations for one assignment. This supports measured traces
and samplers with correlated delays without registering Python code in FLARE.

To specify the exact order at equal times, use ``timing.events`` with
``timing.horizon`` in place of ``timing.starts`` and the delay controls.
Each event contains ``virtual_time``, ``assignment_id`` and ``phase``.
List phases in execution order and start assignments in sequential ID order.
Consecutive starts at the same time form one reservation batch. An upload can
therefore precede a same-time start, allowing that start to bind to the model
published by the upload. Specify ``async_execution.base_versions`` to require
a particular base version for each assignment.

Ordered plans can end after the final publication or include one final phase
at or beyond the horizon. The simulator records that boundary phase and stops
before accepting any upload at that time. ``timing.release_on_train`` allows a
logical client to start again after its training phase. Its pending upload
still occupies a concurrency slot. ``timing.speculative`` allows reservations
within the concurrency limit even when fewer contributions are needed for
the remaining publications. Both options default to false. The
:github_nvflare_link:`FedRevive paper planner <research/fedrevive/schedule.py>`
uses these options to produce a plan before starting SimEnv. It also shows
how to plug in a custom sampler with assignment- and phase-specific RNGs and
export a log-normal delay example as a standard SimEnv plan.

Evidence and Limits
-------------------

FedBuff preflights every plan before the first training call. It applies the
same reservation and acceptance rules using placeholder uploads and a version
counter. Invalid capacity, client eligibility, and base-version expectations
fail before any client trains. Actual logical events are checked against the
preflight during execution. Physical dispatch order can vary with the worker pool.

The reusable input is saved at ``<workspace>/<job>/server/local/simulation_plan.json``.
The run directory ``<workspace>/<job>/server/simulate_job/simulation/`` contains
the expected events in ``preflight.jsonl`` and actual events in
``assignments.jsonl``, ``accepted.jsonl``, ``dispatch.jsonl``, and, for timed
plans, ``phases.jsonl``. After execution, FedBuff reads the saved logical logs
to check completeness and order against the preflight derived from the input
plan. It also checks that the server's saved copy of the input plan still matches
what was loaded at launch. A mismatch fails the run. ``run.jsonl`` records
``logical_trace_verified``, the normalized input's ``plan_sha256``, wall seconds,
and virtual time. By default, ``models/`` and ``results/`` retain intermediate payloads.
Set ``async_execution.retain_artifacts`` to false to keep the traces and reclaim
temporary payloads after use. With preflight, the runner saves uploads only for
planned acceptances and reuses each version's base until its last planned
consumer finishes. Any call still in flight at the stop can read its base until
its result arrives. Unknown plan controls and unused sections fail the run.

This controls logical assignments and server acceptance. Matching numerical
models also requires deterministic initialization, client training, data
loading, optional transforms and reduction behavior. A fixed FedBuff replay
does not assign virtual timestamps; choose a timed plan when the delay regime
is part of the experiment. Stored files are run evidence, not distributed
restart checkpoints.

``SimulationPlanRuntime`` supplies plan claiming, trace writing, model/result
storage and unused-plan checks. ``PlannedAsyncEvents`` supplies virtual events.
For another asynchronous Collab workflow, import ``AsyncExecutionPort`` and
``run_async_execution`` from ``nvflare.collab.replayable_execution``. Supply
the algorithm's state, its existing live loop, Collab submit/receive calls,
result normalization and optional event callbacks:

.. code-block:: python

   from nvflare.collab import collab
   from nvflare.collab.replayable_execution import AsyncExecutionPort, run_async_execution

   port = AsyncExecutionPort(
       state=coordinator,
       clients=physical_clients,
       contribution_budget=max_contributions,
       submit=submit,
       receive=receive,
       live_run=run_live,
       normalize=normalize_result,
       aborted=lambda: collab.is_aborted,
       on_publish=notify_model,
   )
   run_async_execution(port, fl_ctx)

The state exposes ``model``, ``version``, ``active``, ``available_slots``,
``finished``, ``reserve(names, metadata, factory)`` and
``accept(assignment, result)``; ``accept`` returns whether it published a new
model. ``reserve`` must assign sequential IDs starting at zero and return every
requested assignment. An assignment exposes ``assignment_id``,
``logical_name``, ``base_version``, ``metadata`` and its immutable base model.
The port can supply ``make_assignment`` when the algorithm needs a different
assignment class; that factory receives the stored base path instead of an
in-memory model. ``submit(site, assignment, **kwargs)`` schedules a physical
call; ``receive()`` returns ``(site, assignment_id, result, error)``.
``FedBuffWorkflow`` is one implementation of this port. Provide
``make_planning_state`` to enable preflight and ordered timing. This callback
returns a fresh state with the same reservation and publication rules using
only logical state. Its ``accept`` receives a placeholder result. A workflow
that supports speculative reservations or release at training completion
also accepts ``speculative`` or ``released_assignments`` in ``reserve``.
The shared runner owns the live-versus-planned choice; a new algorithm must
still expose these real decisions rather than expecting ``SimEnv`` to infer
them from opaque workflow code. Other workflow families can use the same
simulator-owned runtime once they expose their own assignment, acceptance and
publication boundaries; this release implements the Collab async path only.
