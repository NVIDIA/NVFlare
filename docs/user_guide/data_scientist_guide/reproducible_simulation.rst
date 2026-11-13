.. _reproducible_simulation:

Reproducible Simulation Plans
=============================

``SimEnv(reprod=True, plan=...)`` controls participation and logical execution
without changing a recipe or its exported job. The same job can run in
``ProdEnv`` with live scheduling. A plan is a JSON dictionary or a path to a
JSON file; it is simulator input and is never packaged in the job.

Round Participation
-------------------

Use ``round_participation`` for a round-based controller such as the standard
``FedAvg`` controller. Configure the recipe for two participants per round and
run it with three physical sites:

.. code-block:: python

   from nvflare.recipe import SimEnv

   plan = {
       "format_version": 1,
       "round_participation": {
           "0": ["site-2", "site-1"],
           "1": ["site-3", "site-2"],
       },
   }
   run = recipe.execute(SimEnv(num_clients=3, reprod=True, plan=plan))

The plan must cover every executed round. Each selected site must exist and
the list length must match the controller's expected participant count.
``FedAvg`` checks that every selected contribution was accepted before
publishing a round. Its built-in aggregator supports this check; a custom
aggregator needs its own acceptance adapter. Other controllers can use
``PlannedRoundParticipation`` at their selection and publication boundaries;
its ``record()`` and ``verify()`` methods make accepted participants explicit.

Buffered Asynchronous Execution
-------------------------------

``FedBuffRecipe`` accepts an ``async_execution`` plan. ``logical_sequence``
names assignments in reservation order. ``arrival_sequence`` lists assignment
IDs in acceptance order, independent of physical completion order:

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
determine the assignment base or acceptance order. The plan must have
``num_updates * buffer_size`` assignments, and its acceptance order must be
compatible with the workflow's concurrency and refill threshold. A logical
client cannot have two assignments in the same reservation batch.
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

Evidence and Limits
-------------------

The simulator writes plan traces under
``<workspace>/<job>/server/simulate_job/simulation/``. Round plans write
``rounds.jsonl`` and ``round_acceptance.jsonl``. Buffered plans write
``assignments.jsonl``, ``accepted.jsonl``, and retained ``models/`` and
``results/`` files; timed plans also write ``phases.jsonl`` and
``dispatch.jsonl``. ``run.jsonl`` reports wall seconds and virtual time
separately. Unknown plan controls, unused sections and incomplete planned
rounds fail the run.

This controls participation and server acceptance. Matching numerical models
also requires deterministic initialization, client training, data loading,
optional transforms and reduction behavior. FedAvg's built-in streaming
aggregation still consumes accepted results as callbacks arrive, so its plan
guarantees the selected set, not bitwise equality across different completion
orders. A fixed FedBuff replay does not assign virtual timestamps; choose a
timed plan when the delay regime is part of the experiment. Stored files are
run evidence, not distributed restart checkpoints.

``SimulationPlanRuntime``, ``PlannedRoundParticipation`` and
``PlannedAsyncEvents`` are reusable execution
pieces. FedBuff uses the Collab execution adapter in
``nvflare.collab.replayable_execution``. A different workflow must expose its
assignment, acceptance and publication decisions through an adapter; merely
running in ``SimEnv`` cannot make hidden decisions reproducible.
