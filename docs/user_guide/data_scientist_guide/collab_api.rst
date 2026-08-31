
.. _collab_api:

Collaboration API (Technical Preview)
======================================

.. note::

   The Collaboration API ("Collab API") is a **Technical Preview** introduced
   in NVFlare 2.9.0. The API surface documented here may still change in a
   future release.

The Collab API lets you write a federated workflow as direct, agent-to-agent
style function calls between server and client objects, instead of the
task/\ ``Shareable`` exchange used by the :ref:`Client API <client_api>` and
:ref:`Job Recipe API <job_recipe>`. A server method calls a decorated client
method as if it were local, and clients can call published methods on other
clients in the same way. Calls return ordinary Python objects -- no manual
``Shareable``/``DXO``/``FLModel`` packing.

Use the Collab API when your workflow's control flow is naturally expressed
as calls and return values -- for example fan-out/fan-in patterns with custom
per-call logic, split learning, swarm learning with client-to-client calls,
or asynchronous aggregation. For standard FedAvg-shaped training, the
:ref:`Job Recipe API <job_recipe>` remains the simplest path.

Core Concepts
-------------

A Collab job defines a **server object** and a **client object**. Both are
plain Python classes (or plain module-level functions); NVFlare does not
require them to subclass anything.

Decorators
^^^^^^^^^^

* ``@collab.publish`` marks a method as remotely callable through a client
  proxy. It supports both server-to-client and client-to-client calls. Its
  parameters must be fixed (no ``*args``/``**kwargs``).
* ``@collab.main`` marks the server's single entry point. Every Collab
  server object or module must define exactly one.
* ``@collab.init`` / ``@collab.final`` mark methods to run once at object
  setup and teardown; see `Lifecycle Decorators`_ below.

Facade Members
^^^^^^^^^^^^^^

* ``collab.clients`` is a ``ProxyList``. Call a method directly to use default
  options (``collab.clients.train(weights)``), or call the ``ProxyList`` first
  to set options (``collab.clients(timeout=30).train(weights)``). Both forms
  call the named ``@collab.publish`` method on every client in parallel and
  return a result collection you can iterate as ``(client_id, result)`` pairs;
  a client that raises is recorded in ``client_results.failures`` (keyed by
  client ID) instead of failing the whole call.
* ``collab.site_name`` -- the current site's identity.
* ``collab.caller`` / ``collab.callee`` -- the site that invoked the current
  call, and the receiving app or named object at the current site. The callee
  is a fully qualified target such as ``site-1`` or ``site-1.selector``; it is
  not a downstream target.
* ``collab.other_clients`` / ``collab.child_clients`` / ``collab.leaf_clients``
  -- site-topology lookups, for calls that fan out beyond the server.
* ``collab.get_app_prop()`` / ``collab.set_app_prop()`` -- per-site
  configuration.
* ``collab.get_prop()`` / ``collab.set_prop()`` -- sharing data within a
  single call context.

Lifecycle Decorators
^^^^^^^^^^^^^^^^^^^^

``@collab.init`` runs once after a site's Collab app is set up and before its
main or published methods are called. ``@collab.final`` runs during Collab app
teardown after normal execution or controlled end-run cleanup; a hard process
termination cannot guarantee that finalization code runs. A lifecycle method
may take no arguments or declare a ``context`` parameter to receive the active
Collab call context. The runtime invokes lifecycle methods on the primary
server/client object and on every named object registered through
``server_objects`` or ``client_objects``.

See the runnable :github_nvflare_link:`LLM SFT client
<examples/advanced/collab/pt_llm_sft/client.py>` for an ``@collab.init`` method
that prepares site-local data, model, trainer, dataloader, and optimizer once
before training calls begin.

A Minimal Example
------------------

This is the server and client from the runnable
:github_nvflare_link:`Hello FedAvg with the Collab API
<examples/hello-world/hello-collab>` example, trimmed to the FedAvg loop
itself. The placeholder local trainer keeps the snippet executable while the
linked example contains real PyTorch training. For brevity, this version
assumes every successful client contributes the same number of examples and
therefore uses equal client weighting. If site dataset sizes differ, return
each site's sample count and weight its update and loss by that count.

.. code-block:: python

   from nvflare.collab import CollabRecipe, collab, simple_logging
   from nvflare.recipe import SimEnv


   MIN_CLIENTS = 2


   def run_local_training(weights, local_epochs):
       """Replace this no-op placeholder with the application's training loop."""
       updated_weights = dict(weights or {"weight": 0.0})
       for _epoch in range(local_epochs):
           pass  # run one local training epoch and update `updated_weights`
       return updated_weights, 0.0


   class Trainer:
       @collab.publish
       def train(self, weights=None):
           local_epochs = collab.get_app_prop("local_epochs", 5)
           return run_local_training(weights, local_epochs)


   def simple_avg(client_results, min_successes):
       valid = dict(client_results)
       for client_id, error in client_results.failures.items():
           print(f"Warning: {client_id} failed: {error}")
       total_clients = len(valid) + len(client_results.failures)
       if len(valid) < min_successes:
           raise RuntimeError(f"only {len(valid)} of {total_clients} client calls succeeded")
       all_weights = [result[0] for result in valid.values()]
       avg_weights = {k: sum(w[k] for w in all_weights) / len(all_weights) for k in all_weights[0]}
       avg_loss = sum(result[1] for result in valid.values()) / len(valid)
       return avg_weights, avg_loss


   class FedAvg:
       def __init__(self, num_rounds=3, min_successes=MIN_CLIENTS):
           self.num_rounds = num_rounds
           self.min_successes = min_successes

       @collab.main
       def run(self):
           global_weights = None
           for _round_num in range(self.num_rounds):
               client_results = collab.clients.train(global_weights)
               global_weights, _global_loss = simple_avg(client_results, self.min_successes)
           return global_weights


   simple_logging()
   recipe = CollabRecipe(
       job_name="hello_fedavg",
       server=FedAvg(min_successes=MIN_CLIENTS),
       client=Trainer(),
       min_clients=MIN_CLIENTS,
   )
   run = recipe.execute(SimEnv(num_clients=MIN_CLIENTS))

See the full, runnable example (real PyTorch model, per-site epoch
configuration, argument parsing) at the link above.

Server and client objects do not have to be classes -- ``@collab.main`` and
``@collab.publish`` also work on plain module-level functions. When
``server``/``client`` are omitted from ``CollabRecipe``, it uses the calling
module's own ``@collab.main``/``@collab.publish`` functions.

CollabRecipe
------------

``CollabRecipe`` is a :ref:`Recipe <recipe_api>` like any other: it exposes
``export()``/``execute()``, accepts the standard execution environments
(``SimEnv``, ``PocEnv``, ``ProdEnv`` from ``nvflare.recipe``), and returns the
same ``Run`` handle. Constructor arguments a user is expected to set:

* ``job_name`` -- the job's name.
* ``server`` / ``client`` -- the server and client objects (or ``None`` to
  auto-discover module-level ``@collab.main``/``@collab.publish``
  functions).
* ``min_clients`` -- minimum number of clients required for the job to start.
  It does not enforce a successful-response quorum for each group call; the
  application must check that, as ``simple_avg`` does above.
* ``sync_task_timeout`` -- timeout, in seconds, for the startup client
  synchronization and setup tasks. It does not control remote method calls;
  set their ``timeout`` through the per-call options described below.
* ``server_objects`` / ``client_objects`` -- additional named objects (for
  example a model-selector or a strategy object) that a server or client
  method needs to call into beyond the main server/client object.

Because a Collab job runs your own server and client classes, it is
inherently bring-your-own-code (BYOC): ``CollabRecipe`` packages the objects'
source into the job automatically, and each receiving site must have BYOC
authorized.

Use ``recipe.set_per_site_config({site: {name: value}})`` (inherited from the
base ``Recipe``) to deliver different values to different clients; a client
reads its own value with ``collab.get_app_prop(name)``, as in the example
above. ``recipe.configured_sites()`` is also inherited from ``Recipe`` and
returns the site names supplied through per-site configuration. It does not
infer clients from an execution environment; when no per-site configuration
is set, provide ``num_clients`` or ``clients`` directly to ``SimEnv``.

Advanced Usage
--------------

``collab.clients.train(...)`` above uses default call behavior: call every
client, wait for every outcome, and return the successful results while
recording per-client failures in the result collection. Real workflows often
need more control over who is called and how the call behaves.

**Selecting clients**

* ``collab.get_clients(["site-1", "site-2"])`` calls out to a named subset
  instead of every client; it returns a ``ProxyList`` you call the same way
  as ``collab.clients``.
* Integer indexing such as ``collab.clients[0]`` returns an individual client
  proxy. A single proxy's method call goes to that one site only and returns
  its result directly -- no ``(site_name, result)`` collection to unwrap. A
  slice returns a plain Python list, not a callable ``ProxyList``; use
  ``collab.get_clients([...])`` for a callable subset.
* ``collab.other_clients``, ``collab.child_clients``, and
  ``collab.leaf_clients`` scope a call to a topology-based subset (see
  `Core Concepts`_ above); each also supports indexing into an individual
  proxy.

**Call options**

Both ``ProxyList`` (group calls) and an individual ``Proxy`` (single-site
calls) accept call options by calling them before the method name, for
example ``collab.clients(blocking=False, timeout=30).train(weights)``:

* ``blocking`` (group calls only, default ``True``) -- ``True`` waits for
  every result and returns a re-iterable snapshot; ``False`` returns
  immediately with a live, single-pass result stream you iterate as
  results arrive. Exhaust the stream before treating its ``failures``
  collection as complete.
* ``expect_result`` (default ``True``) -- ``False`` is fire-and-forget: the
  call returns ``None`` immediately without waiting on the remote method at
  all.
* ``timeout`` (default 60s) -- maximum seconds to wait for each result.
* ``optional`` (default ``False``) -- for an individual ``Proxy`` call,
  ``True`` turns a failure into a logged warning and a ``None`` result instead
  of raising ``CollabCallError``. Group calls report per-site failures through
  the returned result collection as described below.
* ``secure`` (default ``False``) -- routes the call over point-to-point
  secure messaging; the site's Cell must be configured with certificates,
  or a secure call raises.
* ``target`` -- name of a specific collab object at the remote site to
  call, for a site that registers more than one (via
  ``server_objects``/``client_objects`` on ``CollabRecipe``).
* ``parallel`` (group calls only, default unlimited) -- caps how many
  calls may be in flight for that group call at once.
* ``process_resp_cb`` (group calls only) -- a callback invoked as each
  response arrives, useful for streaming/in-time aggregation instead of
  waiting for the whole group. See the ``async_aggregation`` example
  below.

**Error handling**

An individual ``Proxy`` call failure raises ``CollabCallError`` (site,
function name, and cause) unless ``optional=True`` was set. A group call does
not raise for a per-site call failure: it returns the successful results and
records each failure in the result collection's ``failures`` dict (site name
to ``CollabCallError``), as in ``weighted_avg`` above. Callers must check both
``failures`` and whether any successful results remain before aggregating.
For a nonblocking group call, outcomes continue to populate the live stream
and its ``failures`` dict in the background. Iterate the stream to exhaustion
before inspecting the complete failure set or making a final aggregation
decision:

.. code-block:: python

   client_results = collab.clients(blocking=False).train(weights)
   valid = dict(client_results)  # drains the stream and waits for every outcome
   for client_id, error in client_results.failures.items():
       print(f"Warning: {client_id} failed: {error}")
   if not valid:
       raise RuntimeError("all client calls failed")

Versioning and Authorization
-----------------------------

The Collaboration API is new in NVFlare 2.9.0. Every site participating in a
Collab job -- server and all clients -- must run NVFlare 2.9.0 or newer:
Collab calls carry a versioned authorization envelope that only 2.9.0+ peers
produce and accept, so there is no mixed-version compatibility path for
Collab jobs specifically (other job types are unaffected).

Limitations
-----------

As a Technical Preview, the Collab API has a smaller feature set than the
Client API / Job Recipe API path:

* **No mixed-version jobs.** See `Versioning and Authorization`_ above --
  every site must run 2.9.0 or newer.
* **BYOC authorization required.** ``CollabRecipe`` packages the supplied
  server, client, and named-object source into the job. Every receiving site
  must authorize BYOC for the submitting user or the job cannot run. See
  :ref:`BYOC authorization <troubleshooting_byoc>` for deployment details.
* **No task/result filter pipeline.** Standard NVFlare task and result
  filters (for example differential privacy or homomorphic encryption
  filters) attach to the ``Shareable``/``Task`` exchange, which Collab
  calls bypass entirely. Apply any needed data transformation explicitly
  inside your server/client methods instead.
* **Fixed client roster.** The set of participating clients is established
  once, when the job starts (from the execution environment's site list
  and ``min_clients``); Collab does not support clients dynamically
  joining or leaving mid-run.
* **No automatic retry.** A failed individual call raises
  ``CollabCallError`` immediately (or returns ``None`` if ``optional=True``),
  while a failed group call records the per-site error in the result
  collection's ``failures`` dict. There is no built-in retry, backoff, or
  resume-from-checkpoint for a failed or interrupted call. Retry logic, if
  needed, is the application's responsibility.
* **``secure=True`` requires certificates.** A secure call needs the
  site's Cell configured with certificates; without that, the call raises
  rather than falling back to a non-secure transport.

Examples
--------

* :github_nvflare_link:`Hello FedAvg with the Collab API
  <examples/hello-world/hello-collab>` -- the minimal example above, complete
  and runnable.
* :github_nvflare_link:`Advanced Collab examples <examples/advanced/collab>`
  -- a table of further examples, including:

  * ``simple_split_learning`` -- split learning on MNIST, with a client-side
    bottom model and server-side top model exchanging activations and
    gradients directly.
  * ``async_aggregation`` -- in-time aggregation using a response callback.
  * ``swarm`` -- decentralized swarm learning with client-to-client calls.
  * ``pt_cifar10`` -- synchronous PyTorch FedAvg, FedProx, and SCAFFOLD
    variants with direct client calls.
  * ``pt_async_cifar10`` -- asynchronous PyTorch CIFAR-10 training with
    prepared logical-client shards.
  * ``pt_llm_sft`` -- full-parameter Hugging Face SFT with frequent direct
    PyTorch tensor exchange and server-side FedAvg.

For the design rationale behind the API, see the
:github_nvflare_link:`Collab API design doc
<docs/design/collab_api_design.md>`; for a step-by-step migration from local
training code, see the
:github_nvflare_link:`migration tutorial
<docs/design/collab_api_migration_tutorial.md>`.
