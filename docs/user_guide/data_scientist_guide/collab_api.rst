
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
method as if it were local, and gets back whatever Python objects the client
method returns -- no manual ``Shareable``/``DXO``/``FLModel`` packing.

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

* ``@collab.publish`` marks a client method as callable from the server. Its
  parameters must be fixed (no ``*args``/``**kwargs``).
* ``@collab.main`` marks the server's single entry point. Every Collab
  server object or module must define exactly one.
* ``collab.clients.<method>(...)`` calls the named ``@collab.publish``
  method on every client in parallel and returns a result collection you can
  iterate as ``(client_id, result)`` pairs; a client that raises is recorded
  in ``client_results.failures`` (keyed by client ID) instead of failing the
  whole call.
* ``CollabRecipe`` builds the job from the server and client objects, the
  same way a concrete recipe builds a job from a model and script.

Additional facade members are available inside a running Collab function:

* ``collab.site_name`` -- the current site's identity.
* ``collab.caller`` / ``collab.callee`` -- who invoked this call, and who
  it's calling.
* ``collab.other_clients`` / ``collab.child_clients`` / ``collab.leaf_clients``
  -- site-topology lookups, for calls that fan out beyond the server.
* ``collab.get_app_prop()`` / ``collab.set_app_prop()`` -- per-site
  configuration.
* ``collab.get_prop()`` / ``collab.set_prop()`` -- sharing data within a
  single call context.
* ``@collab.init`` / ``@collab.final`` -- mark methods to run once at
  object setup and teardown.

A Minimal Example
------------------

This is the server and client from the runnable
:github_nvflare_link:`Hello FedAvg with the Collab API
<examples/hello-world/hello-collab>` example, trimmed to the FedAvg loop
itself:

.. code-block:: python

   from nvflare.collab import CollabRecipe, collab, simple_logging
   from nvflare.recipe import SimEnv


   class Trainer:
       @collab.publish
       def train(self, weights=None):
           local_epochs = collab.get_app_prop("local_epochs", 5)
           # ... local training using `weights` as the starting point ...
           return updated_weights, loss


   def weighted_avg(client_results):
       valid = dict(client_results)
       for client_id, error in client_results.failures.items():
           print(f"Warning: {client_id} failed: {error}")
       all_weights = [result[0] for result in valid.values()]
       avg_weights = {k: sum(w[k] for w in all_weights) / len(all_weights) for k in all_weights[0]}
       avg_loss = sum(result[1] for result in valid.values()) / len(valid)
       return avg_weights, avg_loss


   class FedAvg:
       def __init__(self, num_rounds=3):
           self.num_rounds = num_rounds

       @collab.main
       def run(self):
           global_weights = None
           for _round_num in range(self.num_rounds):
               client_results = collab.clients.train(global_weights)
               global_weights, global_loss = weighted_avg(client_results)
           return global_weights


   simple_logging()
   recipe = CollabRecipe(job_name="hello_fedavg", server=FedAvg(), client=Trainer(), min_clients=2)
   run = recipe.execute(SimEnv(clients=recipe.configured_sites()))

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
* ``min_clients`` -- minimum number of clients required for the job to run.
* ``sync_task_timeout`` -- timeout, in seconds, for the underlying
  synchronization task each Collab call rides on.
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
above.

Versioning and Authorization
-----------------------------

The Collaboration API is new in NVFlare 2.9.0. Every site participating in a
Collab job -- server and all clients -- must run NVFlare 2.9.0 or newer:
Collab calls carry a versioned authorization envelope that only 2.9.0+ peers
produce and accept, so there is no mixed-version compatibility path for
Collab jobs specifically (other job types are unaffected).

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
