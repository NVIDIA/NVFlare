.. _quickstart:
.. _get_started:
.. _getting_started:

###########
Quick Start
###########

This guide runs a small but complete federation, explains what happened, and
then points you to the right API and execution environment for your own work.

Run Your First Federation
=========================

Create and activate a Python virtual environment. Install NVFLARE with its
PyTorch integration, retrieve the Hello PyTorch example that matches the
installed package revision, and run it with focused progress output:

.. code-block:: bash

   python -m pip install "nvflare[PT]"
   nvflare examples get hello-pt
   cd hello-pt
   python job.py --log_config progress

The default run uses two simulated clients and three federated rounds. It runs
on CPU, downloads no dataset, and uses deterministic synthetic images generated
independently at each client. The progress view keeps warnings and errors visible
while focusing normal output on rounds, client metrics, completion, and result
locations.

The :github_nvflare_link:`Hello PyTorch README
<examples/hello-world/hello-pt/README.md>` is the authoritative reference for
the example's dependencies, options, data design, artifacts, and
troubleshooting.

Understand the Run
==================

The example performs a real federated-learning workflow on one machine:

1. The server sends the current global model to two client tasks.
2. Each client evaluates and trains that model on its own local dataset.
3. Clients return model parameters, metrics, and the completed optimizer-step
   count used by the server to derive aggregation weight; their raw samples
   remain in the client processes.
4. FedAvg combines the updates into a new global model.
5. After the last round, both clients evaluate the persisted final global
   model on data excluded from their local training partitions.

The simulator provides fast local validation, but the example uses the same
Recipe and client application structure that can run in POC and production
environments.

Inspect the Result
==================

The command prints the result directory. For the default simulation it is
``/tmp/nvflare/simulation/hello-pt``. The primary artifacts are:

- ``server/simulate_job/app_server/FL_global_model.pt`` -- the persisted final
  global model.
- ``server/simulate_job/metrics/metrics_summary.json`` -- aggregated
  training-round metrics and available best-model metadata.
- ``server/simulate_job/cross_site_val/cross_val_results.json`` -- the final
  model's evaluation by site.
- ``server/log.txt``, ``site-1/log.txt``, and ``site-2/log.txt`` -- detailed
  diagnostic logs.

Metric values depend on the model, data, initialization, and training choices.
Use the artifacts to verify your run; do not treat the quickstart output as a
benchmark.

Adapt Existing Training Code
============================

For supported projects, start with the maintained Agent Skills. Use the manual
API path when the project uses another framework or needs custom integration
and workflow behavior.

Agent-Assisted Adaptation
-------------------------

NVFLARE Agent Skills convert existing PyTorch, PyTorch Lightning, and Hugging
Face training projects and generate federated-statistics jobs. A request can
state the intended workflow and local validation target, for example:

.. code-block:: text

   I have an existing PyTorch training project in ./source. Convert it to
   federated learning using FedAvg and validate it locally with 2 clients and 2
   rounds of training.

The generated code and validation results remain reviewable project artifacts.
See :doc:`Agent Skills <user_guide/agent_skills/index>` for installation,
supported workflows, validation, and limitations.

Before applying a skill to your own project, try one of the
:github_nvflare_link:`Agent Skills runnable examples
<examples/hello-world/agent-skills>`. They provide standalone PyTorch,
Lightning, Hugging Face, tabular-statistics, and image-statistics starting
projects with synthetic inputs, exact prompts, and local validation paths. For
example:

.. code-block:: bash

   nvflare examples get skill-pytorch-conversion

Follow the downloaded README to install the matching skills and run the prompt
with your coding agent. Other entries in the ``AGENT SKILLS`` group from
``nvflare examples list`` cover the remaining supported workflows.

Manual Adaptation
-----------------

Start with :ref:`API Selection <api_selection>` to choose the manual
integration that fits your code. For a conventional PyTorch training loop, the
Client API adds a model-exchange loop around the application's existing model,
data loading, training, and evaluation code:

.. code-block:: python

   import nvflare.client as flare

   flare.init()
   while flare.is_running():
       input_model = flare.receive()
       model.load_state_dict(input_model.params)

       # Existing local evaluation and training code
       # steps = number of optimizer steps completed this round

       output_model = flare.FLModel(
           params={
               name: value.detach().cpu().clone()
               for name, value in model.state_dict().items()
           },
           metrics={"accuracy": accuracy},
           meta={"NUM_STEPS_CURRENT_ROUND": steps},
       )
       flare.send(output_model)

For FedAvg, ``NUM_STEPS_CURRENT_ROUND`` lets the server weight each update by
the amount of local work completed. The maintained Hello PyTorch client records
this value directly from its local training loop.

Define the collaboration with a Recipe and execute it in an environment:

.. code-block:: python

   from nvflare.app_opt.pt.recipes import FedAvgRecipe
   from nvflare.recipe import SimEnv

   recipe = FedAvgRecipe(
       name="my-fedavg-job",
       min_clients=2,
       num_rounds=5,
       model=MyModel(),
       train_script="train.py",
   )
   run = recipe.execute(SimEnv(num_clients=2))
   result = run.get_result()

Review :ref:`Client API <client_api>` for the exchange lifecycle and
:ref:`Available Recipes <available_recipes>` for maintained workflow builders.

Choose an Execution Environment
===============================

FLARE supports three stages that share the same application structure:

- **Simulator** (:ref:`fl_simulator`) -- Runs server and client logic on one
  system for fast application development and validation.
- **Proof of Concept (POC)** (:ref:`poc_command`) -- Simulates a production
  deployment on one local host using separate server and client processes and
  locally generated startup kits.
- **Production** (:ref:`provisioned_setup`) -- Runs across provisioned sites
  with production identities, authorization, networking, policies, and
  operations.

Start with the Simulator, validate deployment behavior in POC, and then follow
the :ref:`Deployment Overview <deployment_overview>` for a real multi-site
environment. The :github_nvflare_link:`Hello PyTorch environment example
<examples/advanced/hello-pt-environments/README.md>` carries the same learning
application from simulation to POC and an already-running production system.

Discover More Examples
======================

Use the resource that matches what you want to do:

- :ref:`Feature Tutorials <tutorials>` provide focused notebook walkthroughs
  of the Simulator, POC, FLARE API, CLI, Recipe, and logging capabilities.
- The `example catalog <https://nvidia.github.io/NVFlare/catalog/>`_ provides
  runnable workloads and implementation references that can be browsed by
  framework, workflow, and use case.
- :ref:`Self-Paced Training <self_paced_training>` provides a sequential
  curriculum across FL concepts, system operation, security, algorithms, and
  industry applications.
- NVIDIA DLI offers hosted courses for `Introduction to Federated Learning with
  NVIDIA FLARE <https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-FX-28+V1>`_
  and `Decentralized AI at Scale with NVIDIA FLARE
  <https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-FX-29+V1>`_.

The `NVIDIA FLARE website <https://nvidia.github.io/NVFlare>`_ brings these
learning resources together with research, webinars, and events. Visit the
`NVIDIA FLARE developer portal <https://developer.nvidia.com/flare>`_ for the
broader product and developer entry point.

From an installed NVFLARE package, list and retrieve curated examples with:

.. code-block:: bash

   nvflare examples list
   nvflare examples get <example-name>

The CLI retrieves source matched to the installed package revision. After
retrieval, follow that example's README for its exact dependencies and commands.
Feature tutorials and the self-paced course are browsed separately and are not
part of the ``nvflare examples`` catalog.
Useful guide-first paths include:

- :doc:`Agent Skills <user_guide/agent_skills/index>` for supported tabular or
  image datasets, or :ref:`Federated Statistics <federated_statistics>` for a
  manual or custom workflow, followed by a runnable statistics example.
- :doc:`Researcher Guide <user_guide/researcher_guide/index>`, then
  :ref:`Collaboration API <collab_api>`, followed by an algorithm, workflow,
  or research implementation.
- :ref:`Federated LLM Fine-Tuning <llm_fine_tuning>`, followed by a Hugging
  Face or NeMo example.
- :ref:`Security Overview <flare_security_overview>`, followed by differential
  privacy, homomorphic encryption, PSI, or confidential-computing examples.

.. toctree::
   :maxdepth: 1
   :hidden:

   hello-world/hello-pt/index
   hello-world/hello-tf/index
   hello-world/hello-jax/index
   hello-world/hello-huggingface/index
   hello-world/hello-lightning/index
   hello-world/hello-xgboost/index
   hello-world/hello-dp/index
   hello-world/hello-flower/index
   hello-world/hello-lr/index
   hello-world/hello-tabular-stats/index
   hello-world/hello-cyclic/index
