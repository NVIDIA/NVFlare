.. _api_evolution:

####################
NVFLARE API Overview
####################

NVFLARE provides two high-level application paths. Choose one based on the
goal of the work, rather than on the ML framework in use.

For the complete decision guide, see :ref:`api_selection`.

High-Level Application APIs
===========================

.. list-table:: Current high-level API paths
   :header-rows: 1
   :widths: 28 32 40

   * - Primary goal
     - Recommended API path
     - Use it when
   * - Research a new FL algorithm, communication pattern, or analysis
       workflow
     - **Collaboration API**: Collab annotations + ``CollabRecipe``
     - You want to express client calls and server coordination directly in
       ordinary Python and rapidly evaluate a new idea. You define and
       validate the algorithm's data handling, failure behavior, and
       operational assumptions.
   * - Adapt an existing training application for production-oriented FL
     - **Client API + Job Recipe**
     - You need a supported FL workflow, a repeatable job definition, local
       validation, export and submission, and clear separation between training
       code and job configuration. This is the recommended path for most data
       scientists.

Collaboration API
-----------------

The Collaboration API is optimized for research flexibility. Client functions
are annotated with ``@collab.publish``; a server-side Python function invokes
them through the Collaboration API; and ``CollabRecipe`` creates the job. Use
this path to explore algorithms beyond the supported Recipe workflows.

Start with the :github_nvflare_link:`Hello Collab example
<examples/hello-world/hello-collab>` or the :github_nvflare_link:`advanced
Collaboration examples <examples/advanced/collab>`.

Client API + Job Recipe
-----------------------

The Client API is optimized for applying FL to an existing ML application. The
training script remains in its framework, while FLARE manages the federated
training exchange. A Job Recipe selects and configures a supported workflow,
such as FedAvg, FedProx, SCAFFOLD, or Swarm.

Start with :ref:`client_api_usage`, :ref:`job_recipe`, and
:ref:`available_recipes`.

Production Considerations
=========================

Client API + Job Recipe is the recommended starting point when data isolation,
security policy, and managed job lifecycle operations are important. Choosing
an API does not itself secure a deployment: configure secure provisioning,
authentication, authorization, network policy, and site operations for the
target environment.

A Collaboration API prototype can be deployed, but it is not the recommended
default for production application development. When moving a successful
research prototype to production, retain and adapt the validated local training
logic with Client API and a supported Job Recipe where possible.

Advanced and Legacy APIs
========================

Controller, Executor, and ``FedJob`` APIs remain available for system
integrators who need custom runtime behavior, component placement, or protocols
outside the high-level application paths. They require more FLARE-specific
knowledge and are not the normal starting point for data scientists or
researchers.

LearnerExecutor/Learner, ModelController, and Job Template CLI are legacy paths
for new applications. Use Client API, Collaboration API, and Job Recipe,
respectively, for new work.
