.. _api_evolution:

.. _api_selection:

##########################
Choose an NVFLARE API Path
##########################

NVFLARE offers two high-level ways to build federated applications, plus
lower-level Controller and Executor APIs for advanced customization. They
serve different goals and are complementary, not interchangeable defaults.

Choose Collaboration API (Technical Preview) for Research
==========================================================

Use the :ref:`Collaboration API <collab_api>` when you are a researcher
exploring a new federated algorithm, communication pattern, or analysis
workflow. Annotate client functions with ``@collab.publish``, express the
coordinating algorithm as ordinary Python, and use ``CollabRecipe`` to run it.
This keeps experimentation close to the algorithm and avoids writing
Controllers, Executors, Shareables, or ``FLModel`` exchange code.

The Collaboration API is the right starting point when the primary goal is to
quickly test algorithmic ideas. It gives the researcher broad control over what
is exchanged and how sites are coordinated. That flexibility also means the
researcher must define, review, and validate the algorithm's data handling,
failure behavior, and operational assumptions. ``CollabRecipe`` still creates
a normal FLARE job, so it uses the standard runtime, site isolation, and
deployment security controls configured for that environment. Collaboration API
does not prescribe a model-training lifecycle, aggregation strategy, or
recipe-specific resource policy; the researcher owns those choices.

Before selecting this path, review the current
:ref:`Collaboration API limitations <collab_api_limitations>`. In particular,
all participating sites must run a compatible NVFLARE version and authorize
BYOC, and the standard task/result filter pipeline is not available. These
constraints can make Client API + Job Recipe the more appropriate choice even
for a custom application.

Start with the :ref:`Collaboration API guide <collab_api>`, the
:github_nvflare_link:`Hello Collab example
<examples/hello-world/hello-collab>` or the :github_nvflare_link:`advanced
Collaboration examples <examples/advanced/collab>`.

Choose Client API + Job Recipe for Production Use
=================================================

Use the Client API with a supported Job Recipe when you are a data scientist
adapting an existing training application for production-oriented federated
learning. The Client API keeps the local training code in its familiar
framework while FLARE handles the federated training exchange. The Job Recipe
selects and configures a supported FL workflow, such as FedAvg, FedProx,
SCAFFOLD, or Swarm.

The Client API defines the client-side training lifecycle: receive a global
model, run local training or evaluation on site-local data, and send the
resulting ``FLModel`` and metrics. The Job Recipe defines the corresponding
Recipe-managed workflow, including round orchestration, aggregation,
validation, and model selection for the chosen supported workflow. In
centralized recipes, aggregation runs on the server. In Swarm, a participating
aggregation client performs the round orchestration and aggregation. This
separation lets the training script remain focused on framework code while the
job definition owns the FL behavior.

For supported PyTorch Recipe workflows, tensor disk offload can materialize
incoming update tensors through temporary disk storage during aggregation,
reducing peak memory pressure for large models. Provision the disk and memory
on the aggregation location: the server for centralized recipes, or each site
that can be selected as the aggregation client for Swarm. The availability and
configuration of these resource controls depend on the selected Recipe and
exchange format.

This is the recommended path when you need a repeatable job definition,
supported algorithm behavior, local simulation and validation, job export and
submission, and a clear separation between training code and FL job
configuration. It is the best fit for deployments where data isolation,
security policy, and managed lifecycle operations matter. The API choice does
not replace secure provisioning and deployment configuration; those controls
must still be enabled and operated for the target environment.

Start with :ref:`client_api_usage`, :ref:`job_recipe`, and
:ref:`available_recipes`.

Choose Controller + Executor APIs for Maximum Control
=====================================================

Use the lower-level :ref:`Controller <controllers>` and :ref:`Executor
<executor>` APIs when an application needs explicit control over FLARE's task
and component runtime. A Controller defines server-side workflow orchestration,
including custom task scheduling and dispatch, response handling, timeouts, and
failure policy. An Executor implements the corresponding client-side task
contract. This path also exposes lower-level facilities such as
``Shareable``/DXO payloads, task and result filters, events, and custom
components.

This path provides the most flexibility and control, but requires more code
and configuration tied to NVFLARE's component APIs. The application owns the
task contract, workflow state, aggregation and failure semantics, and
integration testing.
Choose it when the supported Recipes do not provide the required workflow or
when direct runtime control is more important than the simpler high-level
programming models.

Start with :ref:`controllers`, :ref:`executor`, and
:ref:`component_configuration`.

Shared Platform Capabilities
============================

All three paths create normal FLARE jobs and can use the same simulation, POC,
or provisioned deployment environments. Secure provisioning, authentication,
authorization, site isolation, and operational lifecycle controls are platform
and deployment capabilities, not benefits exclusive to one API. The choice is
among a research-oriented programming model with researcher-defined behavior,
a production-oriented path with a supported Client API training contract and
Job Recipe workflow, and lower-level component APIs for complete task and
runtime control.

At a Glance
===========

.. list-table:: Choosing between the high-level APIs
   :header-rows: 1
   :widths: 25 35 40

   * - If your primary goal is
     - Start with
     - What you get
   * - Explore a new FL algorithm or communication pattern
     - :ref:`Collaboration API (Technical Preview) <collab_api>`
       (Collab annotations + ``CollabRecipe``)
     - Direct Python expression of server coordination and client calls, with
       maximum algorithmic flexibility.
   * - Federate an existing training application for production use
     - **Client API + Job Recipe**
     - A supported FL workflow, a repeatable job definition, and separation of
       training code from job configuration. The Client API manages the local
       training exchange; the Recipe manages server orchestration and can offer
       workflow-specific resource controls.
   * - Move a research prototype toward a production deployment
     - **Client API + Job Recipe**
     - A supported workflow and a deployment-oriented job contract; retain and
       adapt the validated local training logic from the prototype.
   * - Implement a custom workflow or task contract with maximum runtime control
     - :ref:`Controller <controllers>` + :ref:`Executor <executor>` APIs
     - Explicit server-side task orchestration and client-side task execution,
       with direct access to filters, events, payloads, and custom components.

When in doubt, start with Client API + Job Recipe. Choose Collaboration API
only when the research problem requires control beyond the supported Recipe
workflows. Choose Controller + Executor only when the application requires
lower-level task, filter, event, or component control that the high-level APIs
do not provide.
