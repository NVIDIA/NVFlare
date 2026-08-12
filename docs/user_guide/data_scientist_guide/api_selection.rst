.. _api_selection:

##########################
Choose an NVFLARE API Path
##########################

NVFLARE offers two high-level ways to build federated applications. They serve
different goals and are complementary, not interchangeable defaults.

Choose Collaboration API for Research
=====================================

Use the Collaboration API when you are a researcher exploring a new federated
algorithm, communication pattern, or analysis workflow. Annotate client
functions with ``@collab.publish``, express the coordinating algorithm as
ordinary Python, and use ``CollabRecipe`` to run it. This keeps experimentation
close to the algorithm and avoids writing Controllers, Executors, Shareables,
or ``FLModel`` exchange code.

The Collaboration API is the right starting point when the primary goal is to
quickly test algorithmic ideas. It gives the researcher broad control over what
is exchanged and how sites are coordinated. That flexibility also means the
researcher must define, review, and validate the algorithm's data handling,
failure behavior, and operational assumptions. It is not the recommended
starting point for a production application that needs established conventions
for data isolation, security controls, and managed job lifecycle behavior.

Start with the :github_nvflare_link:`Hello Collab example
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

This is the recommended path when you need a repeatable job definition,
supported algorithm behavior, local simulation and validation, job export and
submission, and a clear separation between training code and FL job
configuration. It is the best fit for deployments where data isolation,
security policy, and managed lifecycle operations matter. The API choice does
not replace secure provisioning and deployment configuration; those controls
must still be enabled and operated for the target environment.

Start with :ref:`client_api_usage`, :ref:`job_recipe`, and
:ref:`available_recipes`.

At a Glance
===========

.. list-table:: Choosing between the high-level APIs
   :header-rows: 1
   :widths: 25 35 40

   * - If your primary goal is
     - Start with
     - What you get
   * - Explore a new FL algorithm or communication pattern
     - **Collaboration API** (Collab annotations + ``CollabRecipe``)
     - Direct Python expression of server coordination and client calls, with
       maximum algorithmic flexibility.
   * - Federate an existing training application for production use
     - **Client API + Job Recipe**
     - A supported FL workflow, a repeatable job definition, and separation of
       training code from job configuration.
   * - Move a research prototype toward a production deployment
     - **Client API + Job Recipe**
     - A supported workflow and a deployment-oriented job contract; retain and
       adapt the validated local training logic from the prototype.

When in doubt, start with Client API + Job Recipe. Choose Collaboration API
only when the research problem requires control beyond the supported Recipe
workflows.
