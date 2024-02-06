#########################
Workflows and Controllers
#########################

A workflow has one or more controllers, each implementing a specific coordination strategy. For example, the ScatterAndGather
(SAG) controller implements a popular strategy that is typically used for the fed-average type of federated training. The
CrossSiteValidation controller implements a strategy to let every client site evaluate every other site's model. You can put together
a workflow that uses any number of controllers.

We have implemented several server controlled federated learning workflows (fed-average, cyclic controller, cross-site evaluation) with the server-side :ref:`controllers <controllers>`.
In these workflows, FL clients get tasks assigned by the controller, execute the tasks, and submit results back to the server.

In certain cases, if the server cannot be trusted, it should not be involved in communication with sensitive information.
To address this concern, NVFlare introduces Client Controlled Workflows (CCWF) to facilitate peer-to-peer communication among clients.

Please refer to the following sections for more details.

.. toctree::
   :maxdepth: 3

   controllers/controllers
   controllers/client_controlled_workflows
