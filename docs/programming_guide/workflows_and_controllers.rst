#########################
Workflows and Controllers
#########################

A workflow has one or more controllers, each implementing a specific coordination strategy. For example, the ScatterAndGather
(SAG) controller implements a popular strategy that is typically used for the fed-average type of federated training. The
CrossSiteValidation controller implements a strategy to let every client site evaluate every other site's model. You can put together
a workflow that uses any number of controllers.

Before version 2.4, all federating learning workflows (fed-average, cyclic controller, cross-site evaluation) were server controlled,
implemented with the server-side :ref:`controllers <controllers>`. In these workflows,
FL clients get tasks assigned by the controller, execute the tasks,
and submit results back to the server. The first section covers the server-side
controller API for server-controlled workflows. The second section covers :ref:`client_controlled_workflows` for
workflows that are controlled by the clients.

.. toctree::
   :maxdepth: 3

   controllers/controllers
   controllers/client_controlled_workflows
