#########################
Workflows and Controllers
#########################

A workflow has one or more controllers, each implementing a specific coordination strategy. For example, the ScatterAndGather
(SAG) controller implements a popular strategy that is typically used for the fed-average type of federated training. The
CrossSiteValidation controller implements a strategy to let every client site evaluate every other site's model. You can put together
a workflow that uses any number of controllers.

We provide the FLModel-based :ref:`model_controller` which provides a straightforward way for users to write controllers.
We also have the original :ref:`Controller API <controllers>` with more FLARE-specific functionalities, which many of our existing workflows are based upon.

We have implemented several server controlled federated learning workflows (fed-average, cyclic controller, cross-site evaluation) with the server-side controllers.
In these workflows, FL clients get tasks assigned by the controller, execute the tasks, and submit results back to the server.

In certain cases, if the server cannot be trusted, it should not be involved in communication with sensitive information.
To address this concern, NVFlare introduces Client Controlled Workflows (CCWF) to facilitate peer-to-peer communication among clients.


Controllers can be configured in ``config_fed_server.json`` in the workflows section:

.. code-block:: json

  workflows = [
      {
          id = "fedavg_ctl",
          name = "FedAvg",
          args {
              min_clients = 2,
              num_rounds = 3,
              persistor_id = "persistor"
          }
      }
  ]

To configure controllers using the JobAPI, define the controller and send it to the server.
This code will automatically generate the server configuration for the controller:

.. code-block:: python

  controller = FedAvg(
      num_clients=2,
      num_rounds=3,
      persistor_id = "persistor"
  )
  job.to(controller, "server")

Please refer to the following sections for more details about the different types of controllers.

.. toctree::
   :maxdepth: 3

   controllers/model_controller
   controllers/controllers
   controllers/client_controlled_workflows
