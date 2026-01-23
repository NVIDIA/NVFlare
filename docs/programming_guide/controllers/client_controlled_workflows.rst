.. _client_controlled_workflows:

###########################
Client Controlled Workflows
###########################

Server-based controlling usually assumes that the server is trusted by all clients since results submitted by the FL clients may contain
sensitive information (e.g. trained model weights). The assumption that the server is always trusted may not be true. In case that the
server is not to be trusted, the server must not be involved in communication with sensitive information. To achieve this, NVFlare
introduces Client Controlled Workflows (CCWF) in order to allow peer-to-peer communication among clients.

A federated learning workflow has two aspects that need to be managed: the overall job status management (health of the client sites) and
the training logic management (how and when tasks are assigned). In server-controlled workflows, both aspects are managed by the server.

With client controlled workflows, the learning logic management is done by clients (peers): FL clients conduct the learning control logic
by communicating with other clients without involving the FL server (peer-to-peer). The server's job is now only for the monitoring of the
overall job status - in case any abnormal conditions occur (e.g. a client crashes or gets stuck), so the job can be aborted quickly instead
of running forever.

Client controlled workflows provide the implementation of:

    - A general framework for developing client controlled workflows
    - Three commonly used peer-to-peer workflows:
        - Cyclic learning
        - Swarm learning
        - Cross site model evaluation

************************************************
Client Controlled Workflow Development Framework
************************************************
NVFlare is a multi-job system. A job is submitted to the system. The server schedules and deploys the job to all relevant sites (server and
clients). The framework captures the common patterns for all client controlled workflows:

    - Configuration of the workflow
    - Synchronization of clients before starting the workflow
    - Start the workflow from a specified starting point
    - Monitor overall job progress
    - End the workflow properly

This framework is implemented with two base classes: :class:`nvflare.app_common.ccwf.server_ctl.ServerSideController` and
:class:`nvflare.app_common.ccwf.client_ctl.ClientSideController`. 

Server Side Controller
======================

All FLARE jobs must have a server side controller. With client controlled workflows, the :class:`nvflare.app_common.ccwf.server_ctl.ServerSideController` base class
implements the job lifecycle management that does not involve any sensitive training information. It is the
:class:`nvflare.app_common.ccwf.client_ctl.ClientSideController` (and its subclasses) that controls the execution of training and
sensitive data communications.

All client controlled workflows must have a server side controller that extends this base class.

.. code-block:: python

    class ServerSideController(Controller):
        def __init__(
            self,
            num_rounds: int,
            start_round: int = 0,
            task_name_prefix: str = "wf",
            configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,
            end_workflow_timeout=Constant.END_WORKFLOW_TIMEOUT,
            start_task_timeout=Constant.START_TASK_TIMEOUT,
            task_check_period: float = Constant.TASK_CHECK_INTERVAL,
            job_status_check_interval: float = Constant.JOB_STATUS_CHECK_INTERVAL,
            starting_client=None,
            starting_client_policy: str = DefaultValuePolicy.ANY,
            participating_clients=None,
            result_clients=None,
            result_clients_policy: str = DefaultValuePolicy.ALL,
            max_status_report_interval: float = Constant.PER_CLIENT_STATUS_REPORT_TIMEOUT,
            progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,
            private_p2p: bool = True,
        ):

Init args for ServerSideController
----------------------------------

``num_rounds`` - the number of rounds to be performed. This is a workflow config parameter, and will be sent to all clients.

``start_round`` - the starting round number. This is a workflow config parameter, and will be sent to all clients.

``task_name_prefix`` - the prefix for task names of this workflow. The workflow requires multiple tasks (e.g. config and start) between the server
controller and the client controller. The full names of these tasks are <prefix>_config and <prefix>_start. Subclasses may send additional tasks.
Naming these tasks with a common prefix can make it easier to configure task executors for FL clients: instead of specifying each task name explicitly
for the client side executor in config_fed_client.json, you can simply specify <prefix>_* for that executor. This will route all tasks with the <prefix>
to the specified executor.

``participating_clients`` - the names of the clients that will participate in the job. If None, then all clients will be participants.

``result_clients`` - names of the clients that will receive final learning results. Unlike in server controlled workflows where the final results are sent
to the server and kept by the server, with client controlled workflows, results will only be kept by clients.

``result_clients_policy`` - how to determine result_clients if their names are not explicitly specified. Possible values are:
  - ``ALL`` - all participating clients
  - ``ANY`` - any one of the participating clients
  - ``EMPTY`` - no result_clients
  - ``DISALLOW`` - does not allow implicit - result_clients must be explicitly specified

``configure_task_timeout`` - the amount of time to wait for clients' responses to the config task before timeout.

``starting_client`` - name of the starting client. After all participating clients finished the config task properly, the ServerSideController will send
the task to start the workflow to the specified starting client.

``starting_client_policy`` - how to determine the starting client if the name is not explicitly specified. Possible values are:
  - ``ANY`` - any one of the participating clients (randomly chosen)
  - ``EMPTY`` - no starting client
  - ``DISALLOW`` - does not allow implicit - starting_client must be explicitly specified

``start_task_timeout`` - how long to wait for the starting client to finish the "start" task. If timed out, the job will be aborted. Note that if the
starting_client is not specified, then no start task will be sent.

``max_status_report_interval`` - the maximum amount of time allowed for a client to miss a status report. In other words, if a client fails to report
its status for this much time, the client will be considered in trouble and the job will be aborted.

``progress_timeout``- the maximum amount of time allowed for the workflow to not make any progress. In other words, at least one participating client
must have made progress during this time. Otherwise, the workflow will be considered to be in trouble and the job will be aborted.

``end_workflow_timeout`` - timeout for ending workflow message. 

ServerSideController processing logic
-------------------------------------

The ServerSideController's process logic is as follows:

    - At the start of the job, the server will broadcast config parameters to all participating clients of the job (the <prefix>_config task). This also serves another purpose: making sure that all clients are ready to run this job. If any client fails to retrieve or process the config before timeout, the job will be aborted.
    - If the starting_client is specified, the server will send the <prefix>_start task to the starting client. If the starting client fails to start the workflow, the job is aborted.
    - Waits for the workflow to be completed. During this time, each client should periodically send its status update to the server. If a client fails to send an update for the specified amount of time (max_status_report_interval), the job is aborted. If there is no overall progress from any client for the configured amount of time (progress_timeout), the job is aborted. When a client reports the workflow is all done, the job ends normally.
    - When the job is ended (aborted or normally), send a message to all clients to end the workflow.

Client Side Controller
======================

:class:`nvflare.app_common.ccwf.client_ctl.ClientSideController` is the counterpart of the :class:`nvflare.app_common.ccwf.server_ctl.ServerSideController`
on the client side, implemented as an executor. It collaborates with the ServerSideController to implement job lifecycle management functions
(configuration and starting of the workflow, report job status updates, etc.).
In addition, it also provides convenience methods for common functions (e.g. update status, broadcast final results to result receiving clients)
needed by subclasses that implement concrete workflows.

.. code-block:: python

    class ClientSideController(Executor, TaskController):
        def __init__(
            self,
            task_name_prefix: str,
            learn_task_name=AppConstants.TASK_TRAIN,
            persistor_id=AppConstants.DEFAULT_PERSISTOR_ID,
            shareable_generator_id=AppConstants.DEFAULT_SHAREABLE_GENERATOR_ID,
            learn_task_check_interval=Constant.LEARN_TASK_CHECK_INTERVAL,
            learn_task_ack_timeout=Constant.LEARN_TASK_SEND_TIMEOUT,
            learn_task_abort_timeout=Constant.LEARN_TASK_ABORT_TIMEOUT,
            final_result_ack_timeout=Constant.FINAL_RESULT_SEND_TIMEOUT,
            allow_busy_task: bool = False,
        ):

Init Args:
----------

``task_name_prefix`` - the prefix for task names of this workflow. Unlike server-controlled workflows, with client controlled workflows, clients send tasks to each other. All such tasks are named with this prefix.

``learn_task_name`` - this is the name of the task that is typically executed by a learning executor that may have already been implemented. You can use any existing learning executor with client controlled workflows without having to change it. Simply tell the ClientSideController the name of the learning task.

``persistor_id`` - the ID of the persistor component. The persistor is used to load the initial model and save results (i.e. the best and/or the last model) during the training process. 

``shareable_generator_id`` - the ID of the shareable generator component. The shareable generator is responsible for converting between the learnable object (e.g. a full model) and shareable objects (e.g. the weights to be trained and partial training results like weight diffs).

``learn_task_check_interval`` - the interval for checking a new learning task to execute. Learning tasks are executed in a dedicated thread (one task a time), which periodically checks for the learning task to execute.

``learn_task_ack_timeout`` - the timeout for receiving ack from the client that is assigned the learning task. Learning tasks are assigned from one client to another. When a learning task is received, the receiving client simply queues it for the task execution thread, and then sends an ack to the task sending client. 

``learn_task_abort_timeout`` - the timeout for waiting for the learning task to abort. Under certain circumstances, the currently running learning task needs to be aborted (e.g. when the abort command is received from the user). 

``final_result_ack_timeout`` - the timeout for receiving responses from clients after sending them final results. At the end of the workflow, a client that holds the final results will distribute the final results to all configured "result clients". This arg specifies how long to wait for those clients to acknowledge the recipient of the results.

``allow_busy_task`` - whether to allow a new learning task to be received while still executing the current learning task. If not allowed, the client will report a fatal error to the server to cause the job to be aborted. If allowed, the current learning task is aborted and the newly received task will be executed.

ClientSideController processing logic
-------------------------------------

When the "config" task is received, all configuration parameters are validated and processed. If any error is encountered, error code will be returned to the
server, which will cause the job to be aborted.

When the "start" task is received, the start_workflow method (to be implemented by the subclass) is called. If any error is encountered, error code will be
returned to the server, which will cause the job to be aborted.

Each time when trying to get a task from the server, the current job status report is attached to the ``GetTask`` request.

The :class:`nvflare.app_common.ccwf.client_ctl.ClientSideController` base class provides methods for subclass to update job status. However, job status changes
are not immediately sent to the server. Status changes are only sent with the GetTask requests, which occur periodically. Therefore, it is possible that the
subclass updated the job status multiple times before reporting to the server. Only the last status change is reported to the server. This is okay since the
purpose of status reporting is to let the server know that the job is still progressing.

When the end-of-workflow message is received from the server, it stops the execution of the current learning task, if any.

.. _ccwf_cyclic_learning:

***************
Cyclic Learning
***************

With Cyclic Learning, the learning process is done in several rounds. In each round, participating clients do training in turns,
following a predetermined sequential order. Each client trains from the result received from the previous client in the sequence. 

The starting client is responsible for the initial model, which is loaded by its configured persistor.

When the model is received from the previous client, the following logic is executed:

    - Call the configured shareable generator to convert the received model weights to a Learnable object. This Learnable is the current global model. This step may seem unnecessary, but it is an important step, especially when the model is not PyTorch based, where the Learnable object may not be a simple weight dict.
    - Call the learner executor to execute the training task, which will return its training result.
    - Call the configured shareable generator to apply the training result to the global model learnable object. This will update the global model. Note that this step is necessary in case that the training result only contains weight diff. Weight diff cannot be sent directly to the next client for training.
    - If the client is the last leg in the sequence for this round, and this round is the last round, then the training is all done: broadcast the global model to all configured result clients.
    - If the client is the last leg in the sequence for this round, but this round is not the last round, recompute the client sequence for the next round, based on the configured order policy (fixed or random).
    - Call the shareable generator to convert the global model to shareable model params. This will extract the model params from the Learnable object (which may or may not be a simple weight dict) for the next client's training.
    - Send the model params to the next client in the sequence.

The cyclic learning workflow is implemented with :class:`nvflare.app_common.ccwf.cyclic_server_ctl.CyclicServerController` (as subclass of
:class:`nvflare.app_common.ccwf.server_ctl.ServerSideController`) and :class:`nvflare.app_common.ccwf.cyclic_client_ctl.CyclicClientController`
(as subclass of :class:`nvflare.app_common.ccwf.client_ctl.ClientSideController`).

Cyclic Learning: Server Side Controller
=======================================

.. code-block:: python

    class CyclicServerController(ServerSideController):
        def __init__(
            self,
            num_rounds: int,
            task_name_prefix=Constant.TN_PREFIX_CYCLIC,
            start_task_timeout=Constant.START_TASK_TIMEOUT,
            configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,
            task_check_period: float = Constant.TASK_CHECK_INTERVAL,
            job_status_check_interval: float = Constant.JOB_STATUS_CHECK_INTERVAL,
            participating_clients=None,
            result_clients=None,
            starting_client: str = "",
            max_status_report_interval: float = Constant.PER_CLIENT_STATUS_REPORT_TIMEOUT,
            progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,
            private_p2p: bool = True,
            cyclic_order: str = CyclicOrder.FIXED,
        ):

The only extra init arg is ``cyclic_order``, which specifies how the cyclic sequence is to be computed for each round: fixed order or random order.

Of all the init args, only the ``num_rounds`` must be explicitly specified. All others can take default values:

    - All clients of the job participate
    - Starting client is randomly picked
    - All clients are result clients too - every client will receive the last result
    - The client sequence is fixed for all rounds

Cyclic Learning: Client Side Controller
=======================================

.. code-block:: python

    class CyclicClientController(ClientSideController):
        def __init__(
            self,
            task_name_prefix=Constant.TN_PREFIX_CYCLIC,
            learn_task_name=AppConstants.TASK_TRAIN,
            persistor_id=AppConstants.DEFAULT_PERSISTOR_ID,
            shareable_generator_id=AppConstants.DEFAULT_SHAREABLE_GENERATOR_ID,
            learn_task_check_interval=Constant.LEARN_TASK_CHECK_INTERVAL,
            learn_task_abort_timeout=Constant.LEARN_TASK_ABORT_TIMEOUT,
            learn_task_ack_timeout=Constant.LEARN_TASK_ACK_TIMEOUT,
            final_result_ack_timeout=Constant.FINAL_RESULT_ACK_TIMEOUT,
        ):

There are no extra init args.

On the client side, the workflow requires the following three components:

    - There must be an executor for the specified ``learn_task_name``
    - There must be a persistor component for the specified ``persistor_id``
    - There must be a shareable generator component for the specified ``shareable_generator_id``

You may need to adjust the ``final_result_ack_timeout`` properly if the final result is too large for the default timeout.

Example Cyclic Learning Configuration
=====================================

Cyclic Learning: config_fed_server.json
---------------------------------------

.. code-block:: json

    {
      "format_version": 2,
      "task_data_filters": [],
      "task_result_filters": [],
      "components": [],
      "workflows": [
        {
          "id": "rr",
          "path": "nvflare.app_common.ccwf.CyclicServerController",
          "args": {
            "num_rounds": 10
          }
        }
      ]
    }

Cyclic Learning: config_fed_client.json
---------------------------------------

.. code-block:: json

    {
      "format_version": 2,
      "executors": [
        {
          "tasks": [
            "train"
          ],
          "executor": {
            "path": "nvflare.app_common.ccwf.comps.np_trainer.NPTrainer",
            "args": {}
          }
        },
        {
          "tasks": ["cyclic_*"],
          "executor": {
            "path": "nvflare.app_common.ccwf.CyclicClientController",
            "args": {
              "learn_task_name": "train",
              "persistor_id": "persistor",
              "shareable_generator_id": "shareable_generator"
            }
          }
        }
      ],
      "task_result_filters": [],
      "task_data_filters": [],
      "components": [
        {
          "id": "persistor",
          "path": "nvflare.app_common.np.np_model_persistor.NPModelPersistor",
          "args": {}
        },
        {
          "id": "shareable_generator",
          "path": "nvflare.app_common.ccwf.comps.simple_model_shareable_generator.SimpleModelShareableGenerator",
          "args": {}
        }
      ]
    }

.. note::

    - All tasks prefixed with ``cyclic_`` are routed to the CyclicClientController (which is an executor). 
    - There are two tasks assigned by the CyclicServerController:
        - ``cyclic_config``
        - ``cyclic_start``
    - There are two tasks assigned by clients during the training process:
        - ``cyclic_learn``: this is to ask a client to perform training. 
        - ``cyclic_report_final_learn_result``: this is sent from the client that holds the final result to report the final result to other clients


.. note::

    There is no model-related data in the config and start tasks.


.. note::

    The ``cyclic_learn`` and ``cyclic_rcv_final_learn_result`` contain model data. You can apply ``task_data_filters`` if privacy is a concern (the OUT filter for the sending client, and IN filters for the receiving client).

.. _ccwf_swarm_learning:

**************
Swarm Learning
**************
Swarm learning is a decentralized form of federated learning, wherein the responsibilities of aggregation and model training
control are distributed to all peers rather than consolidated in a central server.

With swarm learning, training is done in multiple rounds. In each round, an aggregator client is randomly chosen from all clients,
and then all training clients perform the training task on the current global model params. Once completed, all clients send their
training results to the designated client for aggregation. The aggregated results are then applied to the current global model,
which will become the base for the next round training. This process repeats until the configured number of rounds are completed.

The starting client is responsible for the initial model, which is loaded by its configured persistor.

At the end of the workflow, the final training result is broadcasted to all clients that are configured to receive final results (the ``result_clients``).

Here is the detailed processing logic of the SwarmClientController:

    - The workflow is started from the starting_client. It loads the initial model using the persistor, and prepares the initial training params using the shareable generator (learnable_to_shareable). 
    - Randomly selects a client as the aggregator for the next round from the configured "aggr_clients" list.
    - Broadcast the "learn" task with training params to all clients configured for training (training_clients) and the aggregation client. The task header contains the aggregation client name, the current round number, among other things.
    - All training clients do training by invoking the executor configured for the ``train`` task.
    - Once completed, all training clients send their results to the aggregation client.
    - When the "learn" task is received, the aggregation client:
        - Calls the shareable generator to compute the current global model based (``shareable_to_learnable``).
        - sets up a Gatherer object to wait for results from training clients. Note that the aggregation client could also be a training client.
    - When a training result is received from another client, the Gatherer object of the aggregation client calls the configured aggregator to accept the result. Events are fired before (``AppEventType.BEFORE_CONTRIBUTION_ACCEPT``) and after (``AppEventType.AFTER_CONTRIBUTION_ACCEPT``) calling the aggregator ``accept`` method. These events are very useful for the implementation of best model selection.
    - After all results are received (or other exit conditions occur such as timeout), the aggregation client:
        - calls the ``aggregate`` method of the aggregator to get the aggregation result. Events are fired before (``AppEventType.BEFORE_AGGREGATION``) and after (``AppEventType.AFTER_AGGREGATION``) the call.
        - Calls the shareable generator to apply the aggregated result to the current global model (``shareable_to_learnable``)
        - If not all rounds are completed, prepare for next round:
            - Randomly selects the aggregation client for the next round
            - Calls the shareable generator to prepare the training params (learnable_to_shareable).
            - Broadcast the "learn" task to other clients for the new round
        - If all rounds are completed:
            - Broadcast the last result to all result_clients
            - Check which client has the best result, and ask that client to distribute the best model to all result_clients.

The swarm learning workflow is implemented with :class:`nvflare.app_common.ccwf.swarm_server_ctl.SwarmServerController` (as subclass of
:class:`nvflare.app_common.ccwf.server_ctl.ServerSideController`) and :class:`nvflare.app_common.ccwf.swarm_client_ctl.SwarmClientController`
(as subclass of :class:`nvflare.app_common.ccwf.client_ctl.ClientSideController`).

Best Model Selection
====================
Optionally, a model selection widget can be used to determine the best global model, just as in the server-controlled
fed-average workflow (SAG). The widget listens to the BEFORE and AFTER events of ``accept`` and ``aggregate`` calls of the
aggregator to dynamically compute the aggregated validation metrics reported from the training clients. When a better
metric is achieved, it fires the ``AppEventType.GLOBAL_BEST_MODEL_AVAILABLE`` event with the best metric value. If the
persistor listens to this event, it can persist the current global model (the current best).

However, unlike the server-controlled SAG where the aggregation is always done on the server and hence only a single
global model is present at any time, many clients could do aggregation during the course of swarm learning. Each aggregation
client could have its own so-called best global model computed by its model selector. We need to find the best of these best
global models. This is achieved as follows:

    - Use the ``learn`` task header to remember the current global best (metric value and name of the client that holds the model). Initially both are None.
    - The SwarmClientController listens to the ``AppEventType.GLOBAL_BEST_MODEL_AVAILABLE`` event. When this event is fired, compare the metric value against the current best value in the task header (if any). Update the task header if the new value is better. This header info will be carried to the next ``learn`` task.
    - Eventually only the global best (if available) will be distributed to result clients.

Swarm Learning: Server Side Controller
======================================

.. code-block:: python

    class SwarmServerController(ServerSideController):
        def __init__(
            self,
            num_rounds: int,
            start_round: int = 0,
            task_name_prefix=Constant.TN_PREFIX_SWARM,
            start_task_timeout=Constant.START_TASK_TIMEOUT,
            configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,
            task_check_period: float = Constant.TASK_CHECK_INTERVAL,
            job_status_check_interval: float = Constant.JOB_STATUS_CHECK_INTERVAL,
            participating_clients=None,
            result_clients=None,
            starting_client: str = "",
            max_status_report_interval: float = Constant.PER_CLIENT_STATUS_REPORT_TIMEOUT,
            progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,
            aggr_clients=None,
            train_clients=None,
        ):

The default value of the task name prefix is "swarm".

The additional init args are:

    - ``aggr_clients``: the clients to do aggregation. If not specified, all participating clients are aggregation clients.
    - ``train_clients``: clients to do training. If not specified, all participating clients are training clients.

Swarm Learning: Client Side Controller
======================================

.. code-block:: python

    class SwarmClientController(ClientSideController):
        def __init__(
            self,
            task_name_prefix=Constant.TN_PREFIX_SWARM,
            learn_task_name=AppConstants.TASK_TRAIN,
            persistor_id=AppConstants.DEFAULT_PERSISTOR_ID,
            shareable_generator_id=AppConstants.DEFAULT_SHAREABLE_GENERATOR_ID,
            aggregator_id=AppConstants.DEFAULT_AGGREGATOR_ID,
            metric_comparator_id=None,
            learn_task_check_interval=Constant.LEARN_TASK_CHECK_INTERVAL,
            learn_task_abort_timeout=Constant.LEARN_TASK_ABORT_TIMEOUT,
            learn_task_ack_timeout=Constant.LEARN_TASK_ACK_TIMEOUT,
            learn_task_timeout=None,
            final_result_ack_timeout=Constant.FINAL_RESULT_ACK_TIMEOUT,
            min_responses_required: int = 1,
            wait_time_after_min_resps_received: float = 10.0,
        ):

On the client side, the workflow requires the following three components:

    - There must be an executor for the specified ``learn_task_name``
    - There must be a persistor component for the specified ``persistor_id``
    - There must be a shareable generator component for the specified ``shareable_generator_id`` 
    - There must be an aggregator component for the specified ``aggregator_id``.
    - An optional Metric Comparator, if ``metric_comparator_id`` is specified. Since the metric value can be of any type, and the Swarm Learning workflow needs to be able to compare the current best metric against the computed metric values, the Metric Comparator will help with the comparison operation. If this arg is not set, the ``NumberMetricComparator`` will be used, which assumes that the metric value is a simple number.

The aggregation behavior is configured by the following args:

    - ``min_responses_required`` - the minimum number of responses required before exiting the gathering
    - ``wait_time_after_min_resps_received`` - how many seconds to wait for potentially more responses after minimum responses are received
    - ``learn_task_timeout`` - how long to wait for the current learn task before timing out the gathering

Example Swarm Learning
======================

This section shows how to set up swarm learning using recipes (recommended) and the traditional JSON configuration.

Using Recipes (Recommended)
---------------------------

Use ``SimpleSwarmLearningRecipe`` for a streamlined swarm learning setup:

.. code-block:: python

    from nvflare.app_common.ccwf.recipes.swarm import SimpleSwarmLearningRecipe
    from nvflare.recipe.sim_env import SimEnv

    # Create swarm learning recipe
    recipe = SimpleSwarmLearningRecipe(
        name="swarm_learning",
        initial_model=MyModel(),
        num_rounds=10,
        train_script="train.py",
        train_args={"batch_size": 32, "epochs": 5},
    )

    # Configure large model parameters if needed (server-side only)
    recipe.add_server_config({
        "np_download_chunk_size": 2097152,
        "tensor_download_chunk_size": 2097152,
        "streaming_per_request_timeout": 600
    })

    # Run in simulation
    env = SimEnv(num_clients=3)
    recipe.execute(env)

For advanced customization, use ``BaseSwarmLearningRecipe`` with explicit server and client configurations:

.. code-block:: python

    from nvflare.app_common.ccwf.recipes.swarm import BaseSwarmLearningRecipe
    from nvflare.app_common.ccwf.ccwf_job import SwarmServerConfig, SwarmClientConfig

    server_config = SwarmServerConfig(
        num_rounds=10,
        start_task_timeout=300,
        progress_timeout=7200,
    )

    client_config = SwarmClientConfig(
        executor=my_executor,
        aggregator=my_aggregator,
        persistor=my_persistor,
        shareable_generator=my_generator,
    )

    recipe = BaseSwarmLearningRecipe(
        name="custom_swarm",
        server_config=server_config,
        client_config=client_config,
    )

Using JSON Configuration (Advanced)
-----------------------------------

For users who need fine-grained control, here is the equivalent JSON configuration.

**config_fed_server.json:**

.. code-block:: json

    {
      "format_version": 2,
      "task_data_filters": [],
      "task_result_filters": [],
      "components": [],
      "workflows": [
        {
          "id": "swarm_controller",
          "path": "nvflare.app_common.ccwf.SwarmServerController",
          "args": {
            "num_rounds": 10
          }
        }
      ]
    }

.. note::

    The only required arg is ``num_rounds``.

**config_fed_client.json:**

.. code-block:: json

    {
      "format_version": 2,
      "executors": [
        {
          "tasks": [
            "train"
          ],
          "executor": {
            "path": "nvflare.app_common.ccwf.comps.np_trainer.NPTrainer",
            "args": {}
          }
        },
        {
          "tasks": ["swarm_*"],
          "executor": {
            "path": "nvflare.app_common.ccwf.SwarmClientController",
            "args": {
              "learn_task_name": "train",
              "learn_task_timeout": 5.0,
              "persistor_id": "persistor",
              "aggregator_id": "aggregator",
              "shareable_generator_id": "shareable_generator",
              "min_responses_required": 2,
              "wait_time_after_min_resps_received": 1
            }
          }
        }
      ],
      "task_result_filters": [],
      "task_data_filters": [],
      "components": [
        {
          "id": "persistor",
          "path": "nvflare.app_common.ccwf.comps.np_file_model_persistor.NPFileModelPersistor",
          "args": {}
        },
        {
          "id": "shareable_generator",
          "name": "FullModelShareableGenerator",
          "args": {}
        },
        {
          "id": "aggregator",
          "name": "InTimeAccumulateWeightedAggregator",
          "args": {
            "expected_data_kind": "WEIGHT_DIFF"
          }
        },
        {
          "id": "model_selector",
          "name": "IntimeModelSelector",
          "args": {}
        }
      ]
    }

.. note::

    - All tasks prefixed with ``swarm_`` are routed to the :class:`nvflare.app_common.ccwf.swarm_client_ctl.SwarmClientController` (which is an executor). 
    
.. note::

    - There are two tasks assigned by the :class:`nvflare.app_common.ccwf.swarm_server_ctl.SwarmServerController`:
        - swarm_config
        - swarm_start
    
.. note::

    - There are several tasks assigned by clients during the training process:
        - swarm_learn: this is to ask a client to perform training. 
        - swarm_report_learn_result: this is sent from a training client to the aggregation client to report its training result.
        - swarm_report_final_learn_result: this is sent from the client that holds the final results (last and/or best global model) to report final results to other clients


.. note::

    There is no model-related data in the swarm_config and swarm_start tasks.


.. note::

    Client assigned tasks contain model data. You can apply task_data_filters if privacy is a concern (the OUT filter for the sending client, and IN filters for the receiving client).

.. _swarm_learning_large_models:

Swarm Learning Parameters for Large Models
==========================================

When running Swarm Learning with large models (e.g., LLMs), you may need to tune various timeout and chunking parameters
to accommodate the larger payloads and longer processing times.

Default Timeout Values
----------------------

The following table lists all default timeout values used in Swarm Learning. Understanding these defaults helps you
determine which parameters need adjustment for your large model workloads.

**How to Override These Values:**

These timeout values can be overridden in your job configuration files:

- **Client-side parameters**: Set in ``config_fed_client.conf`` (or ``.json``) under the ``SwarmClientController`` executor args.
- **Server-side parameters**: Set in ``config_fed_server.conf`` (or ``.json``) under the ``SwarmServerController`` workflow args.
- **Global streaming parameters**: Set at the top level of your config files (e.g., ``np_download_chunk_size``, ``tensor_download_chunk_size``).

For **Recipe API users** (recommended), use the ``add_server_config()`` method:

.. code-block:: python

    # Add streaming parameters to server config (server-side only)
    recipe.add_server_config({
        "np_download_chunk_size": 2097152,
        "tensor_download_chunk_size": 2097152,
        "streaming_per_request_timeout": 600
    })

For **Job API users**, use ``job.to_server()`` with a dict:

.. code-block:: python

    job.to_server({"np_download_chunk_size": 2097152, "streaming_per_request_timeout": 600})

.. list-table:: Swarm Learning Default Timeouts
   :header-rows: 1
   :widths: 25 10 25 40

   * - Constant Name
     - Default
     - Config Parameter
     - Description
   * - ``CONFIG_TASK_TIMEOUT``
     - 300
     - ``config_task_timeout`` (server)
     - Time allowed for clients to respond to the configuration task at job start.
   * - ``START_TASK_TIMEOUT``
     - 10
     - ``start_task_timeout`` (server)
     - Time allowed for the starting client to begin the workflow.
   * - ``END_WORKFLOW_TIMEOUT``
     - 2.0
     - ``end_workflow_timeout`` (server)
     - Time allowed for ending workflow message acknowledgment.
   * - ``TASK_CHECK_INTERVAL``
     - 0.5
     - ``task_check_interval`` (client)
     - Interval between task status checks.
   * - ``JOB_STATUS_CHECK_INTERVAL``
     - 2.0
     - ``job_status_check_interval`` (server)
     - Interval between job status checks by the server.
   * - ``PER_CLIENT_STATUS_REPORT_TIMEOUT``
     - 90.0
     - (internal)
     - Max time a client can go without reporting status.
   * - ``WORKFLOW_PROGRESS_TIMEOUT``
     - 3600.0
     - ``progress_timeout`` (server)
     - Max time allowed without any workflow progress.
   * - ``LEARN_TASK_CHECK_INTERVAL``
     - 1.0
     - ``learn_task_check_interval`` (client)
     - Interval for checking new learning tasks.
   * - ``LEARN_TASK_ACK_TIMEOUT``
     - 10
     - ``learn_task_ack_timeout`` (client)
     - Time allowed for a client to acknowledge receipt of a learn task.
   * - ``LEARN_TASK_ABORT_TIMEOUT``
     - 5.0
     - ``learn_task_abort_timeout`` (client)
     - Time allowed for a learning task to abort when requested.
   * - ``FINAL_RESULT_ACK_TIMEOUT``
     - 10
     - ``final_result_ack_timeout`` (client)
     - Time allowed for clients to acknowledge receipt of final results.
   * - ``GET_MODEL_TIMEOUT``
     - 10
     - ``get_model_timeout`` (client)
     - Time allowed for retrieving a model from another client.
   * - ``MAX_TASK_TIMEOUT``
     - 3600
     - ``learn_task_timeout`` (client)
     - Maximum time allowed for any single task to complete.

Client-Side Parameters
----------------------

The following SwarmClientController parameters are particularly important for large models:

**Timeouts and Flow Control:**

- ``learn_task_timeout``: Upper bound for how long the aggregation client waits for a round to finish. **Default: None (uses MAX_TASK_TIMEOUT=3600)**. **Suggested: 3600 to 7200** for large models.
- ``learn_task_ack_timeout``: Timeout for acknowledging learn task dispatch. **Default: 10**. **Suggested: 300 or higher** since large model initialization can be slow.
- ``final_result_ack_timeout``: Timeout for ACKs after broadcasting final results. **Default: 10**. **Suggested: 300 to 600** as final result distribution is often the largest payload.
- ``request_to_submit_result_msg_timeout``: Timeout for request-to-submit messages. **Default: 5.0**. **Suggested: 10 to 30**.
- ``request_to_submit_result_interval``: Retry interval when submit permission is not granted. **Default: 1.0**. **Suggested: 2 to 5**.
- ``request_to_submit_result_max_wait``: Max total wait time for submit permission. **Default: None**. **Suggested: 600 to 1200** for large models.
- ``max_concurrent_submissions``: Maximum concurrent submissions. **Default: 1**. **Suggested: 1** to reduce memory pressure.
- ``min_responses_required``: Minimum client results required to begin aggregation. **Default: 1**. **Suggested: 2** for 3-client runs.
- ``wait_time_after_min_resps_received``: Extra wait time after minimum responses. **Default: 10.0**. **Suggested: 120 to 300**.

**Example client config for large models:**

.. code-block::

    executors = [
      {
        tasks = ["swarm_*"]
        executor {
          path = "nvflare.app_common.ccwf.SwarmClientController"
          args {
            learn_task_timeout = 3600
            learn_task_ack_timeout = 300
            final_result_ack_timeout = 300
            request_to_submit_result_msg_timeout = 10.0
            request_to_submit_result_interval = 2.0
            request_to_submit_result_max_wait = 600.0
            max_concurrent_submissions = 1
            min_responses_required = 2
            wait_time_after_min_resps_received = 120
          }
        }
      }
    ]

**Download and Chunking Behavior:**

- ``np_download_chunk_size``: Chunk size for numpy array downloads. **Default: 2097152 (2MB)**. Value 0 disables streaming and uses native serialization which can spike memory.
- ``tensor_download_chunk_size``: Chunk size for PyTorch tensor downloads. **Default: 2097152 (2MB)**. Value 0 disables streaming.

.. code-block::

    np_download_chunk_size = 2097152
    tensor_download_chunk_size = 2097152

Server-Side Parameters
----------------------

**SwarmServerController:**

- ``num_rounds``: Total number of training rounds.
- ``start_task_timeout``: Timeout for starting the workflow. **Default: 10 (START_TASK_TIMEOUT)**. **Suggested: 300** for large model initialization.
- ``progress_timeout``: Overall workflow progress timeout. **Default: 3600.0 (WORKFLOW_PROGRESS_TIMEOUT)**. **Suggested: 7200 or higher** for large models.

**Example server config for large models:**

.. code-block::

    workflows = [
      {
        id = "swarm_controller"
        path = "nvflare.app_common.ccwf.SwarmServerController"
        args {
          num_rounds = 25
          start_task_timeout = 300
          progress_timeout = 7200
        }
      }
    ]

**CrossSiteEvalServerController (if enabled):**

- ``eval_task_timeout``: Timeout for evaluation tasks. **Default: 300 (CONFIG_TASK_TIMEOUT)**. **Suggested: 1200** for large models.

Optional NVFlare Global Config
------------------------------

These framework-level settings affect large payload transfers:

- ``streaming_per_request_timeout``: Per-request timeout for streaming downloads. **Default: 300**. **Suggested: 300 to 600** for large models.

.. code-block::

    streaming_per_request_timeout: 300

Recommended Minimal Parameter Set
---------------------------------

If you only adjust a few parameters for large models, start with:

1. ``learn_task_timeout`` - Ensures rounds have enough time to complete
2. ``final_result_ack_timeout`` - Allows time for large result distribution
3. ``request_to_submit_result_max_wait`` - Provides adequate aggregation window
4. ``progress_timeout`` - Prevents premature workflow termination
5. ``np_download_chunk_size`` and ``tensor_download_chunk_size`` - Enables memory-efficient streaming

.. _ccwf_cross_site_evaluation:

*********************
Cross Site Evaluation
*********************

The purpose of the cross site evaluation (CSE) workflow is to let client sites evaluate each other's models. Optionally, additional global models could also be evaluated by clients.

In server-controlled CSE, each site sends its model to the server first, and the server will broadcast the model to other sites to evaluate. The server could also send additional server-owned models to other sites to evaluate. All model evaluation results are sent back to the server so that the user can access the results easily.

In client-controlled CSE, client models do not go to the server for distribution. Instead, clients communicate directly with each other to share their models for validation. Model evaluation results are still sent to the server to allow the user easy access to the results.

There are a few concepts in client-controlled CSE:

  - Evaluators - clients that will evaluate models and produce evaluation metrics.
  - Evaluatees - clients that have local models to be evaluated
  - Global Model Client - the client that has global model(s) to be evaluated

The CSE client controlled workflow can be used for the evaluation of both local and/or global models. 

Here is the detailed control logic:

  - Server broadcasts the "config" task to all clients. The config contains information about who are the evaluators and evaluatees, and which client is the global model client.
  - Each client processes the config info. If the client is configured to be the global model client, it sends global model names to the server. If the client is configured to be an evaluator, it checks to see whether it has the evaluation capability. If not, it reports an error to the server. If the client is configured to be an evaluatee, it checks to see whether it has a local model. If not, it reports an error to the server.
  - The server processes configuration responses from all clients. If any error is reported, the job is aborted.
  - The server first tries to evaluate global models if the global model client has reported any model names. For each global model name, the server broadcasts an "eval" request to all evaluators to evaluate the model. The request only contains the name of the model, and the name of the client that has the model.
  - The server then tries to evaluate clients' local models. For each client configured to be evaluatee, the server broadcasts an "eval" request to all evaluators. The request contains the evaluatee's name.
  - On the client side, when an "eval" request is received, it wii:
  - send the "get_model" task to the client that has the model. 
  - perform the "validate" method on the received model.
  - Send the result back to the server
  - One the client side, when the "get_model" task is received, it will locate the model depending on the type of the model:
  - For global models, it calls the persistor object to locate the model
  - For the local model, it calls the executor configured for the "submit_model" task.
  - On the Server side, when an evaluation result is received, it will:
  - Fire the AppEventType.VALIDATION_RESULT_RECEIVED event type to allow other widgets to process the result
  - Save it in the job's workspace using the same folder structure as in the Server-controlled CSE.

The CSE workflow is implemented with :class:`nvflare.app_common.ccwf.cse_server_ctl.CrossSiteEvalServerController` (as subclass of
:class:`nvflare.app_common.ccwf.server_ctl.ServerSideController`) and :class:`nvflare.app_common.ccwf.cse_client_ctl.CrossSiteEvalClientController`
(as subclass of :class:`nvflare.app_common.ccwf.client_ctl.ClientSideController`).

Cross Site Evaluation: Server Side Controller
=============================================

.. code-block:: python

    class CrossSiteEvalServerController(ServerSideController):
        def __init__(
            self,
            task_name_prefix=Constant.TN_PREFIX_CROSS_SITE_EVAL,
            start_task_timeout=Constant.START_TASK_TIMEOUT,
            configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,
            eval_task_timeout=30,
            task_check_period: float = Constant.TASK_CHECK_INTERVAL,
            job_status_check_interval: float = Constant.JOB_STATUS_CHECK_INTERVAL,
            progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,
            participating_clients=None,
            evaluators=None,
            evaluatees=None,
            global_model_client=None,
            max_status_report_interval: float = Constant.PER_CLIENT_STATUS_REPORT_TIMEOUT,
            eval_result_dir=AppConstants.CROSS_VAL_DIR,
        ):

The default value of the task name prefix is "cse".

The additional init args are:

``eval_task_timeout`` - max time allowed for the evaluation of a model by clients.

``evaluators`` - clients that will evaluate models. By default all clients are evaluators.

``evaluatees`` - clients whose models will be evaluated. By default all clients are evaluatees. If no local models are to be evaluated, you can configure this arg to the special value "@none".

``global_model_client`` - the client that has the global models to be evaluated. By default, a random client is selected from the list of clients. If you don't want to evaluate global models, you can set this arg to the special value "@none".

You cannot set both ``evaluatees`` and ``global_model_client`` to "@none".


Cross Site Evaluation: Client Side Controller
=============================================


.. code-block:: python

    class CrossSiteEvalClientController(ClientSideController):
        def __init__(
            self,
            task_name_prefix=Constant.TN_PREFIX_CROSS_SITE_EVAL,
            submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
            validation_task_name=AppConstants.TASK_VALIDATION,
            persistor_id=AppConstants.DEFAULT_PERSISTOR_ID,
            get_model_timeout=Constant.GET_MODEL_TIMEOUT,
        ):

The default value of the task name prefix is "cse".

The additional init args are:

``submit_model_task_name`` - the task name for submitting a model. This must map to a trainer executor that already supports submitting the local best model.

``validation_task_name`` - the task name for validating a model. This must map to a trainer executor that already supports model validation.

``get_model_timeout`` - When client X tries to evaluate the model of client Y, client X first sends a request to Y to ask for the model. This arg sets the timeout for this request.

Model Persistor
---------------
The CSE workflow requires the global model client to have a Model Persistor that implements the ``get_model_inventory`` method.
This method is called to return the names of available global models. The persistor must also implement the ``get_model`` method,
which is called to get the model from the persistor for other clients to evaluate.

Example Cross Site Evaluation
=============================

This section shows how to set up cross-site evaluation using recipes (recommended) and the traditional JSON configuration.

Using Recipes (Recommended)
---------------------------

**Swarm Learning with Cross-Site Evaluation:**

Use ``SimpleSwarmLearningRecipe`` for swarm learning with optional cross-site evaluation:

.. code-block:: python

    from nvflare.app_common.ccwf.recipes.swarm import SimpleSwarmLearningRecipe
    from nvflare.recipe.sim_env import SimEnv

    # Create swarm learning recipe with cross-site evaluation enabled
    recipe = SimpleSwarmLearningRecipe(
        name="swarm_with_cse",
        initial_model=MyModel(),
        num_rounds=3,
        train_script="train.py",
        do_cross_site_eval=True,
        cross_site_eval_timeout=300,
    )

    # Configure large model parameters if needed (server-side only)
    recipe.add_server_config({
        "np_download_chunk_size": 2097152,
        "tensor_download_chunk_size": 2097152,
        "streaming_per_request_timeout": 600
    })

    # Run in simulation
    env = SimEnv(num_clients=3)
    recipe.execute(env)

.. note::

    This recipe uses the CCWF peer-to-peer cross-site evaluation where clients evaluate each other's
    models directly. For the traditional server-controlled cross-site evaluation, see
    :ref:`cross_site_model_evaluation`.

Using JSON Configuration (Advanced)
-----------------------------------

For users who need fine-grained control, here is the equivalent JSON configuration.

Cross Site Evaluation: config_fed_server.json
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

    {
      "format_version": 2,
      "task_data_filters": [],
      "task_result_filters": [],
      "components": [
        {
          "id": "json_generator",
          "name": "ValidationJsonGenerator",
          "args": {}
        }
      ],
      "workflows": [
        {
          "id": "swarm_controller",
          "path": "nvflare.app_common.ccwf.SwarmServerController",
          "args": {
            "num_rounds": 3
          }
        },
        {
          "id": "cross_site_eval",
          "path": "nvflare.app_common.ccwf.CrossSiteEvalServerController",
          "args": {
          }
        }
      ]
    }


.. note::

    The json_generator component is used to also create a JSON file at the end of the job that
    shows cross-site validation results in human readable format.

Cross Site Evaluation: config_fed_client.json
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

    {
      "format_version": 2,
      "executors": [
        {
          "tasks": [
            "train", "submit_model", "validate"
          ],
          "executor": {
            "path": "nvflare.app_common.ccwf.comps.np_trainer.NPTrainer",
            "args": {}
          }
        },
        {
          "tasks": ["swarm_*"],
          "executor": {
            "path": "nvflare.app_common.ccwf.SwarmClientController",
            "args": {
              "learn_task_name": "train",
              "learn_task_timeout": 5.0,
              "persistor_id": "persistor",
              "aggregator_id": "aggregator",
              "shareable_generator_id": "shareable_generator",
              "min_responses_required": 2,
              "wait_time_after_min_resps_received": 1
            }
          }
        },
        {
          "tasks": ["cse_*"],
          "executor": {
            "path": "nvflare.app_common.ccwf.CrossSiteEvalClientController",
            "args": {
              "submit_model_task_name": "submit_model",
              "validation_task_name": "validate",
              "persistor_id": "persistor"
            }
          }
        }
      ],
      "task_result_filters": [],
      "task_data_filters": [],
      "components": [
        {
          "id": "persistor",
          "path": "nvflare.app_common.ccwf.comps.np_file_model_persistor.NPFileModelPersistor",
          "args": {}
        },
        {
          "id": "shareable_generator",
          "name": "FullModelShareableGenerator",
          "args": {}
        },
        {
          "id": "aggregator",
          "name": "InTimeAccumulateWeightedAggregator",
          "args": {
            "expected_data_kind": "WEIGHT_DIFF"
          }
        },
        {
          "id": "model_selector",
          "name": "IntimeModelSelector",
          "args": {}
        }
      ]
    }

.. note::

      - All tasks prefixed with ``cse_`` are routed to the :class:`nvflare.app_common.ccwf.cse_client_ctl.CrossSiteEvalClientController` (which is an executor). 
      - The following tasks are assigned by the :class:`nvflare.app_common.ccwf.cse_server_ctl.CrossSiteEvalServerController`:
        - ``cse_config``
        - ``cse_eval``
      - The following task is assigned by clients during the training process:
        - ``cse_ask_for_model``: this is sent from a client to another client to ask for its model for evaluation.

.. note::

    There is no "start" task in this workflow.

.. note::

    There is no sensitive model data in the ``cse_config`` and ``cse_eval`` tasks.

.. note::

    The response to the ``ask_for_model`` task contains model data. You can apply ``task_result_filters`` if privacy is a concern (the OUT filter for the responding client, and IN filters for the requesting client).

Cross Site Evaluation Parameters for Large Models
=================================================

When running Cross Site Evaluation with large models, you may need to adjust timeout parameters to accommodate larger model transfers and longer evaluation times.

Server-Side Parameters
----------------------

**CrossSiteEvalServerController:**

- ``eval_task_timeout``: Max time allowed for the evaluation of a model by clients. **Suggested: 1200 or higher** for large models, as evaluation can be expensive.
- ``configure_task_timeout``: Timeout for configuration task. **Suggested: 300** for large model initialization.
- ``progress_timeout``: Overall workflow progress timeout. **Suggested: 7200 or higher** for large models.

**Example server config for large models:**

.. code-block:: json

    {
      "id": "cross_site_eval",
      "path": "nvflare.app_common.ccwf.CrossSiteEvalServerController",
      "args": {
        "eval_task_timeout": 1200,
        "configure_task_timeout": 300,
        "progress_timeout": 7200
      }
    }

Client-Side Parameters
----------------------

**CrossSiteEvalClientController:**

- ``get_model_timeout``: Timeout for requesting a model from another client. **Suggested: 600 or higher** for large models.

**Example client config for large models:**

.. code-block:: json

    {
      "tasks": ["cse_*"],
      "executor": {
        "path": "nvflare.app_common.ccwf.CrossSiteEvalClientController",
        "args": {
          "submit_model_task_name": "submit_model",
          "validation_task_name": "validate",
          "persistor_id": "persistor",
          "get_model_timeout": 600
        }
      }
    }

Download and Chunking Behavior
------------------------------

For large model transfers during cross-site evaluation, ensure chunking is configured:

- ``np_download_chunk_size``: Chunk size for NumPy array downloads. **Suggested: 2097152 (2MB)**
- ``tensor_download_chunk_size``: Chunk size for PyTorch tensor downloads. **Suggested: 2097152 (2MB)**

See :ref:`swarm_learning_large_models` for more details on chunking configuration.
