.. _timeouts_programming_guide:

####################################
Timeouts in NVIDIA FLARE (Reference)
####################################

This document provides a comprehensive overview of all timeout configurations in NVIDIA FLARE,
organized by functional categories with relationships, impacts, and usage examples.

.. contents:: Table of Contents
   :local:
   :depth: 2

Network Communication Timeouts
==============================

This section covers all network-related timeouts including the F3/CellNet communication layer,
server configuration, and client communication settings.

F3/CellNet Layer
----------------

The F3 (Flare-Friendly Framework) and CellNet provide the core communication infrastructure.
These timeouts are configured in ``comm_config.json``.

CommConfigurator Settings
^^^^^^^^^^^^^^^^^^^^^^^^^

Low-level communication configuration (comm_config.py):

.. list-table::
   :header-rows: 1
   :widths: 32 10 58

   * - Parameter
     - Default
     - Purpose
   * - heartbeat_interval
     - varies
     - Interval for heartbeat messages
   * - subnet_heartbeat_interval
     - 5.0
     - Interval for subnet heartbeat checks
   * - streaming_read_timeout
     - 300
     - Timeout for reading streamed data
   * - streaming_ack_interval
     - 4MB
     - Bytes between ACK messages during streaming
   * - streaming_ack_wait
     - varies
     - Time to wait for streaming ACK


CoreCell Settings
^^^^^^^^^^^^^^^^^

Core cell communication parameters (core_cell.py):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Purpose
   * - max_timeout
     - 3600
     - Default timeout for send_and_receive (1 hour)
   * - bulk_check_interval
     - 0.5
     - Interval for bulk message checking
   * - bulk_process_interval
     - 0.5
     - Interval for bulk message processing


Cell Request Timeouts
^^^^^^^^^^^^^^^^^^^^^

Cell-level request timeouts (cell.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - timeout
     - 10.0
     - Default timeout for send_request/broadcast_request

**Timeout Phases**: Requests go through three timeout phases:

1. **Sending timeout**: Time to complete message sending
2. **Remote processing timeout**: Time for remote to process request
3. **Receiving timeout**: Time to receive response


Example ``comm_config.json``:

.. code-block:: json

   {
     "heartbeat_interval": 10,
     "subnet_heartbeat_interval": 5,
     "streaming_read_timeout": 300,
     "streaming_ack_interval": 4194304,
     "max_message_size": 1048576
   }


Server Configuration
--------------------

These timeouts are configured in ``fed_server.json`` or server configuration.

FedServer Timeouts
^^^^^^^^^^^^^^^^^^

Server heartbeat and connection management (fed_server.py):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Purpose
   * - heart_beat_timeout
     - 600
     - Time without heartbeat before client considered dead
   * - remove_interval
     - 5.0
     - Interval for checking/removing dead clients
   * - check_interval
     - 0.2
     - Interval for connection checking loop


ServerRunner Timeouts
^^^^^^^^^^^^^^^^^^^^^

Server runner configuration (server_runner.py, server_json_config.py):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Purpose
   * - heartbeat_timeout
     - 60
     - Client heartbeat timeout in seconds
   * - task_request_interval
     - 2
     - Task request interval in seconds


Admin Server Timeouts
^^^^^^^^^^^^^^^^^^^^^

Admin server command timeouts (admin.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - timeout
     - 10.0
     - Admin command timeout
   * - timeout_secs
     - 2.0
     - Timeout for send_requests to clients

**Example** (fed_server.json):

.. code-block:: json

   {
     "heart_beat_timeout": 600,
     "admin_timeout": 10.0
   }


Client Configuration
--------------------

Client heartbeat and retry configuration (client_train.py, base_client_deployer.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - heart_beat_interval
     - 10.0
     - Interval for sending heartbeats to server
   * - retry_timeout
     - 30
     - Timeout for retry operations

**Note**: ``heart_beat_interval`` must be less than the server's ``heart_beat_timeout`` for
proper client status tracking.

Client-to-Server Communication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Low-level client communication timeouts (communicator.py, fed_client_base.py):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Purpose
   * - communication_timeout
     - 300.0
     - General communication timeout
   * - maint_msg_timeout
     - 30.0
     - Maintenance message timeout
   * - engine_create_timeout
     - 30.0
     - Timeout for engine creation
   * - retry_timeout
     - 30.0
     - Retry timeout for operations

Flare Agent
^^^^^^^^^^^

FlareAgent for external process integration (flare_agent.py):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Purpose
   * - heartbeat_timeout
     - 60.0
     - Time without heartbeat before peer is dead
   * - submit_result_timeout
     - 60.0
     - Timeout for submitting task result

**Note**: FlareAgentWithCellPipe uses 30.0s defaults.

IPC Agent
^^^^^^^^^

IPC Agent for inter-process communication (ipc_agent.py):

.. list-table::
   :header-rows: 1
   :widths: 32 10 58

   * - Parameter
     - Default
     - Purpose
   * - submit_result_timeout
     - 30.0
     - Timeout for submitting results
   * - flare_site_connection_timeout
     - 60.0
     - Timeout for CJ disconnection
   * - flare_site_heartbeat_timeout
     - None
     - Timeout for missing CJ heartbeats


gRPC Utility Timeouts
---------------------

gRPC connection establishment (grpc_utils.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - ready_timeout
     - varies
     - Time to wait for gRPC server to be ready


Reliable Message
----------------

Reliable Messages provide guaranteed delivery with retry logic (reliable_message.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - per_msg_timeout
     - varies
     - Timeout for each individual message attempt
   * - tx_timeout
     - varies
     - Timeout for entire transaction including all retries

**Behavior**:

- If ``tx_timeout <= per_msg_timeout``, request is sent only once without retrying
- Messages are retried until ``tx_timeout`` is reached
- Completed requests are tracked for ``2 × tx_timeout`` to handle late duplicates

**Example**:

.. code-block:: python

   from nvflare.apis.utils.reliable_message import ReliableMessage

   ReliableMessage.send_request(
       target="site-1",
       topic="my_topic",
       request=shareable,
       per_msg_timeout=30.0,   # Each attempt times out after 30s
       tx_timeout=300.0,       # Total transaction timeout 5 minutes
       abort_signal=abort_signal,
       fl_ctx=fl_ctx,
   )


Federated Event Timeouts
========================

Fed event runner intervals (fed_event.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - regular_interval
     - 0.01
     - Regular processing interval
   * - grace_period
     - 2.0
     - Grace period before shutdown
   * - queue_empty_period
     - 2.0
     - Period to wait when queue is empty


Simulator Timeouts
==================

Simulator-specific timeouts (simulator_runner.py, simulator_worker.py):

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Parameter
     - Default
     - Purpose
   * - simulator_worker_timeout
     - 60.0
     - Timeout for simulator worker
   * - app_runner_timeout
     - 60.0
     - Timeout for app runner
   * - CELL_CONNECT_CHECK_TIMEOUT
     - 10.0
     - Timeout for cell connection check
   * - FETCH_TASK_RUN_RETRY
     - 3
     - Number of retry attempts for task fetch


Flare API Session Timeouts
==========================

Session management for programmatic API (flare_api.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - timeout (new_session)
     - 10.0
     - Timeout to establish session
   * - poll_interval
     - 2.0
     - Interval for polling job status
   * - set_timeout()
     - varies
     - Session-specific command timeout

**Example**:

.. code-block:: python

   from nvflare.fuel.flare_api.flare_api import new_secure_session

   # Create session with timeout
   sess = new_secure_session(
       username="admin@nvidia.com",
       startup_kit_location="/path/to/startup",
       timeout=30.0,
   )

   # Set command timeout
   sess.set_timeout(60.0)

   # Monitor job with timeout and poll interval
   rc = sess.monitor_job(job_id, timeout=3600, poll_interval=5.0)


Heartbeat Timeouts
==================

Executor Heartbeat
------------------

Heartbeat mechanisms ensure connectivity between components:

.. list-table::
   :header-rows: 1
   :widths: 25 10 35 30

   * - Timeout
     - Default
     - Location
     - Purpose
   * - heartbeat_interval
     - 5.0
     - ``LauncherExecutor`` launcher_executor.py:49
     - Interval for sending heartbeat messages
   * - heartbeat_timeout
     - 60.0
     - ``LauncherExecutor`` launcher_executor.py:50
     - Timeout for waiting for heartbeat from peer
   * - peer_read_timeout
     - 60.0
     - ``LauncherExecutor`` launcher_executor.py:46
     - Time to wait for peer to accept sent message

Client API Heartbeat
^^^^^^^^^^^^^^^^^^^^

The Client API inherits heartbeat configuration from the task exchange settings (config.py:154-159):

.. code-block:: python

   def get_heartbeat_timeout(self):
       return self.config.get(ConfigKey.TASK_EXCHANGE, {}).get(
           ConfigKey.HEARTBEAT_TIMEOUT,
           self.config.get(ConfigKey.METRICS_EXCHANGE, {}).get(ConfigKey.HEARTBEAT_TIMEOUT, 60),
       )

Executor and Launcher Timeouts
==============================

LauncherExecutor Base Class
---------------------------

The ``LauncherExecutor`` class defines core timeout parameters for external process management
(launcher_executor.py:38-58):

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Parameter
     - Default
     - Purpose
   * - launch_timeout
     - None
     - Timeout for launcher's "launch_task" method completion
   * - task_wait_timeout
     - None
     - Timeout for retrieving task results
   * - last_result_transfer_timeout
     - 300.0
     - Timeout for transmitting final result from external process
   * - external_pre_init_timeout
     - 60.0
     - Time to wait for external process before ``flare.init()`` call

ClientAPILauncherExecutor
-------------------------

The Client API executor extends base timeouts with more conservative defaults
(client_api_launcher_executor.py:29-53):

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Parameter
     - Default
     - Purpose
   * - external_pre_init_timeout
     - 300.0
     - Extended timeout for heavy library imports
   * - peer_read_timeout
     - 300.0
     - Timeout for peer message acceptance
   * - heartbeat_timeout
     - 300.0
     - Extended heartbeat timeout for Client API

External Pre-Init Override
^^^^^^^^^^^^^^^^^^^^^^^^^^

Jobs can override the external pre-init timeout via client configuration (constants.py:20-22):

.. code-block:: python

   # Configuration key for overriding external_pre_init_timeout in ClientAPILauncherExecutor
   EXTERNAL_PRE_INIT_TIMEOUT = "EXTERNAL_PRE_INIT_TIMEOUT"


TaskExchanger
-------------

The ``TaskExchanger`` base class manages pipe-based task exchange with external processes
(task_exchanger.py:38-68):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Purpose
   * - read_interval
     - 0.5
     - How often to read from pipe
   * - heartbeat_interval
     - 5.0
     - How often to send heartbeat to peer
   * - heartbeat_timeout
     - 60.0
     - Time to wait for heartbeat from peer (None = disable)
   * - resend_interval
     - 2.0
     - How often to resend a message if failing to send
   * - peer_read_timeout
     - 60.0
     - Time to wait for peer to accept sent message
   * - result_poll_interval
     - 0.5
     - How often to poll for task result


IPCExchanger
------------

The ``IPCExchanger`` manages IPC-based communication with Flare Agents
(ipc_exchanger.py:50-82):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Purpose
   * - send_task_timeout
     - 5.0
     - How long to wait for response when sending task to Agent
   * - resend_task_interval
     - 2.0
     - How often to resend task if failed
   * - agent_connection_timeout
     - 60.0
     - Time allowed to miss heartbeat before considering agent disconnected
   * - agent_heartbeat_timeout
     - None
     - Time allowed to miss heartbeat before stopping (None = disabled)
   * - agent_heartbeat_interval
     - 5.0
     - How often to send heartbeats to the agent
   * - agent_ack_timeout
     - 5.0
     - How long to wait for agent ack (heartbeat and bye messages)


InProcessClientAPIExecutor
--------------------------

The in-process executor for Client API (in_process_client_api_executor.py:50-70):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - result_pull_interval
     - 0.5
     - How often to poll for task result
   * - log_pull_interval
     - None
     - How often to pull logs (None = same as result_pull_interval)


Pipe Handler
------------

Inter-process communication pipe timeouts for Client API (pipe_handler.py):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Purpose
   * - heartbeat_interval
     - 5.0
     - Interval for sending heartbeats
   * - heartbeat_timeout
     - 30.0
     - Max time without heartbeat before peer is dead
   * - default_request_timeout
     - 5.0
     - Default timeout for requests
   * - resend_interval
     - 2.0
     - Interval between message resends

**Important**: ``heartbeat_interval`` must be less than ``heartbeat_timeout``.


P2P Executor
------------

Peer-to-peer sync executor (sync_executor.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - sync_timeout
     - 10
     - Timeout waiting for values from neighbors


Admin Client Timeouts
=====================

Admin client timeouts control session management and command execution:

.. list-table::
   :header-rows: 1
   :widths: 28 10 32 30

   * - Timeout
     - Default
     - Location
     - Purpose
   * - idle_timeout
     - 900.0
     - Admin config
     - Automatic shutdown after idle period
   * - login_timeout
     - 10.0
     - Admin config
     - Max time to attempt login
   * - authenticate_msg_timeout
     - 2.0
     - Admin config
     - Timeout for authentication messages
   * - Command timeout
     - 5.0
     - fl_admin_api.py
     - Default timeout for admin commands

Session-Specific Timeouts
-------------------------

Admin API supports session-specific command timeouts (api_spec.py:305-318):

.. code-block:: python

   def set_timeout(self, value: float):
       """Set a session-specific command timeout. This is the amount of time the server
       will wait for responses after sending commands to FL clients.
       Note that this value is only effective for the current API session."""


Task Communication and Messaging
================================

These timeouts control task assignment and result collection between server and clients.

WfCommServer (Workflow Communication Server)
--------------------------------------------

Server-side workflow communication (wf_comm_server.py):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Purpose
   * - task.timeout
     - varies
     - Overall task timeout
   * - task_assignment_timeout
     - 0
     - Time to wait for client to pick task
   * - task_result_timeout
     - 0
     - Time to wait for client to return result
   * - task_check_period
     - 0.2
     - Interval for checking task status

**Validation Rules**:

- ``task_assignment_timeout`` must be <= ``task.timeout``
- ``task_result_timeout`` must be <= ``task.timeout``


WfCommClient (Workflow Communication Client)
--------------------------------------------

Client-side workflow communication (wf_comm_client.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - max_task_timeout
     - 3600
     - Maximum single task execution time; used as the effective timeout when the controller sets task.timeout = 0 (i.e., "no timeout")


Task Pull/Fetch Timeouts
------------------------

Client-side task fetching from server (client_runner.py, communicator.py, fed_client_base.py):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Purpose
   * - get_task_timeout
     - None
     - Timeout for client to fetch task from server
   * - submit_task_result_timeout
     - None
     - Timeout for client to submit result to server
   * - timeout (pull_task)
     - None
     - Timeout for pull_task communication

**Configuration**: Set via ``ConfigVarName.GET_TASK_TIMEOUT`` and ``ConfigVarName.SUBMIT_TASK_RESULT_TIMEOUT`` in client config.

**Example** (client params in job):

.. code-block:: python

   recipe.add_client_config({
       "get_task_timeout": 300,  # 5 minutes
   })


Task Manager Timeouts
---------------------

Task managers control sequential and relay task distribution (send_manager.py, seq_relay_manager.py, any_relay_manager.py):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Purpose
   * - task_assignment_timeout
     - 0
     - Time window for client to request task
   * - task_result_timeout
     - 0
     - Time to wait for client result before moving to next

**Behavior**:

- For SendOrder.SEQUENTIAL: Clients are assigned in order with sliding time window
- For SendOrder.ANY: First available client gets the task
- Timeout of 0 means no timeout (wait indefinitely)


Workflow and Controller Timeouts
================================

Client-Controlled Workflows (Server-Side)
-----------------------------------------

Server-side controller timeouts for workflow management (common.py:79-92):

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Timeout
     - Default
     - Purpose
   * - configure_task_timeout
     - 300
     - Time for clients to respond to config task
   * - start_task_timeout
     - 10
     - Time for starting client to begin workflow
   * - end_workflow_timeout
     - 2.0
     - Timeout for ending workflow message
   * - progress_timeout
     - 3600.0
     - Max time without workflow progress
   * - max_status_report_interval
     - 90.0
     - Max time for client to miss status report

Client-Controlled Workflows (Client-Side)
-----------------------------------------

Client-side timeouts for task coordination (common.py:87-92):

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Timeout
     - Default
     - Purpose
   * - learn_task_check_interval
     - 1.0
     - Interval for checking new learning tasks
   * - learn_task_ack_timeout
     - 10
     - Timeout for task acknowledgment
   * - learn_task_abort_timeout
     - 5.0
     - Timeout for task abortion
   * - final_result_ack_timeout
     - 10
     - Timeout for final result acknowledgment
   * - get_model_timeout
     - 10
     - Timeout for getting model from peers
   * - max_task_timeout
     - 3600
     - Maximum single task execution time

ScatterAndGather Controller
---------------------------

The SAG controller manages aggregation timing (scatter_and_gather.py:37-67):

.. list-table::
   :header-rows: 1
   :widths: 35 12 53

   * - Parameter
     - Default
     - Purpose
   * - train_timeout
     - 0
     - Time to wait for clients to do local training (0 = no timeout)
   * - wait_time_after_min_received
     - 10
     - Time to wait for additional responses after min_clients
   * - task_check_interval
     - 0.5
     - Interval for checking task completion


ModelController-Based Workflows
-------------------------------

FedAvg, Cyclic, Scaffold, and other ModelController-based workflows (model_controller.py, base_model_controller.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - timeout
     - 0
     - Time to wait for clients to perform task (0 = no timeout)

**Note**: FedAvg, Scaffold, Cyclic all inherit from ModelController and use the same ``timeout`` parameter.


CyclicController
----------------

Cyclic workflow controller (cyclic_ctl.py):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Purpose
   * - task_assignment_timeout
     - 10
     - Timeout for client to request its assigned task


CrossSiteModelEval / CrossSiteEval
----------------------------------

Cross-site model evaluation workflows (cross_site_model_eval.py, cross_site_eval.py):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Purpose
   * - submit_model_timeout
     - 600
     - Timeout for submit_model_task (10 min)
   * - validation_timeout
     - 6000
     - Timeout for validate_model task (100 min)
   * - wait_for_clients_timeout
     - 300
     - Timeout for clients to appear (5 min)
   * - eval_task_timeout (CCWF)
     - 1200+
     - Time for model evaluation by clients
   * - configure_task_timeout (CCWF)
     - 300
     - Timeout for configuration task
   * - progress_timeout (CCWF)
     - 7200+
     - Overall workflow progress timeout

Example configuration:

.. code-block:: python

   from nvflare.app_common.np.recipes import NumpyCrossSiteEvalRecipe

   recipe = NumpyCrossSiteEvalRecipe(
       submit_model_timeout=600,
       validation_timeout=6000,
   )


GlobalModelEval
---------------

Global model evaluation controller (global_model_eval.py):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Purpose
   * - validation_timeout
     - 6000
     - Timeout for validate_model task
   * - wait_for_clients_timeout
     - 300
     - Timeout for clients to appear


BroadcastAndProcess / InitializeGlobalWeights
---------------------------------------------

Broadcast workflows (broadcast_and_process.py, initialize_global_weights.py):

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Parameter
     - Default
     - Purpose
   * - timeout / task_timeout
     - 0
     - Task timeout (0 = no timeout)
   * - wait_time_after_min_received
     - 0-10
     - Wait time after min responses received


StatisticsController / HierarchicalStatisticsController
--------------------------------------------------------

Statistics workflow controllers (statistics_controller.py, hierarchical_statistics_controller.py):

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Parameter
     - Default
     - Purpose
   * - result_wait_timeout
     - 10
     - Seconds to wait for results per statistic
   * - wait_time_after_min_received
     - 1
     - Seconds to wait after min clients received

**Note**: ``result_wait_timeout`` is reset for each statistic, not an overall timeout.


SplitNNController
-----------------

Split learning controller (splitnn_workflow.py:47-79):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - task_timeout
     - 10
     - Timeout for client to request its assigned task
   * - TIMEOUT (class constant)
     - 60.0
     - Timeout for auxiliary message requests


TIE Controller (Third-party Integration)
----------------------------------------

Base controller for third-party integration (tie/controller.py, tie/defs.py):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Purpose
   * - configure_task_timeout
     - 10
     - Time to wait for clients to complete config task
   * - start_task_timeout
     - 10
     - Time to wait for clients to complete start task
   * - job_status_check_interval
     - 2.0
     - How often to check client job statuses
   * - max_client_op_interval
     - 90.0
     - Max time allowed between app ops from a client
   * - progress_timeout
     - 3600.0
     - Max time allowed with no workflow progress

**Note**: TIE is used by XGBoost, Flower, and other third-party framework integrations.


Flower Integration Timeouts
---------------------------

Flower-specific controller and executor timeouts (flower/controller.py, flower/executor.py):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Purpose
   * - superlink_ready_timeout
     - 10.0
     - Time to wait for Flower superlink to become ready
   * - superlink_min_query_interval
     - 10.0
     - Minimal interval for querying superlink status
   * - monitor_interval
     - 0.5
     - How often to check Flower run status
   * - per_msg_timeout
     - 10.0
     - Per-message timeout for ReliableMessage
   * - tx_timeout
     - 100.0
     - Transaction timeout for ReliableMessage
   * - client_shutdown_timeout
     - 5.0
     - Max time for graceful client shutdown


Private Set Intersection (PSI)
------------------------------

PSI workflows do not have explicit timeout parameters at the PSI controller level. 
PSI inherits general workflow timeouts from the underlying task system.

For PSI operations, timeouts are controlled at lower levels:

- **Task-level timeouts**: Use controller's general ``timeout`` parameter
- **Communication timeouts**: Inherited from system ``heartbeat_timeout`` and ``peer_read_timeout``

**Note**: For large-scale PSI operations, ensure adequate system-level timeouts in 
``application.conf`` to handle the iterative Diffie-Hellman protocol exchanges.


Aggregator Timeouts
-------------------

LazyAggregator
^^^^^^^^^^^^^^

Lazy aggregator for async aggregation (lazy.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - accept_timeout
     - 600.0
     - Max time to wait for accept to finish


Job Scheduler Timeouts
----------------------

The ``DefaultJobScheduler`` controls job scheduling frequency (job.rst:255-270):

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Parameter
     - Default
     - Purpose
   * - min_schedule_interval
     - 10.0
     - Minimum interval between schedule attempts
   * - max_schedule_interval
     - 600.0
     - Maximum interval between schedule attempts
   * - max_schedule_count
     - 10
     - Maximum times to try scheduling a job

**Scheduling Strategy**: The scheduler uses adaptive frequency - doubling interval after each
failure up to the maximum.

Recipe Timeouts
===============

Standard Recipe Timeouts
------------------------

All standard recipes support these timeout parameters (fedavg.py, cyclic.py):

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Parameter
     - Default
     - Purpose
   * - shutdown_timeout
     - 0.0
     - Wait time before shutdown for cleanup
   * - task_assignment_timeout
     - 10
     - Timeout for cyclic task assignment (CyclicRecipe only)

Evaluation Recipe Timeouts
--------------------------

Evaluation recipes have specific timeout requirements (fedeval.py, cross_site_eval.py):

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Parameter
     - Default
     - Purpose
   * - validation_timeout
     - 6000
     - Time allowed for model validation
   * - submit_model_timeout
     - 600
     - Time for clients to submit models for evaluation


Large Model and Streaming Timeouts
==================================

File Streaming Timeouts
-----------------------

File streaming for large files (file_streamer.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - chunk_timeout
     - 5.0
     - Timeout for each chunk sent to targets
   * - chunk_size
     - 1M bytes
     - Size of each chunk streamed

**Example**:

.. code-block:: python

   from nvflare.app_common.streamers.file_streamer import FileStreamer

   FileStreamer.stream_file(
       targets=["site-1", "site-2"],
       file_name="/path/to/large_file.bin",
       fl_ctx=fl_ctx,
       chunk_size=1024 * 1024,  # 1MB chunks
       chunk_timeout=10.0,      # 10 seconds per chunk
   )


Container Streaming Timeouts
----------------------------

Container/object streaming (container_streamer.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - entry_timeout
     - 60.0
     - Timeout for each entry sent to targets

**Example**:

.. code-block:: python

   from nvflare.app_common.streamers.container_streamer import ContainerStreamer

   ContainerStreamer.stream_container(
       targets=["site-1"],
       container=my_large_container,
       fl_ctx=fl_ctx,
       entry_timeout=120.0,  # 2 minutes per entry
   )


Object Retrieval Timeouts
-------------------------

Retrieving files/containers from remote sites (file_retriever.py, container_retriever.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - timeout
     - varies
     - Max seconds to wait for data retrieval
   * - chunk_timeout
     - varies
     - Timeout per chunk during file retrieval


Byte Streaming Timeouts
-----------------------

Byte streaming timeouts and intervals (byte_receiver.py, byte_streamer.py):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Purpose
   * - streaming_read_timeout
     - 300
     - Timeout for reading streamed data
   * - ack_interval
     - 4MB
     - Bytes between acknowledgment messages
   * - ack_wait
     - varies
     - Time to wait for ACK before timing out

**Note**: ACK timeout triggers ``StreamError`` and stops the stream.


Download Transaction Timeouts
-----------------------------

Object download transaction timeouts (download_service.py, obj_downloader.py):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Purpose
   * - timeout
     - varies
     - Transaction timeout (time since last activity)
   * - per_request_timeout
     - varies
     - Timeout for each request to object owner

**Note**: Transaction times out if no activity from any receiver for the specified duration.


Tensor Streaming Timeouts
-------------------------

Tensor streaming provides efficient transfer of large model weights. These timeouts control
the streaming behavior (tensor_stream/server.py, client.py):

.. list-table::
   :header-rows: 1
   :widths: 38 12 50

   * - Parameter
     - Default
     - Purpose
   * - tensor_send_timeout
     - 30.0
     - Timeout for each tensor entry transfer operation
   * - wait_send_task_data_all_clients_timeout
     - 300.0
     - Timeout for sending tensors to all clients
   * - wait_for_tensors timeout
     - 5.0
     - Time to wait for tensors to be received

**Server-side configuration** (TensorServerStreamer):

.. code-block:: python

   from nvflare.app_opt.tensor_stream.server import TensorServerStreamer

   streamer = TensorServerStreamer(
       format="pytorch",
       tensor_send_timeout=60.0,  # Per-tensor timeout
       wait_send_task_data_all_clients_timeout=600.0,  # All clients timeout
   )

**Client-side configuration** (TensorClientStreamer):

.. code-block:: python

   from nvflare.app_opt.tensor_stream.client import TensorClientStreamer

   streamer = TensorClientStreamer(
       format="pytorch",
       tensor_send_timeout=60.0,  # Per-tensor timeout
   )

.. warning::

   **Critical Timeout Relationship for Tensor Streaming**
   
   When using tensor streaming, you **must** ensure that ``get_task_timeout`` is set and is 
   greater than or equal to ``wait_send_task_data_all_clients_timeout``. If ``get_task_timeout`` 
   is not set, it defaults to the communicator's timeout, which may be shorter than the tensor 
   streaming timeout.
   
   **Problem**: If streaming timeout > communicator timeout and no ``get_task_timeout`` is set, 
   some clients may receive weights while others are still waiting. The server may not send the 
   task in time, causing a timeout that restarts the tensor streaming process. This can result 
   in clients receiving empty tensors and job failure.
   
   **Solution**: Always set ``get_task_timeout`` when using tensor streaming:
   
   .. code-block:: python
   
      # Ensure get_task_timeout >= wait_send_task_data_all_clients_timeout
      recipe.add_client_config({
          "get_task_timeout": 600,  # Must be >= streaming timeout
      })


Streaming Download Timeouts
---------------------------

Framework-level settings for large payload transfers (fl_constant.py:553, comm_config.py:41):

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Parameter
     - Default
     - Purpose
   * - streaming_per_request_timeout
     - 600
     - Per-request timeout for streaming chunks
   * - streaming_read_timeout
     - 300
     - Timeout for reading streaming data
   * - np_download_chunk_size
     - 2097152
     - Chunk size for NumPy array downloads (bytes)
   * - tensor_download_chunk_size
     - 2097152
     - Chunk size for PyTorch tensor downloads (bytes)

Swarm Learning Large Model Setup
--------------------------------

Recommended timeouts for large models in Swarm Learning:

.. code-block:: python

   # Server configuration
   recipe.add_server_config({
       "start_task_timeout": 300,
       "progress_timeout": 7200,
       "np_download_chunk_size": 2097152,
       "streaming_per_request_timeout": 600
   })

   # Client configuration
   {
       "learn_task_timeout": 3600,
       "learn_task_ack_timeout": 300,
       "final_result_ack_timeout": 300
   }


XGBoost-Specific Timeouts
=========================

XGBoost Histogram-Based Controller
----------------------------------

XGBoost histogram-based controller timeouts (histogram_based_v2/controller.py):

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Parameter
     - Default
     - Purpose
   * - configure_task_timeout
     - 300
     - Timeout for configuration task
   * - start_task_timeout
     - 10
     - Timeout for start task
   * - progress_timeout
     - 3600.0
     - Overall workflow progress timeout

**Note**: XGBoost uses Reliable Messages for secure training. See the `Reliable Message`_ section
for ``per_msg_timeout`` and ``tx_timeout`` configuration.

XGBoost gRPC Client
-------------------

gRPC client for XGBoost communication (grpc_client.py, grpc_server_adaptor.py):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Purpose
   * - ready_timeout
     - 10
     - Timeout for gRPC server to be ready
   * - xgb_server_ready_timeout
     - varies
     - Timeout for XGBoost server readiness
   * - aggr_timeout
     - 10.0
     - Aggregation timeout for mock servicer

Example configuration for large datasets:

.. code-block:: python

   "per_msg_timeout": 300.0,
   "tx_timeout": 900.0,


Confidential Computing Timeouts
===============================

SNP Authorizer Timeouts
-----------------------

AMD SEV-SNP attestation timeouts (snp_authorizer.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - cmd_timeout
     - 60
     - SNPGuest command execution timeout
   * - retry_interval
     - 10
     - Wait time between retry attempts
   * - max_retries
     - 5
     - Maximum retry attempts

CC Manager Timeouts
-------------------

Cross-site CC verification timeouts (cc_manager.py):

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Parameter
     - Default
     - Purpose
   * - get_site_request_timeout
     - 10.0
     - Timeout for get site request
   * - get_token_request_timeout
     - 10.0
     - Timeout for get token request
   * - verify_frequency
     - 600
     - CC token verification interval (seconds)
   * - cross_validation_interval
     - varies
     - Interval between cross-site validation cycles

**Note**: Other CC authorizers (ACI, TDX, GPU, Azure CVM) do not have explicit timeout parameters
and rely on system defaults.


Job Launcher Timeouts
=====================

Kubernetes Launcher
-------------------

K8s job launcher timeouts (k8s_launcher.py):

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Parameter
     - Default
     - Purpose
   * - timeout
     - None
     - Timeout for pod to enter RUNNING/TERMINATED state

Docker Launcher
---------------

Docker container launcher timeouts (docker_launcher.py):

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Parameter
     - Default
     - Purpose
   * - timeout
     - None
     - Timeout for container to enter target state


Edge Device Timeouts
====================

This section covers all edge device, mobile client, and hierarchical FL timeouts.

Edge Device General
-------------------

Edge devices have specific timeout requirements:

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - update_timeout
     - 5
     - Timeout for model updates from devices
   * - device_wait_timeout
     - None
     - Time to wait for sufficient devices to join
   * - job_timeout
     - 60.0
     - Overall timeout for edge job execution

Example:

.. code-block:: python

   from nvflare.edge.tools.edge_fed_buff_recipe import EdgeFedBuffRecipe

   recipe = EdgeFedBuffRecipe(
       model=MyModel(),
       update_timeout=10,
       job_timeout=120.0,
   )


Hierarchical FL
---------------

Hierarchical FL enables multi-tier federation with edge devices organized in a tree structure.

ScatterAndGatherForEdge (SAGE) Controller
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Server-side controller for hierarchical edge FL (edge/controllers/sage.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - assess_interval
     - 0.5
     - Interval for invoking the assessor during task execution
   * - update_interval
     - 1.0
     - Interval for children to send updates
   * - task_check_period
     - 0.5
     - Interval for checking status of tasks

HierarchicalUpdateGatherer (HUG) Executor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Executor for hierarchical update gathering (edge/executors/hug.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - update_timeout
     - required
     - Timeout for update messages sent to parent

EdgeTaskExecutor (ETE)
^^^^^^^^^^^^^^^^^^^^^^

Edge task executor for leaf nodes (edge/executors/ete.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - update_timeout
     - required
     - Timeout for update messages sent to parent

**Example**:

.. code-block:: python

   from nvflare.edge.controllers.sage import ScatterAndGatherForEdge
   from nvflare.edge.executors.hug import HierarchicalUpdateGatherer

   # Server-side controller
   sage = ScatterAndGatherForEdge(
       num_rounds=5,
       assess_interval=0.5,
       update_interval=1.0,
       task_check_period=0.5,
   )

   # Client-side executor
   hug = HierarchicalUpdateGatherer(
       learner_id="learner",
       updater_id="updater",
       update_timeout=30.0,
   )


Mobile Client
-------------

Android SDK includes job operation timeout (mobile_android.rst:43-58):

.. code-block:: kotlin

   AndroidFlareRunner(
       // ... other parameters
       jobTimeout: Float,  // Timeout in seconds for job operations
   )


SubprocessLauncher Timeouts
===========================

Subprocess launcher timeout (subprocess_launcher.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - shutdown_timeout
     - 0.0
     - Time to wait before forcefully stopping subprocess


Experiment Tracking Timeouts
============================

WandB Receiver
--------------

Weights & Biases integration timeouts (wandb_receiver.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - process_timeout
     - 10.0
     - Timeout for joining WandB processes at shutdown
   * - login timeout
     - 1.0
     - Internal timeout for WandB login verification


MLflow Receiver
---------------

MLflow integration timing (mlflow_receiver.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - buffer_flush_time
     - 1
     - Seconds between deliveries to MLflow tracking server

**Note**: Reducing ``buffer_flush_time`` increases traffic to MLflow server and may cause latency.


TensorBoard Receiver
--------------------

TensorBoard receiver (tb_receiver.py) does not have explicit timeout parameters.
Events are written directly to disk without buffering.


Metrics Relay and Sender
------------------------

Metrics exchange timeouts for experiment tracking (metric_relay.py, metrics_sender.py):

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Default
     - Purpose
   * - heartbeat_timeout
     - 30.0-60.0
     - Timeout for peer heartbeat (MetricRelay: 60s, MetricsSender: 30s)
   * - heartbeat_interval
     - 5.0
     - Interval between heartbeats
   * - read_interval
     - 0.1
     - Interval for reading from pipe

**Example**:

.. code-block:: python

   from nvflare.app_common.widgets.metric_relay import MetricRelay

   metric_relay = MetricRelay(
       heartbeat_interval=5.0,
       heartbeat_timeout=60.0,
       read_interval=0.1,
   )


Timeout Relationships and Dependencies
======================================

Hierarchical Relationships
--------------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                    SYSTEM-LEVEL TIMEOUTS                        │
   ├─────────────────────────────────────────────────────────────────┤
   │  Server Configuration (fed_server.json)                        │
   │  ├── heart_beat_timeout (600s) - Client liveness detection     │
   │  ├── admin_timeout (10s) - Admin command processing            │
   │  └── task_request_interval (2s) - Task polling rate            │
   │                                                                 │
   │  Client Configuration                                           │
   │  ├── heart_beat_interval (10s) - Keep-alive to server          │
   │  ├── retry_timeout (30s) - Operation retry                      │
   │  └── communication_timeout (300s) - Network operations          │
   └─────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────┐
   │                    F3/CELLNET LAYER                             │
   ├─────────────────────────────────────────────────────────────────┤
   │  CommConfigurator (comm_config.json)                            │
   │  ├── heartbeat_interval < heartbeat_timeout (REQUIRED)          │
   │  ├── subnet_heartbeat_interval (5s)                             │
   │  ├── streaming_read_timeout (300s)                              │
   │  └── max_timeout (3600s) - CoreCell default                     │
   │                                                                 │
   │  Cell Requests                                                  │
   │  └── timeout (10s) → Sending → Processing → Receiving           │
   └─────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────┐
   │                    TASK COMMUNICATION                           │
   ├─────────────────────────────────────────────────────────────────┤
   │  Task Lifecycle                                                 │
   │  ├── task_assignment_timeout ≤ task.timeout (REQUIRED)          │
   │  ├── task_result_timeout ≤ task.timeout (REQUIRED)              │
   │  ├── get_task_timeout - Client fetching task                    │
   │  └── submit_task_result_timeout - Client submitting result      │
   │                                                                 │
   │  max_task_timeout (3600s) - Applied when task.timeout = 0       │
   └─────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────┐
   │                    WORKFLOW LAYER                               │
   ├─────────────────────────────────────────────────────────────────┤
   │  ModelController-Based (FedAvg, Cyclic, Scaffold, etc.)         │
   │  └── timeout (0 = no timeout) - Per-task timeout                │
   │                                                                 │
   │  ScatterAndGather / ScatterAndGatherScaffold                    │
   │  ├── train_timeout (0 = no timeout)                             │
   │  └── wait_time_after_min_received (10s)                         │
   │                                                                 │
   │  CyclicController                                               │
   │  └── task_assignment_timeout (10s)                              │
   │                                                                 │
   │  CrossSiteModelEval / CrossSiteEval                             │
   │  ├── submit_model_timeout (600s)                                │
   │  ├── validation_timeout (6000s)                                 │
   │  └── wait_for_clients_timeout (300s)                            │
   │                                                                 │
   │  GlobalModelEval                                                │
   │  ├── validation_timeout (6000s)                                 │
   │  └── wait_for_clients_timeout (300s)                            │
   │                                                                 │
   │  BroadcastAndProcess / InitializeGlobalWeights                  │
   │  ├── timeout / task_timeout (0 = no timeout)                    │
   │  └── wait_time_after_min_received (0-10s)                       │
   │                                                                 │
   │  StatisticsController / HierarchicalStatisticsController        │
   │  └── result_wait_timeout (10s) - Per-statistic timeout          │
   │                                                                 │
   │  SplitNNController                                              │
   │  └── task_timeout (10s)                                         │
   │                                                                 │
   │  TIE Controller (XGBoost, Flower, etc.)                         │
   │  ├── configure_task_timeout (10s)                               │
   │  ├── start_task_timeout (10s)                                   │
   │  ├── job_status_check_interval (2s)                             │
   │  ├── max_client_op_interval (90s)                               │
   │  └── progress_timeout (3600s)                                   │
   │                                                                 │
   │  Flower-Specific                                                │
   │  ├── superlink_ready_timeout (10s)                              │
   │  ├── per_msg_timeout (10s)                                      │
   │  ├── tx_timeout (100s)                                          │
   │  └── client_shutdown_timeout (5s)                               │
   │                                                                 │
   │  CCWF Server-Side                                               │
   │  ├── configure_task_timeout (300s)                              │
   │  ├── start_task_timeout (10s)                                   │
   │  └── progress_timeout (3600s) - Overall workflow                │
   │                                                                 │
   │  CCWF Client-Side (Swarm Learning)                              │
   │  ├── learn_task_ack_timeout (10s)                               │
   │  ├── learn_task_abort_timeout (5s)                              │
   │  └── final_result_ack_timeout (10s)                             │
   └─────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────┐
   │                    EXECUTOR LAYER                               │
   ├─────────────────────────────────────────────────────────────────┤
   │  LauncherExecutor / ClientAPILauncherExecutor                   │
   │  ├── launch_timeout                                             │
   │  ├── external_pre_init_timeout (60-300s)                        │
   │  ├── task_wait_timeout                                          │
   │  ├── last_result_transfer_timeout (300s)                        │
   │  └── heartbeat_timeout (60-300s)                                │
   │                                                                 │
   │  TaskExchanger (Pipe Handler)                                   │
   │  ├── heartbeat_interval < heartbeat_timeout (REQUIRED)          │
   │  ├── read_interval (0.5s)                                       │
   │  ├── resend_interval (2s)                                       │
   │  ├── peer_read_timeout (60s)                                    │
   │  └── result_poll_interval (0.5s)                                │
   │                                                                 │
   │  IPCExchanger (Agent-based)                                     │
   │  ├── send_task_timeout (5s)                                     │
   │  ├── resend_task_interval (2s)                                  │
   │  ├── agent_connection_timeout (60s)                             │
   │  ├── agent_heartbeat_timeout (None)                             │
   │  └── agent_ack_timeout (5s)                                     │
   │                                                                 │
   │  InProcessClientAPIExecutor                                     │
   │  ├── result_pull_interval (0.5s)                                │
   │  └── log_pull_interval (None)                                   │
   └─────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────┐
   │                    STREAMING LAYER                              │
   ├─────────────────────────────────────────────────────────────────┤
   │  Reliable Message                                               │
   │  └── per_msg_timeout ≤ tx_timeout (for retries to work)         │
   │                                                                 │
   │  File/Container Streaming                                       │
   │  ├── chunk_timeout (5s per chunk)                               │
   │  └── entry_timeout (60s per entry)                              │
   │                                                                 │
   │  Tensor Streaming (CRITICAL RELATIONSHIP)                       │
   │  ├── tensor_send_timeout (30s)                                  │
   │  ├── wait_send_task_data_all_clients_timeout (300s)             │
   │  └── get_task_timeout >= wait_send_task_data_all_clients_timeout│
   │      (REQUIRED to prevent task fetch timeout during streaming)  │
   └─────────────────────────────────────────────────────────────────┘


Impact Analysis
---------------

**Too Short Timeouts:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Timeout Category
     - Impact of Too Short Value
   * - heart_beat_timeout
     - Clients incorrectly marked dead, frequent reconnections
   * - task.timeout / train_timeout
     - Training interrupted before completion, lost work
   * - external_pre_init_timeout
     - Large model loading fails, external processes killed
   * - streaming_read_timeout
     - Large file transfers fail mid-stream
   * - per_msg_timeout
     - Reliable messages fail on slow networks
   * - get_task_timeout
     - Clients fail to receive tasks, job stalls
   * - admin_timeout
     - Admin commands fail, poor CLI experience
   * - task_assignment_timeout (Cyclic)
     - Client fails to fetch task in time, job aborts
   * - submit_model_timeout (CrossSiteEval)
     - Model submission fails, evaluation incomplete
   * - validation_timeout (CrossSiteEval)
     - Validation tasks fail prematurely
   * - result_wait_timeout (Statistics)
     - Statistics collection aborted before all clients respond
   * - agent_connection_timeout (IPC)
     - External agent incorrectly marked disconnected
   * - send_task_timeout (IPC)
     - Task delivery to agent fails, triggers resends
   * - superlink_ready_timeout (Flower)
     - Flower integration fails to initialize
   * - configure_task_timeout (TIE)
     - Third-party framework configuration fails
   * - max_client_op_interval (TIE)
     - Healthy clients marked as stuck

**Too Long Timeouts:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Timeout Category
     - Impact of Too Long Value
   * - heart_beat_timeout
     - Dead clients not detected, resources wasted
   * - task_assignment_timeout
     - Slow failover to backup clients
   * - progress_timeout
     - Hung workflows not detected for hours
   * - retry_timeout
     - Long delays before retry attempts
   * - shutdown_timeout
     - Slow job termination, resource cleanup delayed
   * - wait_for_clients_timeout (CrossSiteEval)
     - Long wait for clients that won't join
   * - agent_heartbeat_timeout (IPC)
     - Hung agents not detected, job stalls
   * - resend_task_interval (IPC/TaskExchanger)
     - Slow recovery from transient failures
   * - result_poll_interval (Executor)
     - Delayed result detection, slower job completion
   * - job_status_check_interval (TIE)
     - Delayed detection of job completion or failure
   * - tx_timeout (ReliableMessage)
     - Long waits for failed transactions


Recommended Settings by Use Case
================================

Development Environment
-----------------------

Fast iteration with quick feedback:

.. code-block:: python

   # Server (fed_server.json)
   heart_beat_timeout = 60        # Quick dead client detection
   admin_timeout = 5.0            # Fast admin commands

   # Client parameters
   heartbeat_timeout = 30.0
   task_wait_timeout = 60.0
   external_pre_init_timeout = 60.0

   # Flare API
   login_timeout = 5.0
   poll_interval = 1.0


Production - Standard Training
------------------------------

Balanced settings for typical federated learning:

.. code-block:: python

   # Server (fed_server.json)
   heart_beat_timeout = 600       # 10 min before client considered dead
   admin_timeout = 10.0
   task_request_interval = 2.0

   # comm_config.json
   heartbeat_interval = 10
   subnet_heartbeat_interval = 5
   streaming_read_timeout = 300

   # Executor
   external_pre_init_timeout = 300.0
   heartbeat_timeout = 300.0
   last_result_transfer_timeout = 300.0


Production - Large Models (100M+ parameters)
--------------------------------------------

Extended timeouts for large model training:

.. code-block:: python

   # Server
   heart_beat_timeout = 1200      # 20 min for large model operations

   # Executor/Launcher
   external_pre_init_timeout = 600.0   # 10 min for model loading
   task_wait_timeout = 3600.0          # 1 hour for training

   # Streaming
   streaming_per_request_timeout = 900  # 15 min per chunk
   tensor_send_timeout = 120.0

   # CCWF
   progress_timeout = 14400       # 4 hours
   learn_task_timeout = 7200      # 2 hours


LLM/Foundation Model Training
-----------------------------

For billion-parameter models (examples/advanced/llm_hf):

.. code-block:: python

   # Recipe configuration
   recipe = FedAvgRecipe(
       name="llm_training",
       model=None,  # Use dict config for large models
       shutdown_timeout=120.0,
   )

   # Client parameters - CRITICAL for LLM
   recipe.add_client_config({
       "get_task_timeout": 600,            # 10 min to receive task
       "submit_task_result_timeout": 600,  # 10 min to submit results
       "external_pre_init_timeout": 900,   # 15 min for model init
   })


Unreliable/High-Latency Networks
--------------------------------

Conservative settings for challenging network conditions:

.. code-block:: python

   # More frequent heartbeats with longer tolerance
   heartbeat_interval = 15.0      # Less frequent to reduce traffic
   heartbeat_timeout = 180.0      # 3 min tolerance

   # Extended communication timeouts
   communication_timeout = 600.0
   peer_read_timeout = 180.0
   maint_msg_timeout = 60.0

   # Reliable message settings
   per_msg_timeout = 60.0
   tx_timeout = 600.0             # Long transaction timeout for retries

   # Streaming with larger windows
   streaming_read_timeout = 600
   ack_wait = 30


Edge/Hierarchical FL
--------------------

Settings for edge device deployments:

.. code-block:: python

   # Edge device timeouts
   update_timeout = 30
   job_timeout = 300.0
   device_wait_timeout = 120.0

   # Hierarchical FL
   assess_interval = 1.0
   update_interval = 2.0


XGBoost Secure Training
-----------------------

Settings for histogram-based XGBoost:

.. code-block:: python

   # Controller
   configure_task_timeout = 300
   start_task_timeout = 30
   progress_timeout = 7200

   # Reliable messaging for large histograms
   per_msg_timeout = 120.0
   tx_timeout = 600.0
   xgb_server_ready_timeout = 30


Cross-Site Model Evaluation
---------------------------

Settings for model evaluation across sites:

.. code-block:: python

   from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval

   controller = CrossSiteModelEval(
       submit_model_timeout=900,        # 15 min for large model submission
       validation_timeout=7200,         # 2 hours for thorough validation
       wait_for_clients_timeout=600,    # 10 min for clients to connect
   )


Federated Statistics
--------------------

Settings for statistics computation:

.. code-block:: python

   from nvflare.app_common.workflows.statistics_controller import StatisticsController

   controller = StatisticsController(
       result_wait_timeout=60,          # 1 min per statistic
       min_clients=2,
   )


Split Learning
--------------

Settings for split neural network training:

.. code-block:: python

   from nvflare.app_common.workflows.splitnn_workflow import SplitNNController

   controller = SplitNNController(
       task_timeout=30,                 # 30 sec for task assignment
       num_rounds=10,
   )


Flower Integration
------------------

Settings for Flower framework integration:

.. code-block:: python

   from nvflare.app_opt.flower.flower_job import FlowerJob

   job = FlowerJob(
       superlink_ready_timeout=30.0,    # 30 sec for Flower server
       configure_task_timeout=60,
       start_task_timeout=30,
       progress_timeout=7200,           # 2 hours for training
       per_msg_timeout=30.0,
       tx_timeout=300.0,
       client_shutdown_timeout=10.0,
   )


Configuration File Locations
============================

This section describes where timeout configuration files are located and which timeouts 
each file controls. Configuration is divided into **system-level** (startup kit) and 
**job-level** (application) settings.

System-Level Configuration (Startup Kit)
----------------------------------------

System-level timeouts are configured in the startup kit and apply to all jobs.
These files are located in the ``local/`` directory of each participant.

**Startup Kit Structure:**

.. code-block:: text

   startup_kit/
   ├── server/
   │   └── local/
   │       ├── fed_server.json          # Server heartbeat, admin timeouts
   │       ├── comm_config.json         # F3/CellNet communication layer
   │       └── resources.json           # Resource configuration
   │
   ├── site-1/ (client)
   │   └── local/
   │       ├── fed_client.json          # Client heartbeat, retry timeouts
   │       ├── comm_config.json         # F3/CellNet communication layer
   │       └── resources.json           # Resource configuration
   │
   └── admin/
       └── local/
           └── admin.json               # Admin session timeouts

**Deployed System Paths:**

After deployment, these files are located at:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Component
     - Startup Kit Path
     - Deployed Path
   * - Server
     - ``startup_kit/server/local/``
     - ``/opt/nvflare/workspace/server/local/`` or ``~/nvflare/workspace/server/local/``
   * - Client (Site)
     - ``startup_kit/site-*/local/``
     - ``/opt/nvflare/workspace/site-*/local/`` or ``~/nvflare/workspace/site-*/local/``
   * - Admin
     - ``startup_kit/admin/local/``
     - ``/opt/nvflare/workspace/admin/local/`` or ``~/nvflare/workspace/admin/local/``

**System-Level Configuration Files:**

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - File
     - Location
     - Timeouts Controlled
   * - fed_server.json
     - server/local/
     - ``heart_beat_timeout``, ``admin_timeout``, ``task_request_interval``, ``heartbeat_timeout``
   * - fed_client.json
     - site-*/local/
     - ``heart_beat_interval``, ``retry_timeout``, ``communication_timeout``
   * - comm_config.json
     - server/local/, site-*/local/
     - ``heartbeat_interval``, ``subnet_heartbeat_interval``, ``streaming_read_timeout``, ``streaming_ack_interval``, ``max_message_size``
   * - resources.json
     - server/local/, site-*/local/
     - Resource allocation and limits
   * - admin.json
     - admin/local/
     - ``idle_timeout``, ``login_timeout``, ``command_timeout``

**Note**: Changes to system-level files require restarting the affected FLARE components.


Job-Level Configuration
-----------------------

Job-level timeouts are configured per job and override defaults for that specific job.
These files are located in the job's ``app/config/`` directory.

**Job Configuration Files:**

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - File
     - Location
     - Timeouts Controlled
   * - application.conf
     - app/config/
     - Task timeouts, streaming timeouts, runner sync timeouts
   * - config_fed_client.json
     - app/config/
     - Executor timeouts, Client API task exchange, pipe handler settings
   * - config_fed_server.json
     - app/config/
     - Controller timeouts, workflow component configurations

**Ways to Configure Job-Level Timeouts:**

1. **Recipe API** - Using ``recipe.add_client_config()`` to pass client parameters:

   .. code-block:: python

      # Apply to all clients
      recipe.add_client_config({
          "get_task_timeout": 300,
          "submit_task_result_timeout": 300,
      })

      # Apply to specific clients
      recipe.add_client_config({
          "get_task_timeout": 600,
      }, clients=["site-1", "site-2"])

2. **Job config files** - In ``app/config/`` directory:

   - ``config_fed_client.json`` - Client-side executor and task exchange settings
   - ``config_fed_server.json`` - Server-side controller and workflow settings

Configuration Examples
======================

fed_server.json (Server Configuration)
--------------------------------------

.. code-block:: json

   {
     "heart_beat_timeout": 600,
     "admin_timeout": 10.0,
     "servers": [
       {
         "heart_beat_timeout": 600
       }
     ]
   }


comm_config.json (F3/CellNet Layer)
-----------------------------------

.. code-block:: json

   {
     "heartbeat_interval": 10,
     "subnet_heartbeat_interval": 5,
     "streaming_read_timeout": 300,
     "streaming_ack_interval": 4194304,
     "streaming_chunk_size": 1048576,
     "max_message_size": 1048576
   }


Client API Configuration (config_fed_client.json)
-------------------------------------------------

.. code-block:: json

   {
     "TASK_EXCHANGE": {
       "heartbeat_timeout": 60.0,
       "heartbeat_interval": 5.0,
       "resend_interval": 2.0,
       "pipe": {
         "ARG": {
           "root_url": "tcp://localhost:8002"
         }
       }
     }
   }


application.conf Settings
-------------------------

.. code-block:: hocon

   # Task communication timeouts
   get_task_timeout = 60.0
   submit_task_result_timeout = 120.0
   task_check_timeout = 5.0

   # Cell/messaging timeouts
   cell_wait_timeout = 5.0

   # Streaming timeouts
   streaming_per_request_timeout = 600.0
   np_download_chunk_size = 4194304
   tensor_download_chunk_size = 4194304

   # Runner sync timeouts
   runner_sync_timeout = 10.0
   max_runner_sync_timeout = 60.0

   # Shutdown
   end_run_readiness_timeout = 10.0


Admin Client Session (Python API)
---------------------------------

.. code-block:: python

   from nvflare.fuel.flare_api.flare_api import new_secure_session

   # Create session with connection timeout
   sess = new_secure_session(
       username="admin@nvidia.com",
       startup_kit_location="/path/to/startup",
       timeout=30.0,
   )

   # Set session-specific command timeout
   sess.set_timeout(60.0)  # 60 seconds for commands

   # Monitor job with timeout
   rc = sess.monitor_job(
       job_id,
       timeout=3600,       # 1 hour max
       poll_interval=5.0,  # Check every 5 seconds
   )

   # Reset to server default
   sess.unset_timeout()


Recipe with Extended Timeouts
-----------------------------

.. code-block:: python

   from nvflare.app_opt.pt.recipes import FedAvgRecipe

   recipe = FedAvgRecipe(
       name="large_model_training",
       model={"class_path": "model.LargeModel", "args": {}},
       min_clients=8,
       num_rounds=100,
       shutdown_timeout=120.0,
       train_script="client.py",
   )

   # Client timeout parameters
   recipe.add_client_config({
       "get_task_timeout": 300,
       "submit_task_result_timeout": 300,
   })


CCWF/Swarm Learning Configuration
---------------------------------

.. code-block:: python

   from nvflare.app_common.ccwf.recipes.swarm import SimpleSwarmLearningRecipe

   recipe = SimpleSwarmLearningRecipe(
       min_clients=3,
       num_rounds=10,
       model=model,
       train_script="train.py",
       cross_site_eval_timeout=600.0,
   )


Flower Integration
------------------

.. code-block:: python

   from nvflare.app_opt.flower.recipe import FlowerRecipe

   recipe = FlowerRecipe(
       server_app=ServerApp(...),
       client_app=ClientApp(...),
       superlink_ready_timeout=30.0,
       configure_task_timeout=300,
       start_task_timeout=30,
       progress_timeout=7200,
       per_msg_timeout=30.0,
       tx_timeout=300.0,
       client_shutdown_timeout=10.0,
   )


Edge Device Configuration
-------------------------

.. code-block:: python

   from nvflare.edge.tools.edge_fed_buff_recipe import EdgeFedBuffRecipe

   recipe = EdgeFedBuffRecipe(
       model=MyModel(),
       update_timeout=30,
       job_timeout=600.0,
       device_wait_timeout=120.0,
   )


TaskExchanger Configuration
---------------------------

.. code-block:: python

   from nvflare.app_common.executors.task_exchanger import TaskExchanger

   executor = TaskExchanger(
       read_interval=0.5,
       heartbeat_interval=5.0,
       heartbeat_timeout=120.0,
       resend_interval=5.0,
       peer_read_timeout=120.0,
       result_poll_interval=1.0,
   )


LauncherExecutor Configuration
------------------------------

.. code-block:: python

   from nvflare.app_common.executors.launcher_executor import LauncherExecutor

   executor = LauncherExecutor(
       launch_timeout=60.0,
       task_wait_timeout=3600.0,
       last_result_transfer_timeout=600.0,
       external_pre_init_timeout=300.0,
       peer_read_timeout=120.0,
       monitor_interval=0.5,
       read_interval=0.5,
       heartbeat_interval=10.0,
       heartbeat_timeout=120.0,
   )


ModelController-Based Workflow
------------------------------

.. code-block:: python

   from nvflare.app_common.workflows.fedavg import FedAvg

   controller = FedAvg(
       num_clients=8,
       num_rounds=100,
   )

   # Task with timeout
   controller.send_model_and_wait(
       targets=None,
       data=model,
       timeout=3600,  # 1 hour per round
   )


ScatterAndGather Configuration
------------------------------

.. code-block:: python

   from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather

   controller = ScatterAndGather(
       min_clients=4,
       num_rounds=50,
       train_timeout=7200,              # 2 hours per round
       wait_time_after_min_received=30, # Wait 30s for stragglers
       task_check_interval=1.0,
   )


CyclicController Configuration
------------------------------

.. code-block:: python

   from nvflare.app_common.workflows.cyclic_ctl import CyclicController

   controller = CyclicController(
       num_rounds=10,
       task_assignment_timeout=30,  # 30 sec to request task
   )


TIE Controller Configuration
----------------------------

.. code-block:: python

   from nvflare.app_common.tie.controller import TieController

   controller = TieController(
       configure_task_timeout=60,
       start_task_timeout=30,
       job_status_check_interval=5.0,
       max_client_op_interval=120.0,
       progress_timeout=7200.0,
   )


Notes and Best Practices
========================

**General Rules:**

- Timeout values are in **seconds** unless otherwise specified
- ``None`` or ``0`` often means no timeout limit (wait indefinitely)
- Chunk size values of ``0`` disable streaming and use native serialization

**Critical Constraints:**

- ``heartbeat_interval`` must be **less than** ``heartbeat_timeout``
- ``task_assignment_timeout`` must be **less than or equal to** ``task.timeout``
- ``task_result_timeout`` must be **less than or equal to** ``task.timeout``
- ``per_msg_timeout`` should be **less than or equal to** ``tx_timeout`` for retries to work
- ``agent_heartbeat_interval`` must be **less than** ``agent_connection_timeout``
- **IMPORTANT**: When using tensor streaming, ``get_task_timeout`` must be **greater than or equal to** 
  ``wait_send_task_data_all_clients_timeout`` to prevent task fetch timeouts while waiting for all 
  clients to receive tensors

**Tensor Streaming Timeout Warning:**

When tensor streaming is enabled, if ``get_task_timeout`` is not explicitly set, it defaults to the 
communicator's timeout. If the streaming timeout (``wait_send_task_data_all_clients_timeout``) exceeds 
the communicator timeout, clients may timeout while waiting for other clients to receive weights. This 
can cause the tensor streaming process to restart and clients may receive empty tensors, causing the 
job to fail.

**Recommended relationship for tensor streaming:**

.. code-block:: text

   get_task_timeout >= wait_send_task_data_all_clients_timeout >= tensor_send_timeout * num_clients

**Hierarchy:**

- Session-specific timeouts override server defaults
- Client config overrides can be set via ``recipe.add_client_config()``
- ``comm_config.json`` settings apply to all F3/CellNet communication

**Best Practices by Component:**

*Controllers:*

- Start with ``timeout=0`` (no timeout) during development
- Set appropriate ``train_timeout`` based on expected round duration
- For cross-site eval, ``validation_timeout`` should exceed longest validation time
- Use ``wait_for_clients_timeout`` to limit waiting for slow clients

*Executors:*

- ``external_pre_init_timeout`` should cover model loading + library imports
- ``heartbeat_timeout`` should be 2-3x ``heartbeat_interval``
- Set ``last_result_transfer_timeout`` based on result size
- For IPC: ``agent_connection_timeout`` > ``agent_heartbeat_interval`` * 3

*Workflows:*

- ``progress_timeout`` catches hung jobs; set to 2-3x expected round time
- ``job_status_check_interval`` trades responsiveness vs overhead
- For statistics: ``result_wait_timeout`` per statistic, not total

*Network/Streaming:*

- Increase ``per_msg_timeout`` and ``tx_timeout`` for high-latency networks
- ``streaming_read_timeout`` should handle slowest expected transfer
- Use longer ``ack_wait`` for unreliable connections

**Debugging Tips:**

- Enable debug logging to see timeout-related messages
- Check ``num_timeout_reqs`` counter in CoreCell for timeout statistics
- Monitor heartbeat status to detect connectivity issues early
- Look for "timeout" in logs to identify which timeouts are triggering
- For IPC issues, check ``agent_connection_timeout`` and agent logs
- For third-party integration (TIE), monitor ``max_client_op_interval`` triggers

**Common Timeout Patterns:**

1. **Layered Timeouts**: Higher-level timeouts should exceed lower-level ones
   
   - ``progress_timeout`` > ``train_timeout`` > ``task_wait_timeout``
   - ``validation_timeout`` > per-batch validation time * num_batches

2. **Heartbeat Relationships**: Always maintain proper ratios
   
   - ``heartbeat_timeout`` = 3-6x ``heartbeat_interval``
   - ``agent_heartbeat_timeout`` = 3-6x ``agent_heartbeat_interval``

3. **Retry Allowance**: Leave room for retries
   
   - ``tx_timeout`` > ``per_msg_timeout`` * expected_retries
   - ``task.timeout`` > ``task_assignment_timeout`` + actual_work_time
