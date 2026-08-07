.. _timeout_troubleshooting:

#############################
Timeout Troubleshooting Guide
#############################

This guide covers the most common timeout-related job failures and how to resolve them.
For a comprehensive reference of all timeouts, see :ref:`timeouts_programming_guide`.

.. contents:: Table of Contents
   :local:
   :depth: 2

Common Job Failure Scenarios
============================

Task Fetch Timeout
------------------

**Symptom**: Client fails to receive tasks from server; logs show "timeout" during task fetch.

**Common Causes**:

- Large model weights take too long to transfer
- Network latency exceeds default timeout
- Tensor streaming timeout exceeds task fetch timeout

**Solution**: Set ``get_task_timeout`` in client config:

.. code-block:: python

   recipe.add_client_config({
       "get_task_timeout": 300,  # 5 minutes
   })


External-Process Launch Timeout (Client API Only)
--------------------------------------------------

**Applies to**: ``ClientAPIExecutor(execution_mode="external_process")``

**Symptom**: Job fails before training starts because the launched trainer does
not establish its Client API session before ``launch_timeout``.

This timeout controls how long NVFLARE waits for the launched trainer to call
``flare.init()`` and complete its Cell session setup.

**Common Causes**:

- Large models (LLMs) take time to load before ``flare.init()`` is called
- Heavy library imports (PyTorch, TensorFlow, transformers)
- Slow disk I/O reading model weights

**Solution**: Increase ``launch_timeout`` in the executor configuration:

.. code-block:: python

   from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor

   executor = ClientAPIExecutor(
       execution_mode="external_process",
       command=["python3", "custom/train.py"],
       launch_timeout=600,  # 10 minutes for LLMs
   )


Heartbeat Timeout
-----------------

**Symptom**: Client marked as dead; logs show "heartbeat timeout" or "client not responding".

**Common Causes**:

- Long-running training blocks heartbeat thread
- Network issues causing missed heartbeats
- Client overwhelmed with compute

**Solution**: Adjust heartbeat settings:

.. code-block:: python

   # In executor configuration
   heartbeat_timeout = 300.0   # 5 minutes
   heartbeat_interval = 10.0   # Send every 10 seconds

**Rule**: ``heartbeat_interval`` must be less than ``heartbeat_timeout``.


Training Task Timeout
---------------------

**Symptom**: Training interrupted before completion; logs show task timeout.

**Common Causes**:

- Training round takes longer than expected
- Data loading is slow
- Hardware is slower than anticipated

**Solution**: Set appropriate task timeout in controller:

.. code-block:: python

   # ScatterAndGather controller
   controller = ScatterAndGather(
       train_timeout=7200,  # 2 hours per round
       wait_time_after_min_received=60,
   )

   # Or via ModelController
   controller = FedAvg(
       num_rounds=100,
       timeout=7200,  # 2 hours per round
   )


Result Submission Timeout
-------------------------

**Symptom**: Training completes but result submission fails.

**Common Causes**:

- Large model results take time to transfer
- Network congestion

**Solution**: Set ``submit_task_result_timeout``:

.. code-block:: python

   recipe.add_client_config({
       "submit_task_result_timeout": 300,  # 5 minutes
   })


Out-of-Process Client API Task and Result Waiting
-------------------------------------------------

**Applies to**: Client API ``external_process`` and ``attach`` modes

**Symptom**: The trainer does not accept a task in time, or a long training round
finishes after the Client Job has stopped waiting for its result.

**Cause**: ``task_wait_timeout`` bounds task delivery and acceptance.
``result_wait_timeout`` starts after task acceptance and therefore must cover the
training round. Large payload transfer is progress-aware and is governed by the
shared streaming idle policy rather than ``result_wait_timeout``.

**Solution**:

.. code-block:: python

   from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor

   executor = ClientAPIExecutor(
       execution_mode="attach",
       attach_id="trainer_a",
       task_wait_timeout=600,    # task materialization and trainer acceptance
       result_wait_timeout=7200, # complete training round before result publication
       heartbeat_interval=5,
       heartbeat_timeout=30,
   )

.. note::
   ``result_wait_timeout`` does not bound a result payload that is actively
   streaming. Do not shorten it to control large-object transfer duration.

.. note::
   An attached trainer owns its process and any result-transfer source. Keep it
   alive until ``flare.send()`` returns; NVFLARE will not terminate or restart it.

Swarm Learning P2P Transfer Timeout
------------------------------------

**Applies to**: ``SwarmLearningRecipe`` with large models

**Symptom**: Swarm Learning job fails with P2P ACK timeout during model scatter between peers.

**Cause**: ``round_timeout`` (which sets the P2P model-transfer ACK budget between peers)
defaults to 3600 s.  For very large models (7B+) on congested networks, peer-to-peer
tensor streaming can approach this limit.

**Solution**: Set ``round_timeout`` directly on the recipe:

.. code-block:: python

   recipe = SwarmLearningRecipe(
       name="swarm",
       model=MyModel(),
       min_clients=3,
       num_rounds=5,
       train_script="client.py",
       round_timeout=7200,  # 2 hours for 70B+ models
   )

Cross-Site Evaluation Timeout
-----------------------------

**Symptom**: Model evaluation fails or times out during cross-site validation.

**Solution**: Adjust evaluation timeouts:

.. code-block:: python

   from nvflare.app_common.np.recipes import NumpyCrossSiteEvalRecipe

   recipe = NumpyCrossSiteEvalRecipe(
       submit_model_timeout=900,      # 15 min for model submission
       validation_timeout=7200,       # 2 hours for validation
   )


Quick Reference Table
=====================

Most Commonly Adjusted Timeouts
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Timeout
     - Default
     - When to Increase
   * - get_task_timeout
     - None
     - Large models, slow networks, tensor streaming
   * - submit_task_result_timeout
     - None
     - Large result payloads
   * - launch_timeout (Client API external-process only)
     - 300 s
     - Heavy imports or model initialization before ``flare.init()``
   * - task_wait_timeout (out-of-process Client API)
     - None; Attach applies a 600 s task-delivery budget when unset
     - Slow task materialization or delayed trainer acceptance
   * - result_wait_timeout (out-of-process Client API)
     - None
     - Long training rounds before the trainer publishes its result
   * - round_timeout (Swarm Learning only)
     - 3600 s
     - 7B+ model P2P transfers between Swarm peers
   * - heartbeat_timeout
     - 60-300s
     - Long training iterations, slow networks
   * - attach_timeout (Attach mode only)
     - None
     - Bound how long a Client Job waits for an independently started trainer
   * - job_wait_timeout (Attach profile only)
     - None
     - Bound how long a trainer waits to discover a matching Client Job
   * - train_timeout
     - 0
     - Long training rounds
   * - validation_timeout
     - 6000s
     - Large validation datasets
   * - progress_timeout
     - 3600s
     - Complex multi-round workflows


Configuration Methods
=====================

Via Recipe API
--------------

.. code-block:: python

   # Client-side timeouts (applies to all clients)
   recipe.add_client_config({
       "get_task_timeout": 300,
       "submit_task_result_timeout": 300,
   })

   # Or for specific clients
   recipe.add_client_config({
       "get_task_timeout": 600,
   }, clients=["site-1", "site-2"])


Via Configuration Files
-----------------------

**application.conf** (job-level):

.. code-block::

   get_task_timeout = 300.0
   submit_task_result_timeout = 300.0

   # Server startup/dead-job safety flags
   strict_start_job_reply_check = false
   sync_client_jobs_require_previous_report = true

Server-side safety flags guidance (see :ref:`server_startup_dead_job_safety_flags` for full details):

- ``strict_start_job_reply_check`` (default ``false``): in non-strict mode, start-job timeouts are silently
  excluded from the active set with no ``min_sites``/``required_sites`` enforcement; set to ``true`` to make
  timeouts visible and have ``min_sites``/``required_sites`` constraints enforced at startup.
- ``sync_client_jobs_require_previous_report`` (default ``true``): keep enabled to avoid false dead-job reports
  caused by transient startup or sync races.

**comm_config.json** (system-level, in startup kit):

.. code-block:: json

   {
     "heartbeat_interval": 10,
     "streaming_read_timeout": 600
   }


Recommended Settings by Scenario
================================

Standard Training
-----------------

.. code-block:: python

   recipe.add_client_config({
       "get_task_timeout": 120,
   })


Large Model Training (100M+ parameters)
---------------------------------------

.. code-block:: python

   recipe.add_client_config({
       "get_task_timeout": 600,
       "submit_task_result_timeout": 600,
       "tensor_min_download_timeout": 300,  # use np_min_download_timeout for NumPy
   })


LLM/Foundation Model Training
-----------------------------

.. code-block:: python

   recipe.add_client_config({
       "get_task_timeout": 1200,
       "submit_task_result_timeout": 1800,
       "tensor_min_download_timeout": 600,  # PyTorch; use np_min_download_timeout for NumPy
   })

These recipe settings configure site task exchange and the shared streaming
download service. For ``ClientAPIExecutor`` out-of-process modes, configure
``launch_timeout``, ``task_wait_timeout``, ``result_wait_timeout``, and the
heartbeat settings on the executor itself when those bounds are needed. The
legacy Pipe/FlareAgent settings ``submit_result_timeout``,
``download_complete_timeout``, ``PEER_READ_TIMEOUT``, and ``max_resends`` are
removed legacy settings and have no effect on ``ClientAPIExecutor``. Use the
executor's session timeouts and the shared streaming configuration described
above instead.


High-Latency Networks
---------------------

.. code-block:: python

   # Longer communication timeouts
   recipe.add_client_config({
       "get_task_timeout": 600,
       "submit_task_result_timeout": 600,
   })

System-level (``comm_config.json`` in startup kit):

.. code-block:: json

   {
     "heartbeat_interval": 15,
     "streaming_read_timeout": 600
   }


Streaming Stall Guardrail (``comm_config.json``)
------------------------------------------------

For large payload/model transfers, configure F3 stream stall detection in
``comm_config.json`` (server and client startup kits).

**Runtime defaults** (if not set explicitly):

- ``streaming_send_timeout``: ``30.0`` seconds
- ``streaming_ack_progress_timeout``: ``60.0`` seconds
- ``streaming_ack_progress_check_interval``: ``5.0`` seconds
- ``sfm_send_stall_timeout``: ``45.0`` seconds
- ``sfm_close_stalled_connection``: ``false`` (warn-only)
- ``sfm_send_stall_consecutive_checks``: ``3``

**Recommended deployment guideline**:

1. Start with **warn-only** to observe behavior safely.
2. If repeated stall warnings are observed during large-model streaming, enable auto-close.
3. Keep the guard enabled with consecutive checks to reduce false alarms.

Warn-only baseline:

.. code-block:: json

   {
     "sfm_close_stalled_connection": false,
     "sfm_send_stall_timeout": 75,
     "sfm_send_stall_consecutive_checks": 3
   }

Auto-recovery mode (when needed):

.. code-block:: json

   {
     "sfm_close_stalled_connection": true,
     "sfm_send_stall_timeout": 75,
     "sfm_send_stall_consecutive_checks": 3
   }

**Timing relationship (important)**:

- ``sfm_send_stall_timeout`` is compared against the total continuous blocked-send duration.
- ``sfm_send_stall_consecutive_checks`` counts consecutive heartbeat monitor ticks (every 5 seconds),
  not multiples of ``sfm_send_stall_timeout``.

Approximate auto-close window (when ``sfm_close_stalled_connection=true``):

.. code-block:: text

   close_lower_bound ~= sfm_send_stall_timeout + (HEARTBEAT_TICK * (sfm_send_stall_consecutive_checks - 1))
   close_upper_bound ~= sfm_send_stall_timeout + (HEARTBEAT_TICK * sfm_send_stall_consecutive_checks)

With ``sfm_send_stall_timeout=75`` and ``sfm_send_stall_consecutive_checks=3``, close typically occurs
around ``85``-``90`` seconds of continuous stall (not 225 seconds).

**Outer-timeout guideline**:

Set higher-layer timeouts (for example ``communication_timeout`` or task/request timeouts that include
message transfer time) greater than ``close_upper_bound`` plus a safety margin.

Example: ``communication_timeout=300`` is safely larger than the ~``90`` second stall auto-close window.

**How to interpret logs**:

- Expected warning on real stalls:
  ``Detected stalled send on ... (N/3)``
- In healthy/normal streaming, no stall warning should be emitted.
- Intermittent stalls should not close the connection unless the threshold is reached in consecutive checks.


Large-Scale Hierarchical / HPC Deployments (Slurm, Lustre)
------------------------------------------------------------

When running 100+ FL clients in a hierarchical topology on HPC systems with shared
filesystems (Lustre, GPFS), two settings significantly improve startup reliability:

**1. Set a minimum-client tolerance in** ``config_fed_server.json``

Allow a small number of clients to be late or unavailable at startup without aborting
the job. For a 144-client job, tolerating up to ~4% stragglers is safe:

.. code-block:: json

   {
     "workflows": [{
       "id": "controller",
       "path": "nvflare.app_common.workflows.fedavg.FedAvg",
       "args": {
         "num_clients": 144,
         "min_clients": 138
       }
     }]
   }

**2. Extend the runner sync timeout in** ``config_fed_client.json``

With the default runner sync settings (a 2.0-second per-request timeout with overall
sync bounded by ``max_runner_sync_timeout``), many clients contending for Lustre I/O
at job launch can time out before finishing initialization. Increase these values to
give each client more time to start up:

.. code-block:: json

   {
     "runner_sync_timeout": 120,
     "max_runner_sync_timeout": 7200
   }

These two changes address the most common startup race conditions in large hierarchical
deployments and are compatible with the startup stability fixes in FLARE 2.7.2.


Debugging Timeout Issues
========================

1. **Check logs** for "timeout" messages to identify which timeout triggered
2. **Enable debug logging** to see detailed timing information
3. **Monitor heartbeat status** in admin console
4. **Start with longer timeouts** during development, then optimize

For timeout hierarchies, relationships, and all available timeout parameters, 
see the comprehensive :ref:`timeouts_programming_guide`.
