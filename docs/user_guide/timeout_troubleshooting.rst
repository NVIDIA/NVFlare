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


External Process Pre-Init Timeout (Client API Only)
----------------------------------------------------

**Applies to**: Client API with subprocess launcher (``ScriptRunner``, ``ClientAPILauncherExecutor``)

**Symptom**: Job fails before training starts with "external_pre_init_timeout" error.

This timeout controls how long NVFlare waits for your external training script to call ``flare.init()``.
When using Client API, NVFlare launches your script as a subprocess and waits for it to connect back.

**Common Causes**:

- Large models (LLMs) take time to load before ``flare.init()`` is called
- Heavy library imports (PyTorch, TensorFlow, transformers)
- Slow disk I/O reading model weights

**Solution**: Increase ``external_pre_init_timeout`` in the executor configuration:

.. code-block:: python

   from nvflare.app_common.executors.client_api_launcher_executor import ClientAPILauncherExecutor

   executor = ClientAPILauncherExecutor(
       external_pre_init_timeout=600,  # 10 minutes for LLMs
       ...
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
   * - external_pre_init_timeout (Client API subprocess only)
     - 60-300s
     - LLMs, heavy imports before ``flare.init()``
   * - heartbeat_timeout
     - 60-300s
     - Long training iterations, slow networks
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

**comm_config.json** (system-level, in startup kit):

.. code-block:: json

   {
     "heartbeat_interval": 10,
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

**How to interpret logs**:

- Expected warning on real stalls:
  ``Detected stalled send on ... (N/3)``
- In healthy/normal streaming, no stall warning should be emitted.
- Intermittent stalls should not close the connection unless the threshold is reached in consecutive checks.


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
   })


LLM/Foundation Model Training
-----------------------------

.. code-block:: python

   recipe.add_client_config({
       "get_task_timeout": 1200,
       "submit_task_result_timeout": 1200,
   })


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


Debugging Timeout Issues
========================

1. **Check logs** for "timeout" messages to identify which timeout triggered
2. **Enable debug logging** to see detailed timing information
3. **Monitor heartbeat status** in admin console
4. **Start with longer timeouts** during development, then optimize

For timeout hierarchies, relationships, and all available timeout parameters, 
see the comprehensive :ref:`timeouts_programming_guide`.
