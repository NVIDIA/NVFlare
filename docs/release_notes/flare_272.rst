**************************
What's New in FLARE v2.7.2
**************************

NVIDIA FLARE 2.7.2 is a feature release that builds on the Job Recipe API introduced in 2.7.0,
bringing it to general availability.
This release also delivers major system hardening across the F3 streaming layer, comprehensive
memory management improvements for large-model training, and startup stability fixes for
large-scale hierarchical FL deployments.

Job Recipe API - Generally Available
=====================================

.. sidebar::

    **Recipe-based Example**

    .. code-block:: python

            from nvflare.app_opt.pt.recipes import FedAvgRecipe
            from nvflare.recipe import SimEnv

            recipe = FedAvgRecipe(
                name="hello-pt",
                min_clients=2,
                num_rounds=5,
                model=SimpleNetwork(),
                train_script="client.py",
            )
            env = SimEnv(num_clients=2)
            run = recipe.execute(env)

The Job Recipe API, introduced as a technical preview in 2.7.0, is now generally available with comprehensive coverage across all major examples.
Almost all examples in the NVFlare repository have been converted to use Job Recipes, demonstrating the simplicity and power of this approach.

Key Highlights
~~~~~~~~~~~~~~

- **Unified Recipe Architecture**: All framework-specific recipes (PyTorch, TensorFlow, NumPy, scikit-learn) now inherit from a unified base recipe, ensuring consistent behavior and easier maintenance.

- **Comprehensive Recipe Library**: Ready-to-use recipes for:

  - **FedAvg** (PyTorch, TensorFlow, NumPy, scikit-learn)
  - **FedProx** (via FedAvg with proximal loss helper)
  - **FedOpt** (server-side optimization with SGD, Adam, etc.)
  - **SCAFFOLD** (control variates for data heterogeneity)
  - **Cyclic Learning** (sequential client training)
  - **XGBoost** (horizontal, vertical, and bagging modes)
  - **Federated Statistics** (distributed statistics computation)
  - **FedEval** (federated evaluation of pre-trained models)
  - **Cross-Site Evaluation** (model evaluation across sites)
  - **PSI** (Private Set Intersection)
  - **Flower Integration**
  - **Swarm Learning** (decentralized FL)
  - **Edge Recipes** (for edge device FL)

- **Simplified Example Structure**: All Hello World and advanced examples now follow a consistent pattern with ``job.py`` scripts using the Recipe API.

- **Consolidated Examples**: Examples have been streamlined and consolidated. Redundant examples using deprecated APIs (such as the old Executor-based and ModelLearner-based patterns) have been removed to reduce confusion and maintenance burden.

- **Environment Flexibility**: The same recipe works seamlessly across:

  - **SimEnv**: Local simulation for development
  - **PocEnv**: Multi-process proof-of-concept
  - **ProdEnv**: Production deployment

.. admonition:: Available Recipes

    For a complete list of available recipes with code examples and links to corresponding examples, see :ref:`available_recipes`.

Memory Management
-----------------

FLARE 2.7.2 delivers a full memory management stack for large-model FL, covering every tier
from the FL server down to the client training subprocess. Three complementary features work
together to reduce peak memory usage, prevent RSS (Resident Set Size) growth across rounds,
and eliminate unnecessary serialization overhead in subprocess mode.

**TensorDownloader** — Zero-code memory optimization for PyTorch large models. Model tensors
are serialized incrementally in safetensors format and distributed via a pull-based download
service. The Downloader Service reduces peak memory on both server and client, in both
in-process and subprocess (external-process) modes.

**Zero-copy relay at the Client Job (CJ) process (Pass-Through)** — For subprocess-mode
clients (``ClientAPILauncherExecutor``), the CJ relay process previously deserialized and
re-serialized every model tensor before forwarding it to the training subprocess, doubling
the relay-tier memory footprint per round. FLARE 2.7.2 introduces pass-through forwarding:
the CJ holds lightweight ``LazyDownloadRef`` placeholders and the training subprocess
downloads tensors directly from the FL server, making CJ memory independent of model size.
This is particularly impactful for LLM-scale models (7B–70B parameters).

**Server and Client Memory Cleanup** — FLARE 2.7.2 introduces automatic RSS stabilization
for long-running jobs, preventing unbounded memory growth across both the server and the full
client pipeline:

- **Server**: Periodic ``gc.collect()`` + ``malloc_trim()`` after aggregation rounds returns
  freed pages to the OS. Platform-aware: full heap trimming on Linux/glibc, with safe
  fallbacks on macOS/musl. Overhead is 10–500 ms per round — negligible compared to
  training time.
- **Client (subprocess mode)**: The same GC and heap-trim cycle runs across the entire client
  pipeline — both the CJ relay process and the training subprocess — after every
  ``flare.send()`` call. Optionally includes ``torch.cuda.empty_cache()`` for GPU memory.
  Configurable via ``client_memory_gc_rounds``; no training script changes required.
- ``MALLOC_ARENA_MAX`` tuning guidance is provided for controlling glibc arena fragmentation
  on both server and client.

.. admonition:: Learn More

   For configuration details, platform compatibility, recommended settings, and API reference,
   see :doc:`/programming_guide/memory_management`.

Together, these three features significantly reduce peak memory usage, prevent RSS growth
that leads to OOM errors, and eliminate redundant serialization/deserialization overhead
when training in subprocess mode. In a 5 GB PyTorch model benchmark with FedAvg and 4
clients, server peak memory dropped by up to 85% (in-process mode) and client peak memory
by up to 92% (subprocess mode), as detailed in the tables below.
Prior to 2.7.2, subprocess jobs with large models would OOM after a few rounds; with the
full set of fixes they now complete stably across many rounds.

**FedAvg — 5 GB model, 4 clients**

**In-process mode**

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Metric
     - v2.7.0
     - v2.7.2
     - Improvement
   * - Server peak (GB)
     - 264
     - 40
     - −85%
   * - Client peak avg (GB)
     - ~23
     - ~5.6
     - −76%


**External-process (subprocess) mode**

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Metric
     - v2.7.0
     - v2.7.2
     - Improvement
   * - Server peak (GB)
     - 193.7
     - 48.3
     - −75%
   * - Client peak avg (GB)
     - 62.6
     - 5.3
     - −92%


**Swarm Learning — 2.5 GB model, 3 sites**

**In-process mode**

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Metric
     - v2.7.0
     - v2.7.2
     - Improvement
   * - Site peak avg (GB)
     - 54.2
     - 26.2
     - −52%


**External-process (subprocess) mode**

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Metric
     - v2.7.0
     - v2.7.2
     - Improvement
   * - Site peak avg (GB)
     - 54.2
     - 27.8
     - −49%


Reliability and Performance
-----------------------------------------

FLARE 2.7.2 hardens the F3 streaming layer with fixes for five concurrency and stability
issues that surfaced at scale, particularly in hierarchical and large-model deployments.

**Head-of-line stall mitigation** — A slow or congested connection can no longer hold the
SFM send lock indefinitely, blocking heartbeats and admin traffic behind a large frame send.
A bounded send timeout, an ACK-progress watchdog, and optional connection reset
(``SFM_CLOSE_STALLED_CONNECTION``) work together to detect and recover from stalls.
See :ref:`timeout_troubleshooting` — *Streaming Stall Guardrail* for recommended settings.

**Concurrency fixes** — Three separate issues resolved: stream pool starvation (streaming
callbacks were dispatched on the same thread pool they depended on, causing indefinite stall);
an RxTask self-deadlock on stream error signals; and serialized cache-miss production that
bottlenecked all concurrent clients behind a single lock at high client counts. Concurrent
downloads now complete reliably with a dedicated callback pool and reduced lock scope.

**Streaming download retry** — Transient timeouts during tensor streaming (common in LLM
swarming over congested networks) now trigger structured exponential-backoff retry (up to
3 attempts, capped at 60 s), abort-signal aware and idempotent.


**Large-model subprocess reliability** — Send retries no longer accumulate per-attempt model
copies in memory. Three timeout parameters previously hardcoded at values too short for large
models (``submit_result_timeout``, ``tensor_min_download_timeout`` / ``np_min_download_timeout``,
``max_resends``) are now configurable via ``recipe.add_client_config({...})``. Swarm Learning
and SAG workflows also gain a ``min_clients`` fault-tolerance threshold so a job can proceed
when a small number of configured clients are unavailable.

Hierarchical FL Startup Stability
-----------------------------------

Large-scale hierarchical FL deployments are prone to startup race conditions that can abort
jobs before training begins. FLARE 2.7.2 addresses these with a set of coordinated
reliability fixes: deployment timeouts are now correctly classified as failures (triggering
the existing ``min_sites`` tolerance check rather than silently proceeding); a startup grace
period prevents premature dead-client detection while clients are still initializing; and
stragglers that miss the start-job window are selectively excluded rather than aborting the
entire job, provided the active count still satisfies ``min_clients``. Together these changes
make large hierarchical jobs significantly more robust against transient startup delays.

For recommended configuration settings for HPC environments (Slurm, Lustre filesystems),
see :ref:`timeout_troubleshooting` — *Large-Scale Hierarchical / HPC Deployments* scenario.

Comprehensive Timeout Documentation
------------------------------------

Two new timeout guides have been added:

**Timeout Troubleshooting Guide** (:doc:`/user_guide/timeout_troubleshooting`) — A user-facing guide covering common timeout-related job failures and how to resolve them. Covers the most frequently encountered timeout scenarios with symptoms, causes, and fixes.

**Timeouts Reference** (:doc:`/programming_guide/timeouts`) — A comprehensive programming reference covering all 100+ timeout parameters across NVFlare components, organized by functional categories:

- **Network Communication**: F3/CellNet, server config, client config, gRPC, reliable message
- **Executor and Launcher**: LauncherExecutor, TaskExchanger, IPCExchanger, Pipe Handler
- **Workflow Controllers**: FedAvg, SAG, CrossSiteEval, Statistics, SplitNN, etc.
- **Edge Devices**: Edge general, Hierarchical FL, Mobile client
- **Streaming**: File, container, tensor, object streaming
- **XGBoost**: Histogram controller, reliable message, gRPC client
- **Configuration Locations**: System-level and job-level file paths
- **Recommended Settings**: Use-case specific configurations (development, production, LLM training, edge devices)

Additional Improvements
-----------------------

Example Consolidation
~~~~~~~~~~~~~~~~~~~~~

To provide a cleaner and more focused learning experience, we have consolidated and streamlined the examples:

- **Removed Deprecated Examples**: Most examples using old APIs (Executor-based, ModelLearner-based patterns) have been removed. The majority of examples now use the modern Recipe API or Client API.

- **Unified Example Structure**: Each example now follows a consistent structure with a ``job.py`` entry point that uses the Recipe API, making it easier to understand and adapt.

- **Reduced Redundancy**: Duplicate examples demonstrating the same concepts with different APIs have been consolidated into single, canonical examples.

- **Focus on Best Practices**: Remaining examples showcase the recommended patterns for building federated learning applications with FLARE.

- **New Example**: **Hello Differential Privacy** (``hello-world/hello-dp``) — Demonstrates federated learning with differential privacy using the Recipe API.

.. note::

    A few examples and tutorials still use older APIs. These will continue to be updated in upcoming releases.

MONAI Integration
~~~~~~~~~~~~~~~~~

- **MONAI-FLARE Wheel Deprecated**: The separate ``nvflare-monai`` wheel package is now deprecated. MONAI integration is now achieved directly through the Client API, simplifying the integration and reducing dependency management overhead. For further information, see the `MONAI Migration Guide <https://github.com/NVIDIA/NVFlare/blob/main/integration/monai/MIGRATION.md>`_.

- **Updated MONAI Examples**: All MONAI examples have been updated to use the Client API pattern, making it easier to integrate MONAI training workflows with FLARE without requiring additional packages.

Documentation
~~~~~~~~~~~~~

- **Available Recipes Guide**: New :ref:`available_recipes` guide with code examples and links to working examples for all available recipes.

- **Timeout Documentation**: New :doc:`/user_guide/timeout_troubleshooting` for common timeout-related job failures and fixes (task fetch, external process pre-init, submit result, etc.), and :doc:`/programming_guide/timeouts` as the comprehensive reference for all 100+ timeout parameters by component and use case.

- **Memory Management Guide**: New :doc:`/programming_guide/memory_management` covering server-side and client-side garbage collection, ``MALLOC_ARENA_MAX`` tuning, platform compatibility, and troubleshooting.

- **Tensor Downloader Guide**: Expanded :doc:`/programming_guide/tensor_downloader` with configuration examples, architecture details, and tuning guidance.

- **Hello Differential Privacy**: New :doc:`/hello-world/hello-dp/index` example and documentation.

- **Client-Controlled Workflows**: Expanded documentation for :doc:`/programming_guide/controllers/client_controlled_workflows`.

- **Job Recipe Guide**: Updated :doc:`/user_guide/data_scientist_guide/job_recipe` with dict model config and initial checkpoint examples.

Bug Fixes
~~~~~~~~~

- Fixed OOM accumulation on subprocess send retry: a single serialized payload is now reused across retries rather than re-serializing per attempt.
- Fixed subprocess task-fetch stall: the client training process now acknowledges task receipt immediately instead of waiting for download completion, preventing subprocess timeout during large-model transfers.
- Fixed CSE model-load failure after external-process training: cross-site evaluation now uses the on-disk persistor instead of relaunching the already-exited training subprocess.
- Fixed ``SwarmServerController`` crash when ``min_clients`` is omitted from JSON config (``None < 0`` TypeError replaced with ``int = 0`` default).
- Fixed ``max_resends`` silently ignored in subprocess executor due to private attribute shadowing.
- Fixed gRPC session resource leak in ``nvflare job submit`` when the server is unreachable.
- Fixed connection manager crash on frame arrival after job teardown.
- Fixed F3 streaming Head-of-Line stall: ``send_frame()`` no longer holds the connection lock without a timeout bound.
- Fixed RxTask self-deadlock triggered by stream error signals during active receive.
- Fixed stream thread pool starvation that prevented concurrent model downloads from completing.
- Fixed deployment timeout silent pass-through: timed-out clients are now counted against ``min_sites``.
- Fixed premature dead-job detection: clients are no longer reported missing before their first positive heartbeat.
- Fixed ``TypeError`` crash in client job process when job metadata is absent (replaced with descriptive ``RuntimeError``).
- Fixed Swarm Learning self-message deadlock for local result submission.
- Fixed TLS corruption by replacing ``fork`` with ``posix_spawn`` for subprocess creation.
- Fixed potential data corruption issue in the Streamer component.
- Fixed Swarm Learning controller compatibility with tensor streaming.
- Fixed XGBoost adaptor and recipe integration issues.
- Addressed client-side vulnerability for tree-based horizontal XGBoost.
- Fixed NumPy cross-site evaluation regression.
- Fixed POC Run result caching and environment cleanup.
- Fixed TensorBoard analytics receiver import error.
- Improved error handling in FOBS serialization (raise exception on errors).
- Improved error messages in Client API.
- Security fix: Fixed a Remote Code Execution vulnerability in FOBS deserialization. The ``Packer.unpack()`` method failed to validate the attacker-controlled ``type_name`` before passing it to ``load_class()``, allowing authenticated participants to execute arbitrary Python code on the aggregation server. Fixed by introducing a ``BUILTIN_TYPES`` allowlist and validating ``type_name`` before class loading. A public API ``add_type_name_whitelist()`` is provided for runtime extension with custom types.
- Security fix: Fixed a path traversal vulnerability in ``FileRetriever`` by enforcing source-directory boundary checks on requested files, preventing ``../`` traversal attacks from escaping the allowed directory.
- Updated PEFT/TRL integration for latest API compatibility.
- Updated HuggingFace LLM integration.
- Security dependency updates for web components.

Migration Guide
---------------

For detailed migration steps including API changes, renamed parameters, and backward
compatibility notes, see the :ref:`Migration Guide <migration_guide>`.

Getting Started
---------------

The easiest way to get started with FLARE 2.7.2 is through the Hello World examples:

.. code-block:: bash

    # Run the PyTorch FedAvg example
    cd examples/hello-world/hello-pt
    python job.py

For more examples and tutorials, see:

- :ref:`quickstart` — Get up and running quickly
- :ref:`available_recipes` — Complete list of ready-to-use recipes
- :ref:`job_recipe` — Job Recipe programming guide
- `Hello World Examples <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world>`_
- `Advanced Examples <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced>`_
- `Self-Paced Training Tutorials <https://github.com/NVIDIA/NVFlare/tree/main/examples/tutorials/self-paced-training>`_


