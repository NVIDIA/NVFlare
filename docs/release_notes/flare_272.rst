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

FLARE 2.7.2 delivers a full memory management stack covering the server, the CJ relay process,
and the client training process — addressing the peak memory challenges that arise when running
large-model FL at scale.

Memory Management with Tensor-based Downloader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FLARE 2.7.2 introduces the **TensorDownloader** for PyTorch models, extending the FileDownloader concept introduced in 2.7.0 specifically for tensor data.
This feature addresses critical memory challenges when working with large language models (LLMs) and other large-scale models in federated learning.

Key Features
^^^^^^^^^^^^

- **Zero Code Changes Required**: Your existing PyTorch FL jobs benefit from memory optimization without any modification.

- **Incremental Tensor Serialization**: Instead of serializing all model parameters at once, tensors are serialized individually using safetensors format, significantly reducing peak memory consumption.

- **Pull-based Architecture**: Unlike push-based streaming, each recipient pulls data at its own pace, making it more reliable for heterogeneous network conditions.

Performance Results
^^^^^^^^^^^^^^^^^^^

Based on our internal testing with a 5GB model and 4 clients using FedAvg, we observed **20% to 50% memory usage reduction** on both server and client sides.

.. note::

    Your results may vary depending on model size, number of clients, network conditions, and different FL algorithms and workflows.

Benefits for LLM Training
^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Reduced Memory Footprint**: 20-50% reduction critical for large models that approach memory limits
- **Improved Scalability**: Multiple clients can download at different rates without blocking
- **Safetensors Format**: Secure and efficient tensor serialization without pickle vulnerabilities
- **No Migration Required**: Existing PyTorch jobs automatically benefit from this optimization

.. admonition:: Learn More

    **Transparent & zero code changes** -- the TensorDownloader works automatically in all PyTorch workflows.
    Supports **PyTorch tensors and NumPy arrays** (TensorFlow uses traditional serialization).

    - User guide with configuration and tuning: :ref:`tensor_downloader`
    - FOBS decomposer architecture: :ref:`decomposer_for_large_object`

Zero Tensor Copy at the CJ Process (Pass-Through)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For hierarchical and large-model deployments, the Client Job (CJ) relay process previously
deserialized and re-serialized every model tensor before forwarding it to the client subprocess.
This doubled the memory footprint at the relay tier for every round.

FLARE 2.7.2 introduces a **pass-through architecture** for ``ClientAPILauncherExecutor``:

- **Lazy references instead of full tensors**: The CJ process holds lightweight
  ``LazyDownloadRef`` placeholders rather than materializing the full model, so the CJ
  memory footprint is independent of model size.
- **Direct subprocess download**: The training subprocess fetches tensors directly from the
  FL server, eliminating the CJ as a memory bottleneck and halving network transfers between
  the server and CJ tier.
- **Zero code changes**: Existing jobs using ``ClientAPILauncherExecutor`` benefit
  automatically.

This is particularly impactful for LLM-scale models (7B–70B parameters) where CJ memory
previously equalled the full model size.

Client-Side Memory Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FLARE 2.7.2 extends memory lifecycle control to the client training process, complementing
the existing server-side cleanup:

- **Allocator-aware cleanup**: After each ``flare.send()`` call, FLARE automatically
  invokes ``gc.collect()`` plus allocator-specific trimming — ``malloc_trim(0)`` for
  glibc (Linux), jemalloc arena purge where available, and ``torch.cuda.empty_cache()``
  for GPU memory — returning freed pages to the OS between rounds.
- **Configurable frequency**: Cleanup runs every ``N`` rounds (default: every round),
  configurable via recipe parameters (``client_memory_gc_rounds``) and ``ScriptRunner``.
- **No training script changes**: Cleanup is injected transparently into the FLARE
  client lifecycle without touching user training code.
- **Combined with server-side cleanup**: Together with the server-side garbage collection
  introduced in 2.7.2, this prevents unbounded RSS growth in both the server and client
  processes across long-running jobs with many rounds.

Server-Side Memory Cleanup
~~~~~~~~~~~~~~~~~~~~~~~~~~~

FLARE 2.7.2 adds automatic server-side memory management to address RSS (Resident Set Size — the actual physical memory used by a process) growth in long-running jobs:

- **Periodic garbage collection and heap trimming**: Automatically runs ``gc.collect()`` and ``malloc_trim()`` to return freed memory back to the OS, preventing unbounded RSS growth over many training rounds.
- **Environment variable tuning**: Guidance on ``MALLOC_ARENA_MAX`` settings to control glibc memory arena fragmentation for both server and client processes.
- **Platform-aware**: Memory cleanup adapts to the runtime platform (Linux/glibc, musl, macOS), with full heap trimming on Linux/glibc and safe fallbacks elsewhere.
- **Minimal overhead**: Cleanup takes 10-500ms per invocation — negligible compared to typical training round durations.

On the client side, ``flare.send(..., clear_cache=True)`` (default) releases parameter references
after serialization. This reference-release path is the primary mechanism to reclaim large tensor
objects; ``gc.collect()`` is a supplemental safeguard mainly for cyclic references.

.. admonition:: Learn More

    For configuration details, platform compatibility, recommended settings, and API reference, see :doc:`/programming_guide/memory_management`.

F3 Streaming Reliability and Performance
-----------------------------------------

A focused hardening effort on the F3 streaming layer addresses several concurrency and
stability issues that manifested at scale, particularly in hierarchical and large-model
deployments.

Head-of-Line (HOL) Stall Mitigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In 2.7.0/2.7.1, a slow or congested connection could hold the per-connection SFM send lock
indefinitely, blocking all outgoing traffic on that relay — heartbeats, admin commands, and
task requests — behind a single large frame send.

FLARE 2.7.2 eliminates this with a multi-layer guard:

- **Bounded send timeout**: ``send_frame()`` now has a configurable deadline
  (``STREAMING_SEND_TIMEOUT``); a send that exceeds it raises rather than blocking forever.
- **ACK-progress watchdog**: A background monitor checks that ACKs advance within
  ``STREAMING_ACK_PROGRESS_TIMEOUT``; if a connection stalls it is flagged.
- **Stall detection and optional recovery**: Consecutive stall detections (configurable via
  ``SFM_SEND_STALL_CONSECUTIVE_CHECKS``) can optionally trigger connection reset
  (``SFM_CLOSE_STALLED_CONNECTION``), unblocking all pending traffic.

For recommended settings, see :ref:`timeout_troubleshooting` — *Streaming Stall Guardrail* section.

Stream Pool Starvation Fix
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Concurrent model downloads could stall indefinitely when streaming callbacks were dispatched
on the same thread pool they depended on, exhausting it. The fix routes callbacks to a
dedicated pool, keeping stream workers free. An end-to-end test validates that 8 concurrent
downloads complete without starvation.

Streaming Download Retry on Timeout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transient timeouts during streaming downloads (particularly in LLM swarming scenarios over
congested networks) previously resulted in silent stream loss. FLARE 2.7.2 adds structured
retry semantics:

- **Exponential-backoff retry**: Up to 3 retries with configurable backoff, capped at 60 s.
- **Abort-signal aware**: Retry loop respects abort signals; no stale retries after job stop.
- **State-safe**: Retry is idempotent; re-requesting the same stream is safe for the server.

RxTask Self-Deadlock Fix
~~~~~~~~~~~~~~~~~~~~~~~~~

Stream error signals arriving during an active receive could cause a self-deadlock in the
receiver cleanup path. The fix defers cleanup until after the critical section is exited,
eliminating the deadlock without changing error-handling correctness.

Lock Contention Reduction in Model Downloads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the cacheable streaming layer, cache-miss production previously serialized all concurrent
clients behind a single lock, increasing model-download latency at high client counts (e.g.,
24 per relay). The lock scope has been reduced so production runs concurrently, significantly
improving throughput when many clients request the same model chunk at once.

.. Changes in this section are introduced by PR #4209 (https://github.com/NVIDIA/NVFlare/pull/4209).
.. Merge this release notes PR only after PR #4209 has landed on the 2.7 branch.

Hierarchical FL Startup Stability
-----------------------------------

Large-scale hierarchical FL deployments (many clients across relay tiers) are subject to
startup race conditions that can abort jobs before training begins. FLARE 2.7.2 addresses
these with a set of coordinated fixes and new configuration controls.

.. note::

    This fix set is particularly relevant for HPC deployments such as Frontier (ORNL) and
    similar supercomputers where 100+ FL clients start via a batch scheduler (e.g., Slurm)
    under shared filesystem (Lustre) load.

Deployment Timeout Now Treated as Failure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, a client that did not acknowledge job deployment within the timeout window
(``reply=None``) was silently treated as successfully deployed. The server proceeded to
start the job including that client in the participant list, creating a state inconsistency
that led to premature dead-client detection and job abort.

FLARE 2.7.2 correctly classifies deployment timeouts as failures, applying the existing
``min_sites`` / ``required_sites`` tolerance check at the deployment phase. Timed-out
clients are excluded from the job before ``start_client_job`` is called, preventing the
state inconsistency from ever forming.

Startup Grace Period for Dead-Client Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The server's heartbeat monitor previously fired a dead-job notification on the very first
heartbeat from a client that was not yet running the job — there was no startup grace period.
For clients that were still initializing (slow filesystem, GPU allocation, subprocess
spawning), this caused premature dead-client classification.

FLARE 2.7.2 adds a debounce mechanism: a client must first be positively observed reporting
the job in a heartbeat before a subsequent missing report triggers a dead-job notification.
This gives clients the time they need to start without false alarms.

This behavior is now the **default** (``sync_client_jobs_require_previous_report=true``).
Operators who need the legacy aggressive detection can opt out via configuration.

Selective Client Exclusion on Start-Job Timeout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When strict start-job reply checking is enabled
(``strict_start_job_reply_check=true``), clients that time out at the start-job phase are
now **excluded from the run** rather than causing a full job abort — provided the remaining
active client count still satisfies ``min_clients``. A warning is logged identifying the
excluded clients.

This allows a job to proceed with e.g., 142 of 144 clients when 2 stragglers fail to
respond, rather than aborting when the training majority is ready.

Hardened Client Job Metadata Parsing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a client process started after the job was already aborted, it would crash with an
opaque ``TypeError: 'NoneType' object is not iterable`` when reading job client metadata.
FLARE 2.7.2 replaces this with an explicit ``RuntimeError`` that names the missing field,
making the failure actionable in logs.

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

- **Timeout Documentation**: New :doc:`/user_guide/timeout_troubleshooting` for common job failures, and :doc:`/programming_guide/timeouts` as comprehensive reference for all 100+ timeout parameters.

- **Memory Management Guide**: New :doc:`/programming_guide/memory_management` covering server-side and client-side garbage collection, ``MALLOC_ARENA_MAX`` tuning, platform compatibility, and troubleshooting.

- **Tensor Downloader Guide**: Expanded :doc:`/programming_guide/tensor_downloader` with configuration examples, architecture details, and tuning guidance.

- **Hello Differential Privacy**: New :doc:`/hello-world/hello-dp/index` example and documentation.

- **Client-Controlled Workflows**: Expanded documentation for :doc:`/programming_guide/controllers/client_controlled_workflows`.

- **Job Recipe Guide**: Updated :doc:`/user_guide/data_scientist_guide/job_recipe` with dict model config and initial checkpoint examples.

Bug Fixes
~~~~~~~~~

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


