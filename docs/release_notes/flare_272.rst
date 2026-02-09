**************************
What's New in FLARE v2.7.2
**************************

NVIDIA FLARE 2.7.2 is a feature release that builds on the Job Recipe API introduced in 2.7.0, bringing it to general availability.
This release also introduces significant memory management improvements with the new Tensor-based Downloader for efficient large model handling,
and comprehensive timeout documentation.

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

Memory Management with Tensor-based Downloader
----------------------------------------------

.. sidebar::

    **Transparent Memory Optimization**

    No code changes required! The TensorDownloader works automatically behind the scenes, reducing memory usage without any modifications to your training scripts.

FLARE 2.7.2 introduces the **TensorDownloader** for PyTorch models, extending the FileDownloader concept introduced in 2.7.0 specifically for tensor data.
This feature addresses critical memory challenges when working with large language models (LLMs) and other large-scale models in federated learning.

.. note::

    The Tensor Downloader supports **PyTorch tensors and NumPy arrays**. TensorFlow models currently use traditional serialization.

.. important::

    **Transparent to Users**: This optimization is built into all PyTorch workflows and completely transparent to end users. It works automatically without requiring any changes to your existing training code or job configurations. The TensorDecomposer integrates seamlessly with the existing FOBS serialization system.

Key Features
~~~~~~~~~~~~

- **Zero Code Changes Required**: Your existing PyTorch FL jobs benefit from memory optimization without any modification.

- **Incremental Tensor Serialization**: Instead of serializing all model parameters at once, tensors are serialized individually using safetensors format, significantly reducing peak memory consumption.

- **Pull-based Architecture**: Unlike push-based streaming, each recipient pulls data at its own pace, making it more reliable for heterogeneous network conditions.

Performance Results
~~~~~~~~~~~~~~~~~~~

Based on our internal testing with a 5GB model and 4 clients using FedAvg, we observed **20% to 50% memory usage reduction** on both server and client sides.

.. note::

    Your results may vary depending on model size, number of clients, network conditions, and different FL algorithms and workflows.

How It Works
~~~~~~~~~~~~

The TensorDownloader operates transparently behind the scenes:

1. **Server prepares tensors**: Model state dict is automatically registered with the downloader, generating a reference ID (RID).
2. **RID broadcast**: The server broadcasts the RID to all clients via a lightweight message.
3. **Client-side download**: Each client downloads tensors incrementally, reconstructing the model state dict.

For advanced users who need direct control, the low-level API is available:

.. code-block:: python

    from nvflare.app_opt.pt.tensor_downloader import add_tensors, download_tensors

    # Server side: Register tensors for download
    ref_id = add_tensors(downloader, model.state_dict())

    # Client side: Download tensors incrementally
    status, state_dict = download_tensors(
        from_fqcn=server_fqcn,
        ref_id=ref_id,
        per_request_timeout=30.0,
        cell=cell,
    )

Benefits for LLM Training
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Reduced Memory Footprint**: 20-50% reduction critical for large models that approach memory limits
- **Improved Scalability**: Multiple clients can download at different rates without blocking
- **Safetensors Format**: Secure and efficient tensor serialization without pickle vulnerabilities
- **No Migration Required**: Existing PyTorch jobs automatically benefit from this optimization

.. admonition:: Learn More

    For a complete user guide including configuration examples and how to tune or disable the feature, see :ref:`tensor_downloader`.
    
    For details on the underlying FOBS decomposer architecture, see :ref:`decomposer_for_large_object`.

Server-Side Memory Cleanup
~~~~~~~~~~~~~~~~~~~~~~~~~~

FLARE 2.7.2 adds automatic server-side memory management to address RSS (Resident Set Size — the actual physical memory used by a process) growth in long-running jobs:

- **Periodic garbage collection and heap trimming**: Automatically runs ``gc.collect()`` and ``malloc_trim()`` to return freed memory back to the OS, preventing unbounded RSS growth over many training rounds.
- **Environment variable tuning**: Guidance on ``MALLOC_ARENA_MAX`` settings to control glibc memory arena fragmentation for both server and client processes.
- **Platform-aware**: Memory cleanup adapts to the runtime platform (Linux/glibc, musl, macOS), with full heap trimming on Linux/glibc and safe fallbacks elsewhere.
- **Minimal overhead**: Cleanup takes 10-500ms per invocation — negligible compared to typical training round durations.

.. admonition:: Learn More

    For configuration details, platform compatibility, recommended settings, and API reference, see :doc:`/programming_guide/memory_management`.

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

- **Memory Management Guide**: New :doc:`/programming_guide/memory_management` covering server-side garbage collection, ``MALLOC_ARENA_MAX`` tuning, platform compatibility, and troubleshooting.

- **Tensor Downloader Guide**: Expanded :doc:`/programming_guide/tensor_downloader` with configuration examples, architecture details, and tuning guidance.

- **Hello Differential Privacy**: New :doc:`/hello-world/hello-dp/index` example and documentation.

- **Client-Controlled Workflows**: Expanded documentation for :doc:`/programming_guide/controllers/client_controlled_workflows`.

- **Job Recipe Guide**: Updated :doc:`/user_guide/data_scientist_guide/job_recipe` with dict model config and initial checkpoint examples.

Bug Fixes
~~~~~~~~~

- Fixed TLS corruption by replacing ``fork`` with ``posix_spawn`` for subprocess creation.
- Fixed potential data corruption issue in the Streamer component.
- Fixed Swarm Learning controller compatibility with tensor streaming.
- Fixed Swarm Learning controller bug.
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

This section provides guidance for migrating from 2.7.1 to 2.7.2.

initial_model → model
~~~~~~~~~~~~~~~~~~~~~

The ``initial_model`` parameter in all recipes has been renamed to ``model`` for clarity:

**Before:**

.. code-block:: python

    recipe = FedAvgRecipe(
        ...
        initial_model=SimpleNetwork(),
    )

**After:**

.. code-block:: python

    recipe = FedAvgRecipe(
        ...
        model=SimpleNetwork(),
    )

The ``model`` parameter now also accepts dict-based configuration:

.. code-block:: python

    recipe = FedAvgRecipe(
        ...
        model={"path": "my_module.MyModel", "args": {"hidden_size": 256}},
        initial_ckpt="pretrained.pt",
    )

PTFedAvgEarlyStopping → PTFedAvg
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``PTFedAvgEarlyStopping`` class has been merged into ``PTFedAvg`` with InTime aggregation support. A backward-compatible alias is provided:

**Before:**

.. code-block:: python

    from nvflare.app_opt.pt.fedavg_early_stopping import PTFedAvgEarlyStopping

    controller = PTFedAvgEarlyStopping(...)

**After:**

.. code-block:: python

    from nvflare.app_opt.pt.fedavg import PTFedAvg

    controller = PTFedAvg(...)

MONAI-FLARE Wheel
~~~~~~~~~~~~~~~~~

The separate ``nvflare-monai`` wheel package is deprecated. Use the Client API directly for MONAI integration.
See the updated examples in ``examples/advanced/monai/`` and the `MONAI Migration Guide <https://github.com/NVIDIA/NVFlare/blob/main/integration/monai/MIGRATION.md>`_.

Backward Compatibility Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Job Config API**: Existing ``FedJob``-based configurations continue to work alongside the new Recipe API.
- **Config-based Jobs**: JSON/YAML configuration-based jobs continue to work as before.
- **Executor/ModelLearner APIs**: Still functional but no longer the recommended pattern. Use Recipe API + Client API for new projects.

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


