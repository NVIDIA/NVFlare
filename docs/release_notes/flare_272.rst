**************************
What's New in FLARE v2.7.2
**************************

NVIDIA FLARE 2.7.2 is a feature release that builds on the Job Recipe API introduced in 2.7.0, bringing it to production readiness.
This release also introduces significant memory management improvements with the new Tensor-based Downloader for efficient large model handling.

Job Recipe API - Production Ready
=================================

.. sidebar::

    **Recipe-based Example**

    .. code-block:: python

            from nvflare.app_opt.pt.recipes import FedAvgRecipe
            from nvflare.recipe import SimEnv

            recipe = FedAvgRecipe(
                name="hello-pt",
                min_clients=2,
                num_rounds=5,
                initial_model=SimpleNetwork(),
                train_script="client.py",
            )
            env = SimEnv(num_clients=2)
            run = recipe.execute(env)

The Job Recipe API, introduced as a technical preview in 2.7.0, is now production-ready with comprehensive coverage across all major examples.
Almost all examples in the NVFlare repository have been converted to use Job Recipes, demonstrating the simplicity and power of this approach.

Key Highlights
--------------

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
==============================================

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
------------

- **Zero Code Changes Required**: Your existing PyTorch FL jobs benefit from memory optimization without any modification.

- **Incremental Tensor Serialization**: Instead of serializing all model parameters at once, tensors are serialized individually using safetensors format, significantly reducing peak memory consumption.

- **Pull-based Architecture**: Unlike push-based streaming, each recipient pulls data at its own pace, making it more reliable for heterogeneous network conditions.

Performance Results
-------------------

Based on our internal testing with a 5GB model and 4 clients using FedAvg, we observed **20% to 50% memory usage reduction** on both server and client sides.

.. note::

    Your results may vary depending on model size, number of clients, network conditions, and different FL algorithms and workflows.

How It Works
------------

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
-------------------------

- **Reduced Memory Footprint**: 20-50% reduction critical for large models that approach memory limits
- **Improved Scalability**: Multiple clients can download at different rates without blocking
- **Safetensors Format**: Secure and efficient tensor serialization without pickle vulnerabilities
- **No Migration Required**: Existing PyTorch jobs automatically benefit from this optimization

.. admonition:: Learn More

    For a complete user guide including configuration examples and how to tune or disable the feature, see :ref:`tensor_downloader`.
    
    For details on the underlying FOBS decomposer architecture, see :ref:`decomposer_for_large_object`.

Additional Improvements
=======================

API Refinements
---------------

- **FedAvg Controller Enhancements**: Improved `load_model()` and `save_model()` methods that properly override base class methods for framework-specific serialization (PyTorch, TensorFlow).

- **Simplified Task Management**: The `task_name` parameter is now streamlined (renamed from `task_to_optimize`), with "train" as the default for standard federated learning workflows.

- **Enhanced Type Checking**: Improved validation of `initial_model` types with clear error messages guiding users to appropriate framework-specific recipes.

- **InTime Aggregation by Default**: All FedAvg recipes now use memory-efficient InTime aggregation, processing client results as they arrive rather than waiting for all results.

Example Consolidation
---------------------

To provide a cleaner and more focused learning experience, we have consolidated and streamlined the examples:

- **Removed Deprecated Examples**: Examples using old APIs (Executor-based, ModelLearner-based patterns) have been removed. All examples now use the modern Recipe API or Client API.

- **Unified Example Structure**: Each example now follows a consistent structure with a ``job.py`` entry point that uses the Recipe API, making it easier to understand and adapt.

- **Reduced Redundancy**: Duplicate examples demonstrating the same concepts with different APIs have been consolidated into single, canonical examples.

- **Focus on Best Practices**: Remaining examples showcase the recommended patterns for building federated learning applications with FLARE.

MONAI Integration
-----------------

- **MONAI-FLARE Wheel Deprecated**: The separate ``nvflare-monai`` wheel package is now deprecated. MONAI integration is now achieved directly through the Client API, simplifying the integration and reducing dependency management overhead. For further information, see the `MONAI Migration Guide <https://github.com/NVIDIA/NVFlare/blob/main/integration/monai/MIGRATION.md>`_.

- **Updated MONAI Examples**: All MONAI examples have been updated to use the Client API pattern, making it easier to integrate MONAI training workflows with FLARE without requiring additional packages.

Documentation
-------------

- **Comprehensive Recipe Documentation**: New :ref:`available_recipes` guide with code examples and links to working examples for all available recipes.

- **Updated Examples**: All example READMEs updated to reflect the Recipe-based approach.

Bug Fixes
---------

- Fixed handling of absolute paths for `train_script` in production environments where scripts may be pre-installed.
- Improved environment-specific validation for script resources (SimEnv/PocEnv require local files, ProdEnv allows pre-installed scripts).
- Fixed aggregation state management when using custom aggregators.
- Removed dead code in framework-specific recipes.

Migration Guide
===============

This section provides guidance for migrating from previous FLARE versions to 2.7.2.

PTFedAvgEarlyStopping → PTFedAvg
--------------------------------

The ``PTFedAvgEarlyStopping`` class has been renamed to ``PTFedAvg``. A backward-compatible alias is provided, but you should update your code:

**Before (deprecated):**

.. code-block:: python

    from nvflare.app_opt.pt.fedavg_early_stopping import PTFedAvgEarlyStopping

    controller = PTFedAvgEarlyStopping(
        num_clients=2,
        num_rounds=5,
        ...
    )

**After:**

.. code-block:: python

    from nvflare.app_opt.pt.fedavg import PTFedAvg

    controller = PTFedAvg(
        num_clients=2,
        num_rounds=5,
        ...
    )

task_to_optimize → task_name
----------------------------

The ``task_to_optimize`` parameter has been renamed to ``task_name`` for clarity:

**Before (deprecated):**

.. code-block:: python

    controller = FedAvg(
        ...
        task_to_optimize="train",
    )

**After:**

.. code-block:: python

    controller = FedAvg(
        ...
        task_name="train",
    )

Note: In Recipe API, the task name defaults to "train" and typically does not need to be specified.

MONAI-FLARE Wheel Migration
---------------------------

The separate ``nvflare-monai`` wheel is deprecated. Migrate to using the Client API directly:

**Before (deprecated):**

.. code-block:: bash

    pip install nvflare-monai

.. code-block:: python

    from nvflare.app_opt.monai import MonaiTrainer

    # Using MONAI-specific executor
    executor = MonaiTrainer(...)

**After:**

.. code-block:: python

    import nvflare.client as flare
    from monai.engines import SupervisedTrainer

    # Initialize FLARE client
    flare.init()

    # Use standard MONAI trainer with Client API
    while flare.is_running():
        input_model = flare.receive()
        # ... train with MONAI ...
        flare.send(output_model)

See the updated MONAI examples in ``examples/advanced/monai/`` for complete migration patterns.

Migrating from Old Example Patterns
-----------------------------------

If you have code based on removed examples that used Executor or ModelLearner patterns, migrate to the Recipe API:

**Before (Executor-based, deprecated):**

.. code-block:: python

    from nvflare.apis.executor import Executor

    class MyTrainer(Executor):
        def execute(self, task_name, shareable, fl_ctx, abort_signal):
            # Training logic
            ...

**After (Recipe + Client API):**

``job.py``:

.. code-block:: python

    from nvflare.app_opt.pt.recipes import FedAvgRecipe
    from nvflare.recipe import SimEnv

    recipe = FedAvgRecipe(
        name="my-job",
        min_clients=2,
        num_rounds=5,
        initial_model=MyModel(),
        train_script="client.py",
    )
    env = SimEnv(num_clients=2)
    recipe.execute(env)

``client.py``:

.. code-block:: python

    import nvflare.client as flare

    flare.init()
    while flare.is_running():
        input_model = flare.receive()
        # Training logic
        ...
        flare.send(output_model)

Backward Compatibility Notes
----------------------------

- **Job Config API**: Existing ``FedJob``-based configurations continue to work alongside the new Recipe API.
- **Deprecated APIs**: All deprecated APIs from 2.7.0 remain functional with deprecation warnings but will be removed in a future release.
- **Config-based Jobs**: JSON/YAML configuration-based jobs continue to work as before.

Getting Started
===============

The easiest way to get started with FLARE 2.7.2 is through the Hello World examples:

.. code-block:: bash

    # Run the PyTorch FedAvg example
    cd examples/hello-world/hello-pt
    python job.py

    # Run the TensorFlow FedAvg example
    cd examples/hello-world/hello-tf
    python job.py

For more examples, see:

- `Hello World Examples <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world>`_
- `CIFAR-10 Examples <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/cifar10>`_
- `XGBoost Examples <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/xgboost>`_
- `Job Recipe Tutorial <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/job_recipe.ipynb>`_


