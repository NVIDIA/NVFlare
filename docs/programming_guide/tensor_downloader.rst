.. _tensor_downloader:

#######################
FLARE Tensor Downloader
#######################

This guide explains the Tensor Downloader feature in NVIDIA FLARE, which provides memory-efficient
transfer of large PyTorch models in federated learning workflows.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

What is the Tensor Downloader?
------------------------------

The Tensor Downloader is a memory optimization feature that enables efficient transfer of large PyTorch
tensors (model parameters) between the FL server and clients. Instead of serializing entire models into
memory before transmission, it streams tensors incrementally, significantly reducing peak memory usage.

Why Do We Need It?
------------------

In traditional federated learning, when the server sends a global model to clients (or when clients
send updates back), the entire model must be:

1. **Serialized into memory** - Converting the model to bytes requires additional memory equal to or
   greater than the model size
2. **Held in memory during transmission** - The serialized bytes must remain in memory until
   transmission completes
3. **Multiplied for multiple recipients** - When sending to N clients simultaneously, memory pressure
   increases dramatically

For large language models (LLMs) and other large-scale models, this can cause:

- **Out-of-memory errors** when available RAM is insufficient
- **Severe performance degradation** when memory is saturated
- **System instability** affecting other processes

The Tensor Downloader solves these problems by using a **pull-based, incremental streaming** approach.

Key Benefits
------------

- **Reduced Memory Footprint**: 20-50% reduction in memory usage on both server and client sides
  (based on testing with 5GB models and 4 clients using FedAvg)

- **No Code Changes Required**: The optimization is built into PyTorch workflows and works
  automatically with existing training code

- **Scalable to Multiple Clients**: Each client downloads at its own pace without blocking others

- **Secure Serialization**: Uses the `safetensors` format which avoids pickle-based security
  vulnerabilities

- **Reliable Transfer**: Pull-based architecture handles heterogeneous network conditions gracefully

Limitations
-----------

- **PyTorch and NumPy Only**: The streaming download feature supports PyTorch tensors and NumPy arrays.
  TensorFlow models are not currently supported and will use traditional serialization.

- **Custom Tensor Types**: Custom tensor types or non-standard model formats are not directly supported.
  Convert your custom tensors to PyTorch tensors (``torch.Tensor``) or NumPy arrays (``numpy.ndarray``)
  to benefit from the streaming download feature.


How to Use It (User Perspective)
================================

For Standard Users
------------------

**Good news: You don't need to do anything!**

The Tensor Downloader is built into all PyTorch workflows in FLARE 2.7.2+. When you use:

- ``PTFedAvg`` controller
- ``PTFileModelPersistor``
- ``PTClientAPILauncherExecutor``
- ``PTInProcessClientAPIExecutor``
- Any PyTorch-based Recipe (``FedAvgRecipe`` from ``nvflare.app_opt.pt.recipes``)

The TensorDecomposer is automatically registered and handles tensor streaming transparently.

Client Memory Note for Large Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorDownloader reduces transfer-time memory pressure, and client-side parameter references are
also released after ``flare.send()`` when ``clear_cache=True`` (default). In CPython, tensors are
typically reclaimed as soon as their last reference is dropped.

For multi-GB payloads, avoid keeping extra references longer than needed:

.. code-block:: python

    import nvflare.client as flare

    flare.init()
    while flare.is_running():
        input_model = flare.receive()
        output_model = train(input_model)
        flare.send(output_model)  # clear_cache=True by default

        # Optional: release script-local references promptly.
        del input_model
        del output_model

``gc.collect()`` remains a supplemental safeguard for cyclic objects; it is not the primary
mechanism for releasing tensor memory in this flow.

Example: Using PyTorch FedAvg Recipe
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from nvflare.app_opt.pt.recipes import FedAvgRecipe
    from nvflare.recipe import SimEnv

    # TensorDownloader is automatically used - no configuration needed
    # Model can be class instance or dict config
    # For pre-trained weights: initial_ckpt="/server/path/to/pretrained.pt"
    recipe = FedAvgRecipe(
        name="my-fedavg-job",
        min_clients=2,
        num_rounds=10,
        model=MyLargeModel(),  # Even multi-GB models work efficiently
        train_script="client.py",
    )

    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

Example: Using PTFedAvg Controller Directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from nvflare import FedJob
    from nvflare.app_opt.pt.fedavg import PTFedAvg

    job = FedJob(name="pt-fedavg")

    # TensorDownloader is automatically enabled
    # Model can be class instance or dict config
    controller = PTFedAvg(
        num_clients=2,
        num_rounds=10,
        model=MyLargeModel(),
    )
    job.to(controller, "server")

Configuration
-------------

The Tensor Downloader behavior can be configured via chunk size settings in your job configuration files.

**Configuration Parameters:**

- ``tensor_download_chunk_size``: Chunk size for PyTorch tensor downloads (default: 2097152 = 2MB)
- ``np_download_chunk_size``: Chunk size for NumPy array downloads (default: 2097152 = 2MB)

Using Recipe API (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For users working with recipes, use the ``add_server_config()`` method:

.. code-block:: python

    from nvflare.recipe.fedavg import FedAvgRecipe

    recipe = FedAvgRecipe(
        name="my_job",
        num_rounds=10,
        min_clients=2,
        train_script="train.py",
    )

    # Configure chunk sizes and streaming timeout (server-side only)
    recipe.add_server_config({
        "np_download_chunk_size": 2097152,
        "tensor_download_chunk_size": 2097152,
        "streaming_per_request_timeout": 600
    })

Using Job API
^^^^^^^^^^^^^

For users working directly with the Job API:

.. code-block:: python

    from nvflare import FedJob

    job = FedJob(name="my_job")

    # Add config to server (these are server-side only settings)
    job.to_server({
        "np_download_chunk_size": 2097152,
        "tensor_download_chunk_size": 2097152,
        "streaming_per_request_timeout": 600
    })

Tuning for Large Models
^^^^^^^^^^^^^^^^^^^^^^^

For very large models (multiple GB), you may want to tune chunk sizes for optimal performance.
Larger chunks mean fewer network requests but higher per-chunk memory usage. Smaller chunks
reduce memory but increase network overhead.

**Example config_fed_server.conf with chunk size tuning:**

.. code-block::

    format_version = 2
    
    # Chunk sizes for streaming large models (2MB default)
    np_download_chunk_size = 2097152
    tensor_download_chunk_size = 2097152
    streaming_per_request_timeout = 600
    
    task_data_filters = []
    task_result_filters = []
    
    components = [
      {
        id = "json_generator"
        path = "nvflare.app_common.widgets.validation_json_generator.ValidationJsonGenerator"
        args {}
      }
    ]
    
    workflows = [
      {
        id = "swarm_controller"
        path = "nvflare.app_common.ccwf.SwarmServerController"
        args {
          num_rounds = 3
          # Increased timeouts to accommodate large LLM payload init/broadcast
          start_task_timeout = 300
          progress_timeout = 7200
        }
      }
      {
        id = "cross_site_eval"
        path = "nvflare.app_common.ccwf.CrossSiteEvalServerController"
        args {
          eval_task_timeout = 1200
        }
      }
    ]

Disabling the Tensor Downloader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer to disable the streaming download feature and use traditional serialization instead,
set the chunk sizes to zero.

**Using Recipe API:**

.. code-block:: python

    # Disable streaming (server-side setting)
    recipe.add_server_config({
        "np_download_chunk_size": 0,
        "tensor_download_chunk_size": 0
    })

**Using Job API:**

.. code-block:: python

    job.to_server({"np_download_chunk_size": 0, "tensor_download_chunk_size": 0})

**Using config files directly:**

.. code-block::

    format_version = 2
    
    # Set to 0 to disable streaming download (use native serialization)
    np_download_chunk_size = 0
    tensor_download_chunk_size = 0
    
    task_data_filters = []
    task_result_filters = []
    
    # ... rest of configuration


How It Works (Advanced Users)
=============================

This section explains the internal architecture for developers who want to understand or
extend the Tensor Downloader functionality.

Architecture Overview
---------------------

The Tensor Downloader consists of several components:

1. **TensorDecomposer**: A FOBS decomposer that handles PyTorch tensor serialization
2. **TensorDownloadable**: Represents a collection of tensors ready for incremental download
3. **TensorConsumer**: Processes downloaded tensor chunks on the receiving side
4. **Download Service**: Manages the pull-based download protocol

Pull-Based vs Push-Based Transfer
---------------------------------

Traditional (Push-Based):

.. code-block:: text

    Server                           Client
      |                                 |
      |  [Serialize entire model]       |
      |  [Hold in memory]               |
      |-------- Full Model ------------>|
      |                                 |  [Deserialize]

Tensor Downloader (Pull-Based):

.. code-block:: text

    Server                           Client
      |                                 |
      |  [Prepare reference ID]         |
      |-------- Reference ID ---------->|
      |                                 |
      |<------- Request chunk 1 --------|
      |  [Serialize chunk 1 only]       |
      |-------- Chunk 1 --------------->|
      |                                 |
      |<------- Request chunk 2 --------|
      |  [Serialize chunk 2 only]       |
      |-------- Chunk 2 --------------->|
      |           ...                   |
      |                                 |  [Reassemble model]

The Serialization Flow
----------------------

1. **Registration**: When a PyTorch component initializes, it registers the ``TensorDecomposer``
   with FOBS:

   .. code-block:: python

       from nvflare.app_opt.pt.decomposers import TensorDecomposer
       from nvflare.fuel.utils import fobs

       fobs.register(TensorDecomposer)

2. **Tensor Collection**: During serialization, FOBS collects all tensors in the payload into
   a dictionary.

3. **Downloadable Creation**: The tensors are wrapped in a ``TensorDownloadable`` object:

   .. code-block:: python

       class TensorDownloadable(CacheableObject):
           def __init__(self, tensors: dict[str, torch.Tensor], max_chunk_size: int):
               self.keys = list(tensors.keys())
               super().__init__(tensors, max_chunk_size)

           def produce_item(self, index: int) -> bytes:
               key = self.keys[index]
               tensor_to_send = {key: self.base_obj[key]}
               return save_tensors(tensor_to_send)  # safetensors format

4. **Reference ID Generation**: A unique reference ID (RID) is generated and sent to recipients
   instead of the actual tensors.

5. **Incremental Download**: Each recipient requests tensors one at a time using the RID.

The TensorDecomposer
--------------------

The ``TensorDecomposer`` extends ``ViaDownloaderDecomposer`` and provides:

.. code-block:: python

    class TensorDecomposer(ViaDownloaderDecomposer):

        def supported_type(self):
            return torch.Tensor

        def to_downloadable(self, items: dict, max_chunk_size: int, fobs_ctx: dict):
            return TensorDownloadable(items, max_chunk_size)

        def download(self, from_fqcn, ref_id, per_request_timeout, cell, ...):
            return download_tensors(from_fqcn, ref_id, per_request_timeout, cell, ...)

        def native_decompose(self, target: torch.Tensor, manager=None) -> bytes:
            # Fallback: serialize single tensor using safetensors
            return save({"t": target})

        def native_recompose(self, data: bytes, manager=None) -> torch.Tensor:
            # Fallback: deserialize single tensor
            return load(data).get("t")

Using the Low-Level API
-----------------------

For advanced use cases, you can use the tensor download API directly:

.. code-block:: python

    from nvflare.app_opt.pt.tensor_downloader import add_tensors, download_tensors

    # Server side: Register tensors for download
    ref_id = add_tensors(
        downloader=downloader,
        tensors=model.state_dict(),
        max_chunk_size=2 * 1024 * 1024,  # 2MB chunks
    )

    # Send ref_id to clients via your preferred mechanism
    # ...

    # Client side: Download tensors incrementally
    status, state_dict = download_tensors(
        from_fqcn=server_fqcn,
        ref_id=ref_id,
        per_request_timeout=30.0,
        cell=cell,
        secure=False,
        optional=False,
        abort_signal=abort_signal,
    )

    # Load into model
    model.load_state_dict(state_dict)

See Also
========

- :ref:`decomposer_for_large_object` - Details on the FOBS decomposer system and file-based decomposers
- :ref:`file_streaming` - File streaming for other large data types
- :ref:`swarm_learning_large_models` - Parameter tuning for large model workflows
