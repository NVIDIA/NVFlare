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
- ``ClientAPIExecutor`` with ``params_exchange_format="pytorch"``
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

The Tensor Downloader behavior can be configured via chunk size and timeout settings in your job
configuration files.

**Configuration Parameters:**

- ``tensor_download_chunk_size``: Chunk size for PyTorch tensor downloads (default: 2097152 = 2MB)
- ``np_download_chunk_size``: Chunk size for NumPy array downloads (default: 2097152 = 2MB)
- ``tensor_streaming_per_request_timeout``: Timeout in seconds for each PyTorch tensor download request
  (default: 600)
- ``np_streaming_per_request_timeout``: Timeout in seconds for each NumPy download request (default: 600)
- ``tensor_min_download_timeout`` and ``np_min_download_timeout``: Minimum lifetime in seconds of a
  download transaction

The settings are prefixed by payload type. The unprefixed ``streaming_per_request_timeout`` is not read by
the Tensor Downloader.

**Which process reads which setting.** A transfer has a sender and a receiver, and each setting is read by
only one of them:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Setting
     - Read by
   * - ``tensor_download_chunk_size``, ``np_download_chunk_size``
     - The sender, which serializes the payload and creates the download. Setting a chunk size to 0 disables
       the download for payloads that process sends.
   * - ``tensor_min_download_timeout``, ``np_min_download_timeout``
     - The sender, which creates the download transaction.
   * - ``tensor_streaming_per_request_timeout``, ``np_streaming_per_request_timeout``
     - The receiver, which issues the download requests.

The server sends the global model to clients, and clients send updates to the server, so both directions
occur in a normal job. To apply a setting in both directions, set it on the server and on the clients:
``add_server_config()`` and ``add_client_config()`` for recipes, or ``to_server()`` and ``to_clients()``
for the Job API. A setting placed only on the server affects transfers the server sends (chunk sizes) or
receives (request timeouts), not the client side of the exchange.

``add_client_config()`` and ``to_clients()`` configure the client job process. In ``external_process`` mode the
training script runs in a separate trainer process that does not load the job's application configuration, so
these settings do not reach it. That process still sends results and pulls task data, so it keeps the default
chunk sizes and request timeouts unless you set them in its environment. A setting is read from the
environment variable ``NVFLARE_`` followed by the setting name in upper case, for example
``NVFLARE_TENSOR_DOWNLOAD_CHUNK_SIZE`` or ``NVFLARE_TENSOR_STREAMING_PER_REQUEST_TIMEOUT``. Set these in the
environment the trainer is launched with: either export the variable before starting the client, since the
trainer inherits its environment, or use the launch ``command``, for example
``command="env NVFLARE_TENSOR_DOWNLOAD_CHUNK_SIZE=0 python3 -u"``. The command is run without a shell, so a bare
``NAME=value`` prefix does not work; use ``env``. When the training script runs in the client job process
(the default in-process mode), the job configuration applies.

Using Recipe API (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For users working with recipes, use ``add_server_config()`` and ``add_client_config()``:

.. code-block:: python

    from nvflare.recipe.fedavg import FedAvgRecipe

    recipe = FedAvgRecipe(
        name="my_job",
        num_rounds=10,
        min_clients=2,
        train_script="train.py",
    )

    config = {
        "np_download_chunk_size": 2097152,
        "tensor_download_chunk_size": 2097152,
        "np_streaming_per_request_timeout": 600,
        "tensor_streaming_per_request_timeout": 600,
    }

    # Server to clients: the server chunks, the clients time requests.
    recipe.add_server_config(config)
    # Clients to server: the clients chunk, the server times requests.
    recipe.add_client_config(config)

Using Job API
^^^^^^^^^^^^^

For users working directly with the Job API:

.. code-block:: python

    from nvflare import FedJob

    job = FedJob(name="my_job")

    config = {
        "np_download_chunk_size": 2097152,
        "tensor_download_chunk_size": 2097152,
        "np_streaming_per_request_timeout": 600,
        "tensor_streaming_per_request_timeout": 600,
    }

    # Apply to the server and to all clients to cover both transfer directions
    job.to_server(config)
    job.to_clients(config)

Tuning for Large Models
^^^^^^^^^^^^^^^^^^^^^^^

For very large models (multiple GB), you may want to tune chunk sizes for optimal performance.
Larger chunks mean fewer network requests but higher per-chunk memory usage. Smaller chunks
reduce memory but increase network overhead.

When tensor streaming is used with ``ClientAPIExecutor``, keep
``tensor_min_download_timeout`` (or ``np_min_download_timeout`` for NumPy)
aligned with the configured streaming per-request timeout. In
``external_process`` mode, an active task download extends the
``task_wait_timeout`` wait while progress remains live. Attach instead applies
an absolute ``task_wait_timeout`` deadline to task download and trainer
acceptance, so configure it for the complete delivery path.
``result_wait_timeout`` bounds result publication; subsequent payload streaming
uses the shared transfer idle policy. See :ref:`timeout_troubleshooting` and
:doc:`/programming_guide/timeouts`.

**Example config_fed_server.conf with chunk size tuning** (the same settings belong in
``config_fed_client`` to cover transfers that the clients send and receive):

.. code-block::

    format_version = 2
    
    # Chunk sizes for streaming large models (2MB default), read by the sending process
    np_download_chunk_size = 2097152
    tensor_download_chunk_size = 2097152
    # Per-request timeouts, read by the receiving process
    np_streaming_per_request_timeout = 600
    tensor_streaming_per_request_timeout = 600
    
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

    # Disable streaming: chunk sizes are read by the sender, so set them on the
    # server (server to clients) and on the clients (clients to server)
    config = {"np_download_chunk_size": 0, "tensor_download_chunk_size": 0}
    recipe.add_server_config(config)
    recipe.add_client_config(config)

In ``external_process`` mode also set ``NVFLARE_NP_DOWNLOAD_CHUNK_SIZE=0`` and
``NVFLARE_TENSOR_DOWNLOAD_CHUNK_SIZE=0`` in the trainer's environment, because the job configuration does not
reach the trainer process (see above). Otherwise downloads stay enabled for the results the trainer sends.

**Using Job API:**

.. code-block:: python

    config = {"np_download_chunk_size": 0, "tensor_download_chunk_size": 0}
    job.to_server(config)
    job.to_clients(config)

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
- :ref:`timeout_troubleshooting` - Timeout tuning for large Client API subprocess jobs
