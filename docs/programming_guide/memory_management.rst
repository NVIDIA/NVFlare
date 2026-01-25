.. _memory_management:

#############################
Memory Management Best Practices
#############################

This guide describes memory management techniques for long-running federated learning jobs
using Python, PyTorch, and glibc.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

Federated learning jobs can run for hours or days. Without proper memory management,
RSS (Resident Set Size) can grow continuously due to:

- Python garbage collection delays
- glibc memory arena fragmentation
- PyTorch CUDA cache retention

NVFlare provides utilities and configuration options to manage memory effectively.

Environment Variables
=====================

Set these environment variables before starting NVFlare processes:

Client (Training Nodes)
-----------------------

.. code-block:: bash

    export MALLOC_ARENA_MAX=2

**Why:** Clients are GPU-constrained with limited CPU memory. Setting ``MALLOC_ARENA_MAX=2``
prevents arena explosion and reduces fragmentation.

Server (Aggregation Node)
-------------------------

.. code-block:: bash

    export MALLOC_ARENA_MAX=4

**Why:** Servers are CPU memory heavy (4-7× model size) with multi-threaded networking.
``MALLOC_ARENA_MAX=4`` balances throughput vs memory. Use ``8`` for high parallelism.

Server-Side Memory Cleanup
==========================

The FedAvg controller supports automatic memory cleanup via the ``server_memory_gc_rounds`` parameter.

Configuration
-------------

.. code-block:: python

    from nvflare.recipe.fedavg import FedAvgRecipe

    recipe = FedAvgRecipe(
        name="my_job",
        min_clients=4,
        num_rounds=100,
        train_script="client.py",
        server_memory_gc_rounds=5,  # Cleanup every 5 rounds
    )

**Values:**

- ``0`` = Disabled (default for BaseFedAvg-based controllers)
- ``1`` = Cleanup every round (default for legacy controllers like ScatterAndGather)
- ``5`` = Cleanup every 5 rounds (recommended for server)

What It Does
------------

When enabled, at the end of every N rounds:

1. Runs Python garbage collection (``gc.collect()``)
2. Returns free heap pages to OS (``malloc_trim()``, Linux/glibc only)

Client-Side Memory Cleanup
==========================

For client-side cleanup, call the utility function in your training script.

Using the Utility Function
--------------------------

.. code-block:: python

    import nvflare.client as flare
    from nvflare.fuel.utils.memory_utils import cleanup_memory

    flare.init()

    while flare.is_running():
        input_model = flare.receive()

        # ... training code ...

        flare.send(output_model)

        # Memory cleanup after each round
        cleanup_memory(cuda_empty_cache=True)

**Parameters:**

- ``cuda_empty_cache=True``: Also call ``torch.cuda.empty_cache()`` (for GPU clients)
- ``cuda_empty_cache=False``: Skip CUDA cache clearing (for CPU-only clients)

Recommended Settings
====================

+--------+-----------------------------+----------------------+--------------------+
| Role   | ``server_memory_gc_rounds`` | ``cuda_empty_cache`` | ``MALLOC_ARENA_MAX`` |
+========+=============================+======================+====================+
| Server | 5                           | N/A                  | 4                  |
+--------+-----------------------------+----------------------+--------------------+
| Client | N/A (future)                | true                 | 2                  |
+--------+-----------------------------+----------------------+--------------------+

API Reference
=============

cleanup_memory
--------------

.. code-block:: python

    from nvflare.fuel.utils.memory_utils import cleanup_memory

    cleanup_memory(cuda_empty_cache: bool = False) -> None

Performs memory cleanup:

1. Runs ``gc.collect()``
2. Calls ``malloc_trim(0)`` (Linux/glibc only, safe no-op elsewhere)
3. Optionally calls ``torch.cuda.empty_cache()``

try_malloc_trim
---------------

.. code-block:: python

    from nvflare.fuel.utils.memory_utils import try_malloc_trim

    result = try_malloc_trim() -> Optional[int]

Low-level function to return free heap pages to OS.

**Returns:**

- ``1`` if memory was released
- ``0`` if no memory to release
- ``None`` if not available (non-Linux or non-glibc)

Troubleshooting
===============

High RSS on Server
------------------

1. Check ``MALLOC_ARENA_MAX`` is set
2. Enable ``server_memory_gc_rounds=5``
3. Monitor with ``top`` or ``htop``

High RSS on Client
------------------

1. Check ``MALLOC_ARENA_MAX=2`` is set
2. Add ``cleanup_memory(cuda_empty_cache=True)`` after ``flare.send()``
3. Ensure GPU memory is released before next round

OOM Errors
----------

1. Reduce batch size
2. Enable memory cleanup every round (``server_memory_gc_rounds=1``)
3. Check for memory leaks in training code

