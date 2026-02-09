.. _memory_management:

###################
Memory Management
###################

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

Platform Compatibility
======================

Not all memory management features work on all platforms. The table below summarizes compatibility:

+---------------------------+-------------+-------------+-------------+
| Feature                   | Linux/glibc | Linux/musl  | macOS       |
+===========================+=============+=============+=============+
| ``gc.collect()``          | ✓           | ✓           | ✓           |
+---------------------------+-------------+-------------+-------------+
| ``MALLOC_ARENA_MAX``      | ✓           | ✗           | ✗           |
+---------------------------+-------------+-------------+-------------+
| ``malloc_trim()``         | ✓           | ✗           | ✗           |
+---------------------------+-------------+-------------+-------------+
| ``torch.cuda.empty_cache``| ✓           | ✓           | ✓           |
+---------------------------+-------------+-------------+-------------+

**Notes:**

- **Linux/glibc**: Standard Linux distributions (Ubuntu, RHEL, Debian, etc.)
- **Linux/musl**: Alpine Linux and other musl-based distributions
- **macOS**: ``malloc_trim()`` is silently skipped (safe no-op)

.. warning::

   For maximum memory efficiency, use Linux with glibc. Alpine Linux (musl) and
   macOS/Windows will still benefit from ``gc.collect()`` but cannot release
   fragmented heap memory back to the OS.

Environment Variables
=====================

Set these environment variables before starting NVFlare processes:

Client (Training Nodes)
-----------------------

.. code-block:: bash

    export MALLOC_ARENA_MAX=2

**Why:** Clients typically have limited CPU memory. Setting ``MALLOC_ARENA_MAX=2``
prevents arena explosion and reduces memory fragmentation.

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

Performance Impact
------------------

Memory cleanup has minimal overhead in typical federated learning workloads:

+---------------------------+------------------+--------------------------------+
| Operation                 | Typical Duration | Notes                          |
+===========================+==================+================================+
| ``gc.collect()``          | 10-500 ms        | Depends on Python object count |
+---------------------------+------------------+--------------------------------+
| ``malloc_trim()``         | < 1 ms           | Very fast (page table ops)     |
+---------------------------+------------------+--------------------------------+

**Overhead analysis:**

- **Training round duration**: Typically 30 seconds to 10+ minutes
- **Cleanup duration**: 10-500 ms total
- **Overhead per round**: Usually < 1%

**With** ``server_memory_gc_rounds=5``:

- Cleanup runs once every 5 rounds
- Total overhead: < 0.2% of training time

**Recommendation**: The default ``server_memory_gc_rounds=5`` provides good memory
management with negligible performance impact. Only disable (``=0``) if you've
measured and confirmed RSS is stable without cleanup.

Recommended Settings
====================

+--------+-------------------------------+----------------------+
| Role   | ``server_memory_gc_rounds``   | ``MALLOC_ARENA_MAX``   |
+========+===============================+======================+
| Server | 5                             | 4                    |
+--------+-------------------------------+----------------------+

API Reference
=============

cleanup_memory
--------------

.. code-block:: python

    from nvflare.fuel.utils.memory_utils import cleanup_memory

    cleanup_memory(torch_cuda_empty_cache=True)

**Signature:** ``cleanup_memory(torch_cuda_empty_cache: bool = False) -> None``

Performs memory cleanup:

1. Runs ``gc.collect()``
2. Calls ``malloc_trim(0)`` (Linux/glibc only, safe no-op elsewhere)
3. Optionally calls ``torch.cuda.empty_cache()``

try_malloc_trim
---------------

.. code-block:: python

    from nvflare.fuel.utils.memory_utils import try_malloc_trim

    result = try_malloc_trim()

**Signature:** ``try_malloc_trim() -> Optional[int]``

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

OOM Errors
----------

1. Reduce batch size
2. Enable memory cleanup every round (``server_memory_gc_rounds=1``)
3. Check for memory leaks in training code

