.. _memory_management:

#############################
Memory Management Best Practices
#############################

This guide describes memory management techniques for long-running federated learning jobs
using Python, PyTorch, and glibc/jemalloc.

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

NVFlare provides utilities and configuration options to manage memory effectively on
both server and client sides. The framework automatically detects the memory allocator
in use (glibc or jemalloc) and adapts its cleanup strategy accordingly.

Allocator Support
=================

NVFlare supports two memory allocators:

**glibc (default on most Linux)**
    Uses ``malloc_trim()`` to release free heap pages to the OS.
    Requires ``MALLOC_ARENA_MAX`` for optimal memory behavior.

**jemalloc (recommended for PyTorch)**
    Uses auto-decay for memory management. Configure via ``MALLOC_CONF``.
    No ``malloc_trim()`` calls needed (jemalloc handles this automatically).

NVFlare automatically detects which allocator is in use at runtime.

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

Client-Side Memory Cleanup
==========================

The FedAvg recipe and ScriptRunner support automatic memory cleanup on clients via
``client_memory_gc_rounds`` and ``torch_cuda_empty_cache`` parameters.

Configuration
-------------

.. code-block:: python

    from nvflare.recipe.fedavg import FedAvgRecipe

    recipe = FedAvgRecipe(
        name="my_job",
        min_clients=4,
        num_rounds=100,
        train_script="client.py",
        
        # Server-side cleanup
        server_memory_gc_rounds=5,
        
        # Client-side cleanup
        client_memory_gc_rounds=1,   # Cleanup every round
        torch_cuda_empty_cache=True, # Clear GPU cache
    )

**Parameters:**

- ``client_memory_gc_rounds``: Run cleanup every N rounds on client (0 = disabled)
- ``torch_cuda_empty_cache``: If True, call ``torch.cuda.empty_cache()`` on cleanup

What It Does
------------

When enabled, after each ``flare.send()`` on the client:

1. Runs Python garbage collection (``gc.collect()``)
2. For glibc: Returns free heap pages to OS (``malloc_trim()``)
3. For jemalloc: Relies on auto-decay (no manual action needed)
4. Optionally clears PyTorch CUDA cache

**Note:** The cleanup is transparent to the user's training script. No code changes
are required in ``train.py``.

External Process Support
------------------------

For external process execution (``launch_external_process=True``), memory settings
are passed via environment variables:

- ``NVFLARE_CLIENT_MEMORY_GC_ROUNDS``: Cleanup interval
- ``NVFLARE_TORCH_CUDA_EMPTY_CACHE``: GPU cache cleanup (``true``/``false``)

Performance Impact
==================

.. note::

   The performance numbers below are **estimates** based on typical workloads.
   Actual impact varies depending on model size, training complexity, hardware,
   and workflow configuration. We recommend profiling your specific use case
   to determine optimal settings.

Memory Cleanup Overhead
-----------------------

The ``cleanup_memory()`` function has minimal overhead:

+--------------------------------+------------------+--------------------------------+
| Operation                      | Typical Duration | Notes                          |
+================================+==================+================================+
| ``gc.collect()``               | 10-500 ms        | Depends on Python object count |
+--------------------------------+------------------+--------------------------------+
| ``malloc_trim()`` (glibc)      | < 1 ms           | Very fast (page table ops)     |
+--------------------------------+------------------+--------------------------------+
| ``torch.cuda.empty_cache()``   | < 50 ms          | Synchronizes CUDA stream       |
+--------------------------------+------------------+--------------------------------+

For GPU-based deep learning workloads where each training round takes seconds to minutes,
the cleanup overhead is negligible compared to the actual training time.

Memory Savings
--------------

Memory reduction depends on model size and training patterns:

- **Small models (< 1B params)**: Modest savings, may not be necessary
- **Medium models (1-10B params)**: Noticeable reduction, prevents gradual growth
- **Large models (70B+ params)**: Critical for preventing OOM, enables longer training runs

Allocator Configuration Impact
------------------------------

For optimal memory behavior, consider node-level allocator settings:

+---------------------------+-----------------------------------------------------+
| Setting                   | Impact                                              |
+===========================+=====================================================+
| ``MALLOC_ARENA_MAX=4``    | Limits glibc memory arenas, reduces fragmentation  |
+---------------------------+-----------------------------------------------------+
| ``MALLOC_ARENA_MAX=2``    | More aggressive, slightly more lock contention     |
+---------------------------+-----------------------------------------------------+
| jemalloc + ``MALLOC_CONF``| Automatic background cleanup via decay mechanism   |
+---------------------------+-----------------------------------------------------+

These settings can be configured in ``start.sh`` or container environment variables.

Recommended Settings
====================

Default Settings by Role
------------------------

+--------+-----------------------------+-----------------------------+----------------------+----------------------+
| Role   | ``server_memory_gc_rounds`` | ``client_memory_gc_rounds`` | ``MALLOC_ARENA_MAX`` | ``cuda_empty_cache`` |
+========+=============================+=============================+======================+======================+
| Server | 5                           | N/A                         | 4                    | N/A                  |
+--------+-----------------------------+-----------------------------+----------------------+----------------------+
| Client | N/A                         | 1                           | 2                    | True (for GPU)       |
+--------+-----------------------------+-----------------------------+----------------------+----------------------+

Best Practices by Scenario
--------------------------

+-------------------------------+-----------------------------+-----------------------------+----------------------+----------------------------------+
| Scenario                      | ``server_memory_gc_rounds`` | ``client_memory_gc_rounds`` | ``cuda_empty_cache`` | Notes                            |
+===============================+=============================+=============================+======================+==================================+
| Quick experiments             | 0 (disabled)                | 0 (disabled)                | False                | No overhead, short runs          |
+-------------------------------+-----------------------------+-----------------------------+----------------------+----------------------------------+
| Standard training             | 5                           | 5-10                        | False                | Balance cleanup and overhead     |
+-------------------------------+-----------------------------+-----------------------------+----------------------+----------------------------------+
| Long training (100+ rounds)   | 5                           | 5                           | True                 | Prevent memory growth            |
+-------------------------------+-----------------------------+-----------------------------+----------------------+----------------------------------+
| Large models (10B+ params)    | 1-3                         | 1-3                         | True                 | Aggressive cleanup               |
+-------------------------------+-----------------------------+-----------------------------+----------------------+----------------------------------+
| Memory-constrained edge       | 5                           | 1                           | True                 | Maximum memory efficiency        |
+-------------------------------+-----------------------------+-----------------------------+----------------------+----------------------------------+
| Swarm Learning                | N/A                         | 1-3                         | True                 | Clients also perform aggregation |
+-------------------------------+-----------------------------+-----------------------------+----------------------+----------------------------------+

Configuration Guidelines
------------------------

1. **Start with defaults** (``0``, disabled) for initial development and debugging

2. **Enable for production** with ``gc_rounds=5`` as a reasonable starting point

3. **Monitor memory usage** during pilot runs to determine if more aggressive settings are needed

4. **GPU memory**: Enable ``torch_cuda_empty_cache=True`` when:

   - Running multiple jobs on the same GPU
   - Model size is close to GPU memory limit
   - Experiencing CUDA OOM errors

5. **Symmetric configuration**: For balanced memory management, consider matching server and client settings:

   .. code-block:: python

       FedAvg(
           num_rounds=100,
           server_memory_gc_rounds=5,   # Server-side
           client_memory_gc_rounds=5,   # Client-side
           torch_cuda_empty_cache=True,
       )

6. **Swarm Learning**: Use more aggressive client settings since clients perform both training AND aggregation

Using jemalloc
==============

For PyTorch workloads, jemalloc is recommended over glibc malloc. NVFlare's startup
scripts automatically detect and use jemalloc if available.

Startup Script
--------------

The generated ``sub_start.sh`` script includes jemalloc auto-detection:

.. code-block:: bash

    # Auto-detects jemalloc at standard locations
    for JEMALLOC in /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
                    /usr/lib64/libjemalloc.so.2 \
                    /usr/local/lib/libjemalloc.so; do
        if [ -f "$JEMALLOC" ]; then
            export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$JEMALLOC"
            export MALLOC_CONF="${MALLOC_CONF:-dirty_decay_ms:5000,muzzy_decay_ms:5000}"
            break
        fi
    done

Installing jemalloc
-------------------

.. code-block:: bash

    # Ubuntu/Debian
    apt-get install libjemalloc2
    
    # RHEL/CentOS
    yum install jemalloc

API Reference
=============

cleanup_memory
--------------

.. code-block:: python

    from nvflare.fuel.utils.memory_utils import cleanup_memory

    cleanup_memory(torch_cuda_empty_cache=True)

**Signature:** ``cleanup_memory(torch_cuda_empty_cache: bool = False) -> None``

Performs allocator-aware memory cleanup:

1. Runs ``gc.collect()``
2. For glibc: Calls ``malloc_trim(0)``
3. For jemalloc: Relies on auto-decay (no action needed)
4. Optionally calls ``torch.cuda.empty_cache()``

get_allocator_type
------------------

.. code-block:: python

    from nvflare.fuel.utils.memory_utils import get_allocator_type

    allocator = get_allocator_type()  # "glibc", "jemalloc", or "unknown"

**Signature:** ``get_allocator_type() -> str``

Detects which memory allocator is in use at runtime. Result is cached.

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
3. Consider using jemalloc (LD_PRELOAD)
4. Monitor with ``top`` or ``htop``

High RSS on Client
------------------

1. Check ``MALLOC_ARENA_MAX=2`` is set
2. Enable ``client_memory_gc_rounds=1``
3. Enable ``torch_cuda_empty_cache=True`` for GPU
4. Consider using jemalloc

OOM Errors
----------

1. Reduce batch size
2. Enable memory cleanup every round (``client_memory_gc_rounds=1`` or ``server_memory_gc_rounds=1``)
3. Check for memory leaks in training code
4. Use jemalloc with appropriate decay settings

