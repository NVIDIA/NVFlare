.. _memory_management:

###################
Memory Management
###################

This guide describes memory management techniques for long-running federated learning jobs
using Python, PyTorch, and glibc/jemalloc.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

Federated learning jobs can run for hours or days. Without proper memory management,
RSS (Resident Set Size) can grow continuously due to:

- Long-lived references that keep large model params alive between rounds (primary cause on clients)
- glibc memory arena fragmentation (freed memory not returned to the OS)
- PyTorch CUDA cache retention
- Cyclic references delaying Python garbage collection (supplementary; usually not the main driver)

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
   macOS still benefit from client-side parameter reference release (and optional
   ``gc.collect()``), but cannot release fragmented heap memory back to the OS
   via ``malloc_trim()``.

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

Server-Side Configuration
--------------------------

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

- ``0`` = Disabled (default for FedAvg-based recipes)
- ``1`` = Cleanup every round (default for FedOpt, FedAvgHE, and Cyclic recipes)
- ``5`` = Cleanup every 5 rounds (recommended for server)

Server Cleanup Effects
----------------------

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

**Recommendation**: Using ``server_memory_gc_rounds=5`` provides good memory
management with negligible performance impact. Only disable (``=0``) if you've
measured and confirmed RSS is stable without cleanup.

Client-Side Memory Cleanup
==========================

The primary client-side memory control is ``clear_cache=True`` (the default) in
``flare.send()``, which immediately releases parameter references after serialization.
In CPython, this reference release is what actually frees large tensor/array memory —
no explicit GC call is needed for that.

``client_memory_gc_rounds`` and ``cuda_empty_cache`` provide *supplemental* cleanup on
top of the reference release: periodic ``gc.collect()`` for cyclic objects,
``malloc_trim()`` to return freed pages to the OS, and optional CUDA cache clearing.

Client-Side Configuration
--------------------------

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
        cuda_empty_cache=True, # Clear GPU cache
    )

Swarm Learning Configuration
----------------------------

Swarm Learning uses ``memory_gc_rounds`` (not ``client_memory_gc_rounds``) and
``cuda_empty_cache`` on ``SimpleSwarmLearningRecipe``:

.. code-block:: python

    from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

    recipe = SimpleSwarmLearningRecipe(
        name="swarm_job",
        model=MyModel(),
        num_rounds=10,
        train_script="train.py",
        memory_gc_rounds=1,   # Cleanup every round on trainer and aggregator roles
        cuda_empty_cache=True,
    )

.. note::

   ``memory_gc_rounds`` and ``cuda_empty_cache`` are top-level Swarm recipe arguments.
   Do not pass them inside ``train_args`` (they are reserved keys).

**Parameters:**

- ``client_memory_gc_rounds``: Run *supplemental* cleanup (``gc.collect()`` + ``malloc_trim()``) every N rounds on client (0 = disabled). The primary cleanup is reference release via ``clear_cache=True`` in ``flare.send()``.
- ``cuda_empty_cache``: If True, call ``torch.cuda.empty_cache()`` on cleanup
- ``memory_gc_rounds`` (Swarm): Run supplemental cleanup every N rounds (0 = disabled)

When to use ``client_memory_gc_rounds > 1``
-------------------------------------------

Use values greater than ``1`` only when memory is already stable and you are tuning
for lower cleanup overhead:

- RSS trend is flat/bounded across rounds
- No CPU/GPU OOM pressure
- You want slightly better throughput/latency

Start with ``client_memory_gc_rounds=1``, then tune to ``2`` and optionally ``5`` while monitoring RSS.
If RSS begins to climb or OOM risk increases, revert to ``1``.

Client Cleanup Effects
----------------------

After each ``flare.send()`` on the client (with default ``clear_cache=True``):

1. FLARE releases references to sent and received model params.
2. In CPython, this reference release is the primary mechanism that reclaims large tensors/arrays.

Supplemental cleanup is also available and configurable:

3. Runs Python garbage collection (``gc.collect()``), mainly for cyclic references.
4. For glibc: returns free heap pages to OS (``malloc_trim()``).
5. For jemalloc: relies on auto-decay (no manual action needed).
6. Optionally clears PyTorch CUDA cache.

.. note::

   RSS may not drop immediately even after object release because allocators can retain memory
   for reuse. A flat RSS trend across rounds is typically the expected healthy behavior.

.. note::

   The lifecycle handling is transparent to user training scripts. No code changes are required
   in ``train.py`` for default behavior.

External Process Support
------------------------

For external process execution (``launch_external_process=True``), memory settings
are passed via environment variables:

- ``NVFLARE_CLIENT_MEMORY_GC_ROUNDS``: Cleanup interval
- ``NVFLARE_CUDA_EMPTY_CACHE``: GPU cache cleanup (``true``/``false``)

Recommended Settings
====================

+--------+-----------------------------+-----------------------------+----------------------+----------------------+
| Role   | ``server_memory_gc_rounds`` | ``client_memory_gc_rounds`` | ``MALLOC_ARENA_MAX`` | ``cuda_empty_cache`` |
+========+=============================+=============================+======================+======================+
| Server | 5                           | N/A                         | 4                    | N/A                  |
+--------+-----------------------------+-----------------------------+----------------------+----------------------+
| Client | N/A                         | 1                           | 2                    | True (for GPU)       |
+--------+-----------------------------+-----------------------------+----------------------+----------------------+

Using jemalloc
==============

For PyTorch workloads, jemalloc is recommended over glibc malloc. NVFlare startup
scripts preload jemalloc only when explicitly enabled via
``NVFLARE_ENABLE_JEMALLOC_PRELOAD=true`` and jemalloc is available.

Startup Script
--------------

The generated ``sub_start.sh`` script includes opt-in jemalloc preload:

.. code-block:: bash

    # Enable jemalloc preload only when opted in
    if [ "${NVFLARE_ENABLE_JEMALLOC_PRELOAD:-false}" = "true" ]; then
        for JEMALLOC in /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
                        /usr/lib64/libjemalloc.so.2 \
                        /usr/local/lib/libjemalloc.so; do
            if [ -f "$JEMALLOC" ]; then
                export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$JEMALLOC"
                export MALLOC_CONF="${MALLOC_CONF:-dirty_decay_ms:5000,muzzy_decay_ms:5000}"
                break
            fi
        done
    fi

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

    cleanup_memory(cuda_empty_cache=True)

**Signature:** ``cleanup_memory(cuda_empty_cache: bool = False) -> None``

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

1. Confirm ``flare.send()`` uses default ``clear_cache=True`` (or explicitly set it)
2. Check ``MALLOC_ARENA_MAX=2`` is set
3. Start with ``client_memory_gc_rounds=1``
4. Increase to ``2`` or ``5`` only if RSS is already stable and you are tuning performance
5. Enable ``cuda_empty_cache=True`` for GPU
6. Consider using jemalloc

OOM Errors
----------

1. Reduce batch size
2. Confirm ``flare.send()`` uses default ``clear_cache=True`` — this is the primary client fix
3. Enable supplemental cleanup every round (``client_memory_gc_rounds=1`` or ``server_memory_gc_rounds=1``)
4. Check for memory leaks in training code
5. Use jemalloc with appropriate decay settings
