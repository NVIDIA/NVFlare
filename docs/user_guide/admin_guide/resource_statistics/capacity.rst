.. _resource_statistics_capacity:

Resource capacity and capacity-time
***********************************

``resource_capacity`` records the CPU, RAM and GPU capacity visible when the job
process starts. It describes available resources, not their utilization or
exclusive allocation. CPU capacity can be fractional when a CPU-time quota
applies; GPU counts and RAM bytes are integers.

``resource_time`` adds each execution's capacity multiplied by its own duration.
Two selected GPUs over ten minutes contribute one-third of a GPU-hour. For task
jobs, worker contributions are added once, while the startup snapshot still
describes the coordinator. Overlapping participants can see the same hardware;
the sum is not an inventory of unique machines.

The CLI shows CPU core-hours, RAM GiB-hours and GPU-hours. ``--site`` also shows
the participant's startup capacity. A missing GPU observation leaves usable CPU
and RAM capacity available with partial quality. An explicitly empty GPU pool
is zero, while an unsuccessful inventory query is unavailable.

.. toctree::
   :maxdepth: 1

   capacity_collection
