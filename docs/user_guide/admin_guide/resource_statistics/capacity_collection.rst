.. _resource_statistics_capacity_collection:

Capacity collection
*******************

CPU and RAM
===========

On Linux, CPU capacity starts with process affinity and is capped by visible
cgroup CPU quotas. RAM starts with physical memory and is capped by visible
cgroup memory limits. The probes resolve controller membership and mount roots
and inspect applicable ancestors in cgroup v1, v2 and hybrid layouts. A zero RAM
limit is a valid observation. Unreadable limits retain usable bounds with partial
quality; hidden ancestors cannot be reconstructed.

GPU inventory
=============

The probe uses the shared ``gpu_utils`` inventory and integer-mask selection
rules used by ``GPUResourceManager``. With no ``CUDA_VISIBLE_DEVICES`` mask,
it counts the IDs returned by ``nvidia-smi``. An empty mask establishes zero
without running the command. Integer selectors remove duplicates and stop at
the first negative index, empty item or ID absent from inventory.

A noninteger selector reached before selection stops makes statistics
unavailable. The host-backed resource manager retains its existing fallback to
all inventory IDs for unsupported masks. Statistics do not use that fallback.

Inventory runs once per execution with a two-second timeout. Missing commands,
failed queries, malformed output, timeouts and empty inventory are unavailable.
This selected-inventory count does not verify application CUDA access and does
not describe GPU utilization, GPU memory or exclusive ownership. The probe
neither initializes CUDA nor starts a persistent helper.

Execution timing
================

Capacity probing finishes before the monotonic duration and CPU baselines.
Collection freezes those counters at the execution's final reporting boundary.
Task workers use their own environments and durations. Later application changes
do not resize the startup snapshot, and queries do not recompute capacity-time
from the participant's containing duration.
