# Resource statistics: measurement and accounting

This page describes how resource statistics are measured and accounted for. The
[overview](resource_statistics_design.md) defines the user contract, and the
[reporting design](resource_statistics_reporting.md) defines wire validation and
identity. Execution records and intervals are internal artifacts; the public
result contains one rollup per participant.

## Collection boundary and ownership

The existing client [job entry point](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/private/fed/app/client/worker_process.py)
and server [job entry point](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/private/fed/app/server/runner_process.py)
prepare or download the workspace before constructing application components.
The collector starts after that preparation and before configuration constructs
user components. It does not run in the persistent site or server parent, change
Python startup, or count the initial workspace download as job Cell traffic.

For each physical execution, CPU, RAM, and the GPU inventory are observed before the
monotonic duration window and CPU counters start. At the job process's
pre-streaming-shutdown publication boundary, the collector freezes CPU, duration,
and traffic. If known callbacks or managed trainers can outlive that snapshot,
the affected coverage is partial; a later callback drain does not extend the
observation window retroactively. The collector scans retained files once per
participant after CPU and duration are frozen, before report files and the final
archive are written. There is no sampling thread or periodic report sender.

Task collection runs inside the actual worker environment after launcher limits
and visibility are established. The existing [worker](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/private/fed/task_worker/worker.py)
and [supervisor](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/private/fed/task_worker/supervisor.py)
remain responsible for the lifecycle. Statistics may observe accepted executions
and their settlement, but may not reap, signal, retry, promote results, or delay
settlement while awaiting a report. An optional worker artifact can be missing
even when task completion is valid. The worker stops its CPU and capacity window
after application finalization and result or checkpoint staging, but before the
completion commit. Its final traffic snapshot follows normal Cell close.

## CPU and RAM capacity probes

Capacity is based on Linux OS observations, not resource requests, scheduler
reservations, free resources, or framework discovery. Each snapshot discovers
membership from `/proc/self/cgroup` and mounts from `/proc/self/mountinfo`. It
decodes mount escapes and maps membership relative to a covering mount's root,
rejecting traversal or a path outside that root. The broadest covering visible
mount is preferred so that ancestor restrictions can be checked. In a hybrid
hierarchy, a controller assigned to v1 must not be read from an unrelated v2 mount.

For CPU, the probe takes the minimum usable bound from:

- `len(os.sched_getaffinity(0))`, which reflects effective CPU-set restrictions;
- each readable ancestor's v2 `cpu.max` quota and period, or v1
  `cpu.cfs_quota_us` and `cpu.cfs_period_us`, up to the visible mount root.

Quota ratios express logical CPU units rather than integer processor counts. For
example, affinity to eight CPUs with quota `150000 100000` yields 1.5 units.
Unlimited v2 `max` or v1 `-1` is not a numeric bound. Periods, quotas, and
arithmetic must be validated. An unreadable or malformed applicable limit makes
usable results partial, but does not discard valid affinity or another quota.

If affinity is unavailable, the probe uses a positive online CPU count as a
partial fallback and inspects readable effective cpusets along the visible
hierarchy. It merges overlapping CPU-list ranges before counting. In v1, it
falls back from `cpuset.effective_cpus` to `cpuset.cpus`; empty, inherited, or
invalid lists must not be treated as a confident zero. CPU shares and weights
are not capacity bounds.

For RAM, the probe takes the minimum of physical page count times page size and
finite, readable ancestor limits: v2 `memory.max` or v1 `memory.limit_in_bytes`.
It recognizes the kernel's v1 page-aligned unlimited sentinel rather than
reporting it as usable RAM, accepts zero as a valid hard limit, and rejects
malformed or negative limits. A v1 ancestor applies only when its
`memory.use_hierarchy` is `1`; `0` ends the ancestor chain. An unreadable or
invalid hierarchy flag makes the retained bounds partial. The probe does
not substitute `memory.high`, swap limits, free memory, resident set size, or
GPU memory. A readable physical bound remains usable with partial quality when a
cgroup limit is unreadable; if no bound is usable, the dimension is unavailable.

An absent controller is different from inaccessible discovery or required files:
an absent optional controller does not itself imply failure. Hidden ancestors
and limits changed after startup cannot be reconstructed. These snapshots describe
the visible environment, which may be shared with other jobs. They are neither
exclusive allocations nor an integral of changing limits.

CPU and GPU hardware descriptions are not collected. CPU/RAM probes expose
scalar capacity values. Non-Linux probes require tested equivalents; without
them, affected dimensions are unsupported rather than guessed.

## GPU inventory and visibility

GPU capacity counts the `nvidia-smi` inventory IDs selected by the execution's
`CUDA_VISIBLE_DEVICES` mask. The probe and `GPUResourceManager` share
`gpu_utils.get_visible_gpu_ids`, so reporting follows the resource manager's
integer-index selection rules.

With no mask, the pool is the verified `nvidia-smi` inventory. An empty mask
establishes zero without running `nvidia-smi`. Integer selectors keep their order,
remove duplicates, and stop at the first negative index, empty item, or index
absent from the inventory.
A noninteger selector reached before selection stops leaves the statistics
observation unavailable. The host-backed resource manager retains its fallback
to host inventory for unsupported masks. Externally managed pools retain their
`ignore_host` behavior.

The statistics probe calls `get_host_gpu_ids(timeout=2.0)`. A missing command,
command failure, timeout, malformed output, or empty inventory is unavailable,
not an inferred zero. CPU and RAM observations remain usable independently.
The probe runs once in each execution's established environment, before its
measurement window, without initializing application CUDA, retaining a CUDA
context, importing a framework, or changing resource assignments.

`gpu.count` is an integer count of selected inventory IDs. `gpu.instance_seconds`
records that count multiplied by execution seconds, expressed as GPU-seconds. Neither value measures
GPU utilization, GPU memory, or exclusive hardware ownership. Inventory selection
does not verify that the application can initialize or access those devices
through CUDA.

## CPU work and disjoint execution scopes

CPU work uses user and system CPU counters at both boundaries, rather than
instantaneous CPU percentages. On supported Unix platforms, the collector uses
`getrusage(RUSAGE_SELF)` and, where appropriate, `RUSAGE_CHILDREN`. The latter
covers terminated, waited children, including deeper descendants only when
intervening processes waited normally. It does not enumerate arbitrary descendants
or authorize statistics to reap a child owned by another runtime component.

For a resident job, the collector subtracts self and children baselines from the
final counters. It includes children normally waited before finalization; known
live, detached, unwaited, or incompletely observed managed descendants make CPU
attribution partial. A negative, nonfinite, or unreadable delta is unavailable
and is never clamped to zero. The shared parent process is excluded.

For task lifetime, the coordinator switches to self-only accounting over its
original window. Disjoint worker deltas are added, each including only its own
normally waited descendants. The same workers must not also be added through
coordinator children counters. Independently measured trainers and ranks must
likewise be excluded from enclosing children totals; otherwise coverage is partial
rather than a sum of overlapping scopes. Forked processes must discard inherited
collector baselines and copied locks before installing their own.

Each physical worker session contributes once, even across several assignments or
saved-result retransmissions. A retry that executes again adds work, including
observable failed or aborted work. The reducer retains accepted scope identities;
it does not deduplicate by task name, result identity, PID alone, or hardware UUID
across sessions. Missing artifacts leave known subtotals with incomplete quality.
The ledger cannot establish lifecycle ownership. Its retention is capped at
10,000 scopes and 4 MiB of encoded state, with at most
8 MiB of outstanding worker artifacts and 64 KiB per accepted artifact. On
exhaustion, accepted subtotals are frozen and marked incomplete; replay keys are
never evicted to admit duplicate work. These bounds limit serialized state, not
total Python heap.

## Duration, capacity-time, and aggregation

Let execution `e` have monotonic interval `[s_e, f_e]`, duration `d_e`, CPU
capacity `c_e`, RAM capacity `m_e`, and GPU count `n_e`. The
following values are computed before physical executions are reduced into a
participant:

```text
d_e = (finish_monotonic_ns - start_monotonic_ns) / 1,000,000,000
CPU unit_seconds = sum_e(c_e * d_e)
RAM byte_seconds = sum_e(m_e * d_e)
GPU instance_seconds = sum_e(n_e * d_e)
cpu_seconds = sum_disjoint_scopes(delta_user + delta_system)
average logical CPUs = cpu_seconds / participant duration_seconds
```

Values are native JSON numbers. Byte/count values are integers; durations,
CPU units/work and capacity-time may be fractional. Integer products and sums
remain exact Python integers; mixed arithmetic uses floating point and rejects
nonfinite results. There is no decimal-string conversion or fixed fractional
precision. An invalid dimension does not erase independent valid subtotals.

`duration_seconds` is one participant elapsed window, not the sum of worker
durations. Coordinator and worker capacity-time uses their own physical windows;
worker capacity is not applied to coordinator waiting time. Startup
`resource_capacity` remains the job-process snapshot rather than the sum of
successive workers' capacities. Job and study totals add already computed
`resource_time`; they do not multiply startup capacity by duration again.

CPU-unit-seconds, RAM byte-seconds and GPU-seconds are summed as scalar
quantities, without model/type grouping. Multi-rank
adapters expose disjoint CPU scopes but only one capacity observation per shared
execution environment. Rank count cannot multiply shared GPU or node-wide CPU/RAM
capacity. Multiple nodes require node-local observations rather than multiplying
rank-zero capacity by node count. Unattributable coverage remains a gap.

If lifecycle evidence cannot place every included execution inside the participant
window, the duration and CPU average are omitted while valid work and capacity-time
are retained. Monotonic timestamps are not compared across hosts. Zero duration
has no CPU average. Concurrent executions may expose the same hardware, so
capacity-time is an observation rather than unique ownership. Filesystem capacities
are never added or multiplied by time.

## Retained files and filesystem observation

At participant finalization, the collector streams a depth-first run-directory
traversal and sums regular entries' `st_size`. It counts hard-linked names
separately and sparse files by logical length. The scan includes existing inputs,
configuration, and logs, not just new outputs. It excludes symlinks, nonregular
entries, and only the top-level `resource_stats/` directory. A nested application
directory with that name is ordinary content. The scan does not follow symlinks
to files, directories, or the run root.

Descriptor-relative directory operations use no-follow checks. Children are
resolved relative to their open parent so that a checked directory replaced with
an external symlink cannot escape the scan. All iterators and descriptors are
closed on every exit. If safe primitives are unsupported, the measurement is
unavailable rather than falling back to a followable path walk.

The scan budgets are 100,000 visited entries, depth 64, and one second of
monotonic traversal, checked between entries. Iterators are depth-bounded rather
than materializing directory listings. A limit, permission error, or vanishing
entry retains a partial subtotal. Failure before observing either an entry or an
empty directory is unavailable; a successfully observed empty directory is zero.
File contents are not opened.

The scan is non-atomic: concurrent writes and renames, as well as ordinary mounted
subtrees, affect it. The result measures logical retained content, not peak storage,
physical blocks, I/O bytes, or archive size. Separately, `statvfs` on the run
directory observes visible filesystem capacity without walking mounts or adding
capacities.

Filesystem observations run in an isolated process behind a bounded finalization
wait of two seconds, including helper startup, followed by at most 0.25 seconds
of synchronous reaping. The cooperative scan deadline limits ordinary traversal;
the outer wait also covers filesystem calls that do not return. At expiry,
cleanup proceeds with any safely captured measurement and marks unfinished
observations unavailable. A captured scan subtotal retains partial quality. Late
results are ignored. The runtime owns observer termination and cleanup. A
late-starting observer is also abandoned, and the helper monitors its owner
process and control pipe. Job termination and resource release do not wait
indefinitely for the observer. These bounds apply to optional filesystem
observations, not to handoff or archive I/O. A scan can enter a stalled mounted
subtree while the run-directory root remains accessible. The separate process
allows termination of that observation; a daemon-thread timeout would leave its
filesystem call running inside the job process. If startup consumes the observation
budget, unfinished measurements are unavailable.

## Message payload accounting

The design reuses existing [Cell counters](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/fuel/f3/traffic_stats.py)
and [CoreCell transport hooks](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/fuel/f3/cellnet/core_cell.py).
Send hooks record buffer length after encoding and encryption, once the local
communicator accepts the message. Receive hooks record it before decoding and
decryption on a real connection. The [buffer-length helper](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/fuel/f3/cellnet/utils.py)
sums all segments. Optional payload encryption overhead is included; Cell headers,
framing, TLS and network headers, and transport-level retransmissions are not.

The counters include control, heartbeat, request/reply, and streamed-chunk
payloads observed at job, worker, and managed-trainer endpoints. Repeated
application transfers and forwarded hops count when crossing those endpoints.
Separate streamer totals are not added to the same CoreCell chunks. Direct
in-process Cell delivery has no transport connection and is excluded; local
subprocess transport counts. Endpoint counters are not replaced by a model-size
estimate or semantic deduplication.

Snapshots do not reset counters, and sent and received bytes are summed separately
across owned endpoints. Shared parent, site, and relay process traffic is excluded.
Worker-to-coordinator transfers intentionally appear at both local endpoints, so
this is not a count of unique network bytes. A successful send does not prove
receipt, and traffic after the snapshot can be absent. Trainer and session
snapshots require monotonic cumulative replacement rather than repeated addition;
each source is retired once, with missing-source quality preserved independently.

## Quality propagation and worked example

Successful measurements omit `status`. A `partial` measurement retains usable
values with `observation_incomplete` or `attribution_incomplete`; an
`unavailable` measurement has no usable value. Its owning scope records a
supported issue such as `unsupported` or `malformed_source`, tagged with the
affected metric. Only reporting-contract issue codes are used, never raw paths,
native errors, or host data. Missing fields are not zero. Capacity and time
quality belongs to their owning objects rather than individual dimensions, so
known CPU and RAM values are retained when GPU measurement fails. CPU work,
retained content, and traffic each own their quality; filesystem is complete or
unavailable. A participant, worker, handoff or aggregate owns one `issues` list;
metrics retain their own status. Complete scopes use an empty list. Derived
totals describe their own coverage gaps rather than copying source issue
details.

For example, unreadable worker GPU capacity makes capacity-time partial while
valid CPU work and traffic can remain complete. A wholly missing worker report
makes all three rollups incomplete. Neither case changes the coordinator's valid
startup snapshot or the final file scan.

Consider these disjoint executions, all within one 60-second participant window:

| Execution | Duration | CPU capacity | RAM capacity | GPU count | CPU work |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coordinator | 60 s | 2 units | 2 GiB | 0 | 2 CPU s |
| Worker A | 10 s | 2 units | 4 GiB | 1 | 8 CPU s |
| Worker B | 5 s | 2 units | 4 GiB | 1 | 4 CPU s |

The participant records 150 CPU-unit seconds, 180 GiB-seconds (`193273528320`
byte-seconds), 15 GPU-seconds, and 14 CPU seconds. Its duration remains
60 seconds and average CPU use is 14/60, about 0.23 logical CPUs. Its startup
snapshot remains 2 CPU units, 2 GiB RAM, and zero GPUs. Workers need not run
consecutively: overlap changes neither the addition rule nor the meaning of
observed capacity-time.

If Worker B is settled but its report is missing, the coordinator and A subtotals
are retained with incomplete quality, along with the known containing duration.
If B's settlement is unresolved, the duration and average are also omitted. A
retry that only resends A's saved result adds actual payload traffic, not another
ten seconds of GPU capacity-time.
