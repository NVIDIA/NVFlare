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

For each physical execution, the collector first probes capacity, then opens a
monotonic duration window and records baseline CPU counters. For the job process,
it closes the CPU and duration window and freezes traffic at the existing
pre-streaming-shutdown publication boundary. If known callbacks or managed
trainers can outlive that snapshot, the affected coverage is partial; a later
callback drain does not extend the observation window retroactively. The collector
scans retained files once per participant, before report files and the final
archive are written. Probe and scan overhead is outside the CPU window, and there
is no sampling thread or periodic report sender.

Task collection runs inside the actual worker environment after launcher limits
and visibility are established. The existing [worker](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/private/fed/task_worker/worker.py)
and [supervisor](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/private/fed/task_worker/supervisor.py)
remain responsible for the lifecycle. Statistics may observe accepted executions
and their settlement, but may not reap, signal, retry, promote results, or delay
settlement while awaiting a report. An optional worker artifact can be missing
even when task completion is valid. The worker stops its CPU and capacity window
after application finalization and result or checkpoint staging, but before the
completion commit. Its final traffic snapshot follows normal Cell close. The
[runtime sequence](resource_statistics_implementation.md) describes these boundaries.

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
reporting it as usable RAM, and rejects malformed or nonpositive limits and
overflow. It does not substitute `memory.high`, swap limits, free memory,
resident set size, or GPU memory. A readable physical bound remains usable with
partial quality when a cgroup limit is unreadable; if no bound is usable, the
dimension is unavailable.

An absent controller is different from inaccessible discovery or required files:
an absent optional controller does not itself imply failure. Hidden ancestors
and limits changed after startup cannot be reconstructed. These snapshots describe
the visible environment, which may be shared with other jobs. They are neither
exclusive allocations nor an integral of changing limits.

Optional architecture and model descriptions cannot invalidate numeric capacity.
The probe splits heterogeneous affinity-visible CPU models only when every
selected CPU is identified and no tighter shared quota needs to be allocated
among models. Otherwise, it retains aggregate units without inventing per-model
fractions. Non-Linux probes require tested equivalents; without them, affected
dimensions are reported as unsupported rather than guessed from a host-wide count.

## CUDA visibility and classification

A fresh helper process calls the system CUDA Driver, inheriting the execution's
final environment, namespaces, cgroup membership, and affinity. The probe does
not initialize CUDA in the application process, import PyTorch, use a CUDA Runtime
bundled with a job package, run `nvidia-smi`, or count tokens in
`CUDA_VISIBLE_DEVICES`. The Driver applies visibility and enumeration rules.

On Linux, the helper loads the versioned system Driver library, calls `cuInit`,
and then calls `cuDeviceGetCount`. A successful zero count, including the explicit
no-device result, is an observed zero and produces an empty GPU group list. A
missing library, another initialization or enumeration failure, an invalid count,
or a timeout makes the measurement unavailable rather than zero. Enumeration
is bounded to 4,096 ordinals. Every counted ordinal is retained exactly once,
even if all optional metadata calls fail.

The helper reads the optional model and total device memory for each ordinal. It
uses the MIG-aware Driver UUID entry point when available to match an NVML handle.
NVML may classify that handle as a full GPU or MIG compute instance; it cannot
add devices or substitute a host-wide count. Type must not be inferred from a
UUID prefix, model name, memory size, or environment string. Missing or ambiguous
UUID mappings, permission errors, and older APIs retain `cuda_device_unclassified`
counts. UUIDs and host identifiers remain private and are discarded after matching.

CUDA-visible MIG enumeration depends on Driver/runtime capabilities and policy;
it does not necessarily include every configured instance on the host. An ordinal
does not prove exclusive allocation, actual kernels, GPU busy time, or an equal
fraction of a full GPU. Full-GPU, MIG, and unclassified groups remain distinct,
and only equivalent known descriptions are grouped. Missing classification cannot
erase a successful count or fabricate hardware equivalence.

The helper has a two-second observation wait and bounded one-second reap waits
after a kill, including a final cleanup attempt. The collector accepts at most
64 KiB of JSON output; oversized, invalid, or failed output makes only the GPU measurement
unavailable. The helper consolidates overlarge optional descriptions into counts
by kind. Output is captured in a temporary file rather than an unbounded in-memory
pipe. The read limit is not a hard disk-write quota if interpreter startup itself
writes output. There is no sampling loop, framework import, or change to the
application's CUDA initialization. Helper startup is a fixed per-execution cost
and can be significant for short tasks.

This approach differs from copying [PyTorch's discovery implementation](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/cuda/__init__.py#L894-L966):
NVFlare does not need framework device caching or its NVML-based count and
fallback machinery. The system Driver is the numeric authority inside the same
execution environment; NVML only enriches already counted ordinals. The helper
isolates Driver initialization without adding PyTorch as a dependency. It must
inherit deployed container and scheduler visibility, because a host NVML count
cannot substitute for observation inside the execution environment.

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
10,000 scopes, 128 aggregate groups, and 4 MiB of encoded state, with at most
8 MiB of outstanding worker artifacts and 64 KiB per accepted artifact. On
exhaustion, accepted subtotals are frozen and marked incomplete; replay keys are
never evicted to admit duplicate work. These bounds limit serialized state, not
total Python heap.

## Duration, capacity-time, and aggregation

Let execution `e` have monotonic interval `[s_e, f_e]`, duration `d_e`, CPU
capacity group `c_e,g`, RAM capacity `m_e`, and GPU group count `n_e,g`. The
following values are computed before physical executions are reduced into a
participant:

```text
d_e = (finish_monotonic_ns - start_monotonic_ns) / 1,000,000,000
CPU unit_seconds[g] = sum_e(c_e,g * d_e)
RAM byte_seconds = sum_e(m_e * d_e)
GPU instance_seconds[g] = sum_e(n_e,g * d_e)
cpu_seconds = sum_disjoint_scopes(delta_user + delta_system)
average logical CPUs = cpu_seconds / participant duration_seconds
```

Calculations use checked decimal arithmetic. Values are nonnegative numeric
strings no greater than the unsigned 128-bit maximum. Seconds and CPU quantities
allow at most nine fractional digits, while bytes and device counts are integers.
Precision is retained until the serialization boundary without binary floating-point
multiplication. The reporting contract defines field bounds and rounding.

`duration_seconds` is one participant elapsed window, not the sum of worker
durations. Coordinator and worker capacity-time uses their own physical windows;
worker capacity is not applied to coordinator waiting time. Startup
`resource_capacity` remains the job-process snapshot rather than the sum of
successive workers' capacities. Job and study totals add already computed
`resource_time`; they do not multiply startup capacity by duration again.

CPU and GPU capacity-time is grouped by compatible kind and optional descriptions,
with unknown descriptions distinct from known ones. RAM byte-seconds add directly.
A full-GPU equivalent is not derived from heterogeneous or MIG groups. Multi-rank
adapters expose disjoint CPU scopes but only one capacity observation per shared
execution environment; rank count cannot multiply shared GPU or node-wide CPU or
RAM capacity. Multiple nodes require node-local observations rather than rank-zero
capacity multiplied by node count. Coverage that cannot be attributed is marked
as a gap.

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
than materializing directory listings. A limit, permission error, vanishing entry,
or overflow retains a partial subtotal. Failure before observing either an entry
or an empty directory is unavailable; a successfully observed empty directory
is zero. File contents are not opened.

The scan is non-atomic: concurrent writes and renames, as well as ordinary mounted
subtrees, affect it. A cooperative deadline cannot interrupt a blocked filesystem
syscall. The result measures logical retained content, not peak storage, physical
blocks, I/O bytes, or archive size. Separately, `statvfs` on the run directory
observes visible filesystem capacity without walking mounts or adding capacities.

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
values with `observation_incomplete` or `attribution_incomplete`; an `unavailable`
measurement has no usable value and carries a supported issue such as `unsupported`
or `malformed_source`. Only reporting-contract issue codes are used, never raw
paths, native errors, or host data. Missing fields are not zero. Capacity and time
quality belongs to their owning objects rather than individual dimensions, so
known CPU and RAM values are retained when GPU measurement fails. CPU work,
retained content, and traffic each own their quality; filesystem is complete or
unavailable. Derived values inherit relevant gaps, not unrelated errors or missing
labels.

For example, unreadable worker GPU capacity makes capacity-time partial while
valid CPU work and traffic can remain complete. A wholly missing worker report
makes all three rollups incomplete. Neither case changes the coordinator's valid
startup snapshot or the final file scan.

Consider these disjoint executions, all within one 60-second participant window:

| Execution | Duration | CPU capacity | RAM capacity | Visible GPUs | CPU work |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coordinator | 60 s | 2 units | 2 GiB | 0 | 2 CPU s |
| Worker A | 10 s | 2 units | 4 GiB | 1 | 8 CPU s |
| Worker B | 5 s | 2 units | 4 GiB | 1 | 4 CPU s |

Assuming compatible CPU and GPU descriptions, the participant records 150
CPU-unit seconds, 180 GiB-seconds (`193273528320` byte-seconds), 15 GPU-instance-seconds,
and 14 CPU seconds. Its duration remains 60 seconds and average CPU use is
14/60, about 0.23 logical CPUs. Its startup snapshot remains 2 CPU units, 2 GiB
RAM, and zero GPUs. Workers need not run consecutively: overlap changes neither
the addition rule nor the meaning of observed capacity-time.

If Worker B is settled but its report is missing, the coordinator and A subtotals
are retained with incomplete quality, along with the known containing duration.
If B's settlement is unresolved, the duration and average are also omitted. A
retry that only resends A's saved result adds actual payload traffic, not another
ten seconds of GPU capacity-time.
