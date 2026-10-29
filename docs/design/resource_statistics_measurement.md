# Resource statistics: measurement and accounting

This is the proposed measurement design for task-aware format 4.0, not a claim
that collection is implemented on main. The [overview](resource_statistics_design.md)
defines the user contract; the [reporting design](resource_statistics_reporting.md)
defines wire validation and identity. Execution records and intervals here are
internal artifacts, not a public task list. Prototype collectors inform this
proposal but do not establish support for a launcher or execution mode.

## Collection boundary and ownership

The existing client [job entry point](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/private/fed/app/client/worker_process.py)
and server [job entry point](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/private/fed/app/server/runner_process.py)
prepare/download the workspace before constructing application components. Add
the collector after that preparation and before configuration constructs user
components. Do not install it in the persistent site/server parent, change
Python startup, or count the initial workspace download as job-Cell traffic.

For each physical execution, probe capacity first, then open one monotonic
duration window and baseline CPU counters. For the job process, close CPU/duration
and freeze traffic at the existing pre-streaming-shutdown publication boundary. If known callbacks
or managed trainers can outlive that snapshot, mark affected coverage partial;
the later callback drain does not retroactively extend the observation window.
Scan retained files once per participant, before report files and the final
archive are written. Probe and scan overhead is outside the CPU window; there
is no sampling thread or periodic report sender.

Task collection runs inside the actual worker environment after launcher limits
and visibility are established. The existing [worker](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/private/fed/task_worker/worker.py)
and [supervisor](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/private/fed/task_worker/supervisor.py) remain lifecycle
owners: statistics may observe their accepted execution/settlement, but may not
reap, signal, retry, promote results, or delay settlement awaiting a report.
An optional worker artifact can be missing even when task completion is valid.
The worker stops its CPU/capacity window after application finalization and
result/checkpoint staging, before completion commit. Its final traffic snapshot
follows normal Cell close. These boundaries are described in the
[runtime sequence](resource_statistics_implementation.md).

## CPU and RAM capacity probes

Use Linux OS observations, not resource requests, scheduler reservations, free
resources, or framework discovery. Discover membership from `/proc/self/cgroup`
and mounts from `/proc/self/mountinfo` once per snapshot. Decode mount escapes
and map membership relative to a covering mount's root; reject traversal or a
path outside that root. Prefer the broadest covering visible mount to check
ancestor restrictions. In a hybrid hierarchy, a controller assigned to v1 must
not be read from an unrelated v2 mount.

For CPU, take the minimum usable bound from:

- `len(os.sched_getaffinity(0))`, which reflects effective CPU-set restrictions;
- each readable ancestor's v2 `cpu.max` quota/period, or v1
  `cpu.cfs_quota_us` / `cpu.cfs_period_us`, up to the visible mount root.

Quota ratios are logical CPU units, not integer processor counts: affinity of
eight CPUs plus quota `150000 100000` yields 1.5 units. Unlimited v2 `max` or
v1 `-1` is not a numeric bound. Validate periods, quotas and arithmetic; an
unreadable/malformed applicable limit makes usable results partial rather than
discarding valid affinity or another quota.

Without affinity, use a positive online CPU count as a partial fallback and
inspect readable effective cpusets along the visible hierarchy. Merge overlapping
CPU-list ranges before counting. In v1, fall back from `cpuset.effective_cpus`
to `cpuset.cpus`; empty/inherited or invalid lists must not become confident
zero. CPU shares/weights are not capacity bounds.

For RAM, take the minimum of physical page count times page size and finite
readable ancestor limits: v2 `memory.max`, or v1 `memory.limit_in_bytes`.
Recognize the kernel's v1 page-aligned unlimited sentinel rather than reporting
it as usable RAM. Reject malformed/nonpositive limits and overflow. Do not
substitute `memory.high`, swap limits, free memory, resident set size, or GPU
memory. A readable physical bound survives an unreadable cgroup limit with
partial quality; with no usable bound the dimension is unavailable.

Absent controllers differ from inaccessible discovery or required files.
An absent optional controller does not itself imply failure. Hidden ancestors
and limits changed after startup cannot be reconstructed. These are snapshots
of the visible environment, potentially shared with other jobs, not exclusive
allocations or an integral of changing limits.

Optional architecture/model descriptions cannot invalidate numeric capacity.
Split heterogeneous affinity-visible CPU models only when every selected CPU
is identified and no tighter shared quota needs allocation among models.
Otherwise retain aggregate units without inventing per-model fractions.
Non-Linux probes require tested equivalents; otherwise report unsupported
dimensions rather than guessing from a host-wide count.

## CUDA visibility and classification

Use a fresh helper process to call the system CUDA Driver, inheriting the
execution's final environment, namespaces, cgroup membership and affinity.
Do not initialize CUDA in the application process, import PyTorch, use a CUDA
Runtime bundled with a job package, run `nvidia-smi`, or count tokens in
`CUDA_VISIBLE_DEVICES`. The Driver applies visibility and enumeration rules.

On Linux load the versioned system Driver library and call `cuInit`, then
`cuDeviceGetCount`. A successful zero count, including the explicit no-device
result, is observed zero: an empty GPU group list. A missing library, other
initialization/enumeration failure, invalid count or timeout is unavailable,
not zero. Bound enumeration to 4,096 ordinals. Each counted ordinal survives
exactly once even if every optional metadata call fails.

Read optional model and total device memory for each ordinal. Use the MIG-aware
Driver UUID entry point when available to match an NVML handle. NVML may classify
that handle as a full GPU or MIG compute instance; it cannot add devices or
substitute a host-wide count. Do not infer type from a UUID prefix, model name,
memory size, or environment string. Missing/ambiguous UUID mapping, permission
errors and older APIs retain `cuda_device_unclassified` counts. UUIDs and host
identifiers stay private and are discarded after matching.

CUDA-visible MIG enumeration depends on Driver/runtime capabilities and policy;
it is not every configured instance on the host. An ordinal does not prove
exclusive allocation, actual kernels, GPU busy time, or an equal fraction of
a full GPU. Keep full-GPU, MIG and unclassified groups distinct; group only
equivalent known descriptions. Missing classification cannot erase a successful
count or fabricate hardware equivalence.

The helper has a two-second observation wait and bounded one-second reap waits
after a kill, including a final cleanup attempt. Accept at most 64 KiB of JSON;
oversized, invalid or failed output makes only GPU measurement unavailable.
The helper consolidates overlarge optional descriptions into counts by kind.
Output is captured in a temporary file, not an unbounded in-memory pipe. The read
limit is not a hard disk-write quota if interpreter startup itself writes output.
There is no sampling loop, framework import or change to the application's
CUDA initialization. Probe startup cost still needs qualification for short tasks.

The choice differs from copying [PyTorch's discovery implementation](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/cuda/__init__.py#L894-L966):
NVFlare does not need framework device caching or its NVML-based count/fallback machinery.
The system Driver is the numeric authority inside the same execution environment;
NVML only enriches already counted ordinals. A helper isolates Driver initialization
without adding PyTorch as a dependency. Tests must still establish that the helper
inherits the deployed container/scheduler visibility; a host NVML count cannot
substitute for that evidence.

## CPU work and disjoint execution scopes

Read user plus system CPU counters at both boundaries, not instantaneous CPU
percentages. On supported Unix platforms use `getrusage(RUSAGE_SELF)` and,
where appropriate, `RUSAGE_CHILDREN`. The latter covers terminated, waited
children, including deeper descendants only when intervening processes waited
normally. It does not enumerate arbitrary descendants or authorize statistics
to reap a child owned by another runtime component.

For a resident job, subtract self/children baselines from final counters.
Children normally waited before finalization are included; known live, detached,
unwaited, or incompletely observed managed descendants mark CPU attribution
partial. A negative, nonfinite or unreadable delta is unavailable, never clamped
to zero. The shared parent process is excluded.

For task lifetime, switch the coordinator to self-only accounting over its
original window. Add disjoint worker deltas, each including only its own normally
waited descendants. Never add those workers again through coordinator children
counters. Independently measured trainers/ranks must also be excluded from
enclosing children totals; otherwise report partial coverage rather than sum
overlapping scopes. Forked processes must discard inherited collector baselines
and copied locks before installing their own.

A physical worker session contributes once even across several assignments or
saved-result retransmissions. A retry that executes again adds work, including
observable failed/aborted work. The reducer retains accepted scope identities;
it does not dedupe by task name, result identity, PID alone, or hardware UUID
across sessions. Missing artifacts preserve known subtotals with incomplete
quality. The ledger cannot establish lifecycle ownership. Cap its retention at
10,000 scopes, 128 aggregate groups and 4 MiB encoded state, with at most 8 MiB
of outstanding worker artifacts and 64 KiB per accepted artifact. On exhaustion
freeze accepted subtotals and mark incomplete; never evict replay keys and then
admit duplicate work. These bounds limit serialized state, not total Python heap.

## Duration, capacity-time, and aggregation

Let execution `e` have monotonic interval `[s_e, f_e]`, duration `d_e`, CPU
capacity group `c_e,g`, RAM capacity `m_e`, and GPU group count `n_e,g`.
Compute once before reducing physical executions into a participant:

```text
d_e = (finish_monotonic_ns - start_monotonic_ns) / 1,000,000,000
CPU unit_seconds[g] = sum_e(c_e,g * d_e)
RAM byte_seconds = sum_e(m_e * d_e)
GPU instance_seconds[g] = sum_e(n_e,g * d_e)
cpu_seconds = sum_disjoint_scopes(delta_user + delta_system)
average logical CPUs = cpu_seconds / participant duration_seconds
```

Use checked decimal arithmetic. Values are nonnegative numeric strings no
greater than unsigned 128-bit maximum; seconds/CPU quantities allow at most
nine fractional digits, while bytes and device counts are integers. Retain
precision until the serialization boundary, never through binary floating-point
multiplication. Apply the reporting contract's field bounds and rounding.

`duration_seconds` is one participant elapsed window, not summed worker durations.
Include coordinator/worker capacity-time over their own physical windows; do not
apply worker capacity to coordinator waiting time. Startup `resource_capacity`
remains the job-process snapshot, not successive workers' summed capacities.
Job/study totals add already computed `resource_time`; they never multiply
startup capacity by duration again.

Group CPU/GPU capacity-time by compatible kind and optional descriptions, keeping
unknown descriptions distinct from known ones. RAM byte-seconds add directly.
Do not derive a full-GPU equivalent from heterogeneous/MIG groups. Multi-rank
adapters expose disjoint CPU scopes but one capacity observation per shared
execution environment; rank count cannot multiply a shared GPU or node-wide
CPU/RAM capacity. Multiple nodes need node-local observations, not rank-zero
capacity times node count. Without attributable coverage mark gaps.

If lifecycle evidence cannot place every included execution inside the participant
window, omit its duration/CPU average; retain valid work/capacity-time. Do not
compare monotonic timestamps across hosts. Zero duration has no CPU average.
Concurrent executions may expose the same hardware: capacity-time is observation,
not unique ownership. Never add filesystem capacities or multiply them by time.

## Retained files and filesystem observation

At participant finalization stream a depth-first run-directory traversal.
Sum regular entries' `st_size`; count hard-linked names separately and sparse
files by logical length. Include existing inputs, configuration and logs, not
just new outputs. Exclude symlinks, nonregular entries and only the top-level
`resource_stats/` directory. A nested application directory with that name is
ordinary content. Do not follow symlinks to files, directories, or the run root.

Use descriptor-relative directory operations with no-follow checks. Resolve
children relative to their open parent so a checked directory replaced with an
external symlink cannot escape the scan. Close all iterators/descriptors on every
exit. If safe primitives are unsupported, report unavailable rather than fall
back to a followable path walk.

Proposed fixed budgets: 100,000 visited entries, depth 64, and one second of
monotonic traversal, checked between entries. Keep depth-bounded iterators, not
a materialized directory listing. A limit, permission error, vanishing entry or
overflow retains a partial subtotal. Failure before observing either an entry
or an empty directory is unavailable; a successfully observed empty directory
is zero. No file contents are opened.

The scan is non-atomic: concurrent writes/renames and ordinary mounted subtrees
affect it. A cooperative deadline cannot interrupt a blocked filesystem syscall.
This is logical retained content, not peak storage, physical blocks, I/O bytes
or archive size. Separately use `statvfs` on the run directory for visible
filesystem capacity; do not walk mounts or add capacities.

## Message payload accounting

Reuse existing [Cell counters](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/fuel/f3/traffic_stats.py) and
[CoreCell transport hooks](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/fuel/f3/cellnet/core_cell.py). Send hooks
record buffer length after encoding/encryption and local communicator acceptance;
receive hooks record before decoding/decryption on a real connection. The
[buffer-length helper](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/fuel/f3/cellnet/utils.py) sums all segments.
Optional payload encryption overhead is included; Cell headers, framing,
TLS/network headers and transport-level retransmissions are not.

Include control, heartbeat, request/reply and streamed-chunk payloads actually
observed at job/worker/managed-trainer endpoints. Repeated application transfers
and forwarded hops count when crossing those endpoints. Do not add separate
streamer totals to the same CoreCell chunks. Direct in-process Cell delivery
has no transport connection and is excluded; local subprocess transport counts.
No model-size estimate or semantic deduplication replaces endpoint counters.

Snapshot without resetting; sum sent and received separately across owned
endpoints. Exclude shared parent/site/relay process traffic. Worker-to-coordinator
transfers appear at both local endpoints intentionally; this is not unique
network bytes. A successful send is not proof of receipt; traffic after the
snapshot can be absent. Trainer/session snapshots need monotonic cumulative
replacement, not repeated addition; retire once and preserve missing-source
quality independently.

## Quality propagation and worked example

Successful measurements omit `status`. `partial` retains usable values with
`observation_incomplete` or `attribution_incomplete`; `unavailable` has no usable
value, with a supported issue such as `unsupported` or `malformed_source`.
Use only reporting-contract issue codes, never raw paths/native errors/host
data. Missing fields are not zero. Capacity/time quality belongs to their owning
objects, not dimensions: retain known CPU/RAM when GPU fails. CPU work, retained
content and traffic each own their quality; filesystem is complete/unavailable.
Derived values inherit relevant gaps, not unrelated errors or missing labels.
For example, unreadable worker GPU capacity makes capacity-time partial while
valid CPU work and traffic can remain complete. A wholly missing worker report
makes all three rollups incomplete. Neither case changes the coordinator's valid
startup snapshot or the final file scan.

Consider these disjoint executions, all within one 60-second participant window:

<table class="docutils">
<thead>
<tr>
  <th>Execution</th>
  <th>Duration</th>
  <th>CPU capacity</th>
  <th>RAM capacity</th>
  <th>Visible GPUs</th>
  <th>CPU work</th>
</tr>
</thead>
<tbody>
<tr>
  <td>Coordinator</td>
  <td>60 s</td>
  <td>2 units</td>
  <td>2 GiB</td>
  <td>0</td>
  <td>2 CPU s</td>
</tr>
<tr>
  <td>Worker A</td>
  <td>10 s</td>
  <td>2 units</td>
  <td>4 GiB</td>
  <td>1</td>
  <td>8 CPU s</td>
</tr>
<tr>
  <td>Worker B</td>
  <td>5 s</td>
  <td>2 units</td>
  <td>4 GiB</td>
  <td>1</td>
  <td>4 CPU s</td>
</tr>
</tbody>
</table>

Assuming compatible CPU/GPU descriptions, the participant records 150 CPU-unit
seconds, 180 GiB-seconds (`193273528320` byte-seconds), 15 GPU-instance-seconds,
and 14 CPU seconds. Its duration remains 60 seconds and average CPU use is
14/60, about 0.23 logical CPUs. Its startup snapshot remains 2 CPU units, 2 GiB
RAM and zero GPUs. Workers need not run consecutively: overlap changes neither
the addition rule nor the meaning of observed capacity-time.

If Worker B is settled but its report is missing, keep the coordinator/A
subtotals with incomplete quality and the known containing duration. If B's
settlement is unresolved, omit the duration and average as well. A retry that
only resends A's saved result adds actual payload traffic, not another ten
seconds of GPU capacity-time.
