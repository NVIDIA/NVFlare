# Resource statistics design

## Purpose

Resource statistics provides a final, best-effort account of the resources visible
to a job, its recorded CPU work, retained file sizes, and message payloads. It
lets users inspect a participant or compare retained jobs without installing a
monitoring service or changing their applications.

Collection follows the runtime's execution boundaries and admission checks. Each
execution mode provides attributable measurements and lifecycle ownership; resource
statistics does not select or enable backends.

## Relationship to JobStatsReporter

Resource statistics automatically records startup capacity, capacity-time, CPU
work, message payload traffic and retained file sizes. Use it for final
participant reports and job/study comparisons; no application widget needs to
be enabled.

Use the optional [JobStatsReporter](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/app_common/widgets/job_stats_reporter.py)
widget for task/round timing, errors and communication diagnostics. Configure it
on the server and on clients whose task measurements are needed. With
`collect_resources=True` (default), clients also sample process-family CPU
utilization, resident memory (RSS), GPU utilization and GPU memory when available.
`resource_sample_interval` controls the interval; `collect_resources=False`
keeps diagnostics without resource sampling.

Both can run in the same job and write separate reports. Their scopes and
observation windows differ: CPU work and GPU capacity-time are not derived from
utilization samples, and overlapping measurements should not be added together.
Sampling and telemetry add job overhead that resource statistics can include.

## Detailed design

This page introduces the contract. The following pages describe the measurement
rules and reporting behavior in detail:

1. [Measurement and accounting](resource_statistics_measurement.md): the APIs,
   limits, execution windows and formulas used to obtain each measurement.
2. [Reports, delivery and queries](resource_statistics_reporting.md): record
   formats, validation, deduplication, archive publication and authorized reads.

## Measurement contract

A participant is the server or a client site. Each participant has one final
rollup, a shared `duration_seconds`, and separate capacity and utilization
measurements. `resource_capacity` is the job-process startup snapshot.
`resource_time` rolls up capacity multiplied by each execution's own duration;
it uses the same representation as job/study totals. Worker records stay internal.
The following abbreviated example for a resident job omits identity:

```json
{
  "duration_seconds": 10,
  "resource_capacity": {
    "cpu": {"units": 2},
    "memory": {"bytes": 2147483648},
    "gpu": {"count": 1}
  },
  "resource_time": {
    "cpu": {"unit_seconds": 20},
    "memory": {"byte_seconds": 21474836480},
    "gpu": {"instance_seconds": 10}
  },
  "resource_utilization": {
    "cpu_consumed": {"cpu_seconds": 2},
    "retained_content": {"bytes": 1024},
    "message_traffic": {"sent_bytes": 2048, "received_bytes": 4096}
  },
  "issues": []
}
```

The example describes one resident participant. CPU and RAM capacity are
observed at job-process startup alongside the GPU count, using the
execution's launcher-established environment. Work and traffic follow their respective
collection boundaries, and retained content is scanned at finalization. In task
mode, capacity-time also includes worker windows, while participant startup
capacity still describes the job process. Quantities use native finite JSON
numbers; byte and device counts are integers.

| Field | Meaning, scope, and collection |
| --- | --- |
| `duration_seconds` | Elapsed seconds in the participant's observation window, including waiting. A monotonic clock measures the window from the start of CPU accounting to its final snapshot. |
| `resource_capacity.cpu.units` | Logical CPU capacity visible to the job process at startup, based on its CPU affinity and readable quota limits. Units can be fractional. |
| `resource_capacity.memory.bytes` | Host physical RAM capacity in bytes, reduced by readable memory limits, as observed by the job process at startup. It is not free memory or peak usage. |
| `resource_capacity.gpu.count` | The number of `nvidia-smi` inventory IDs selected by the execution's visibility mask, using the same integer-index selection rules as NVFlare's GPU resource manager. It does not verify CUDA access, use, or exclusive allocation. |
| `resource_time.cpu.unit_seconds` | The sum of each included execution's CPU capacity multiplied by its duration, in CPU-unit-seconds. In this resident example, 2 CPU units × 10 seconds = 20 CPU-unit-seconds. |
| `resource_time.memory.byte_seconds` | The sum of each included execution's RAM capacity in bytes multiplied by its duration in seconds. Here, 2,147,483,648 bytes × 10 seconds = 21,474,836,480 byte-seconds. |
| `resource_time.gpu.instance_seconds` | The sum of each included execution's GPU count multiplied by its duration, in GPU-seconds. The example records one GPU for 10 seconds. |
| `resource_utilization.cpu_consumed.cpu_seconds` | Total user and system CPU work in seconds across included, disjoint process scopes and normally waited descendants, measured from CPU counters over their execution windows; arbitrary unwaited descendants are not guaranteed covered. One CPU working for two seconds and two CPUs working for one second both contribute two CPU-seconds. |
| `resource_utilization.retained_content.bytes` | Logical length in bytes of regular files retained in the participant's run directory at finalization, measured by a bounded scan. Symlinks and the top-level report directory are excluded. It is not physical disk occupancy or peak storage. |
| `resource_utilization.message_traffic.sent_bytes` and `received_bytes` | Payload bytes counted separately at the participant's job Cell and included worker and managed-trainer endpoints, from their observed traffic through their final snapshots. Control payloads and repeated transfers are included; shared parent traffic and network headers are excluded. |

The workspace filesystem's visible capacity is also observed at finalization and
appears in participant detail. It is not storage reserved for the job.

`resource_utilization` groups recorded work, content, and traffic; it does not
claim GPU busy time or RAM utilization. Each metric retains its own
status beside its value. One scope-level `issues` list identifies affected metrics
and codes, so a partial file scan does not affect CPU results. A complete scope
uses `issues: []`. Field names identify units.

The collector computes capacity-time for each execution and combines it in
`resource_time`. Job/study queries sum those participant values without multiplying
startup capacity by duration again. The example yields 20 CPU-unit seconds and
10 GPU-seconds; average CPU use is 2/10 = 0.2 logical CPUs. Capacity-time
can compare the visible capacity and durations of retained jobs: for example, it
can distinguish a job that observed one GPU for 60 seconds from one that observed
one GPU for 10 seconds. It does not measure GPU busy time. Shared
hardware can be visible to several executions, so totals are observations, not
unique hardware, reservations or billing. Filesystem capacity is never
summed or multiplied by duration.

Sent and received counters are summed separately, including local endpoint
traffic. A successful local send is not proof of receipt. Their sum is not a
deduplicated network-byte total. Direct in-process delivery is excluded.

## Collection and delivery

```text
Job process: prepare workspace -> probe CPU/RAM and GPU inventory
             -> open counters -> run application -> freeze CPU/duration
             -> scan files
Parent:      consume terminal handoff -> release resources -> report outcome
Server:      validate participant report -> finalize workspace archive
User:        authorized job/site/study query -> derive totals -> render
```

Collection starts after existing workspace preparation and before application
construction in the client/server job entry points. Observations are finalized
during cleanup, so application construction or execution failures can still
produce statistics. Collection does not change normal Python startup, resource
assignment, or process ownership. CPU and memory probes use existing affinity
and visible cgroup limits. GPU capacity uses NVFlare's shared integer-index visibility
parser and a verified `nvidia-smi` inventory, with a two-second command timeout.
An empty visibility mask establishes zero without running the command;
unsupported selectors or failed inventory leave the GPU observation unavailable.
Collection does not initialize application CUDA or retain a CUDA context.

The parent supplies participant identity and attaches the report to the existing
`REPORT_JOB_OUTCOME` request. It releases resources through the normal lifecycle
even if statistics are missing, invalid, or delayed. An error during resource
release is logged without suppressing the terminal outcome report. The server
retains accepted reports until finalization, writes participant files and then
`resource_summary.json` under `resource_stats/`, and includes them in the normal
workspace archive. Queries use existing per-job authorization. There is no
separate statistics database, periodic sender, or live resource-monitoring
service.

## Quality, failure, and cost

Successful measurements omit `status`. `partial` retains usable values with a
coverage explanation; `unavailable` has no usable value. An unreadable limit
does not erase another usable observation. A missing or failed `nvidia-smi`
inventory is unavailable; an explicit empty GPU pool is zero.

Reports are untrusted site self-reports. Validation checks sender/job identity,
shape, numeric limits, and duplicates, but does not attest hardware. Equivalent
duplicates do not add measurements; conflicting replacements are rejected.
Missing participants remain explicit gaps. A server crash before archive
publication or a killed job process can lose statistics. Collection failures
must not change the training outcome, delay resource release indefinitely, or
create a second recovery protocol.

When a Client Job restarts, the replacement has a new observation window. The
final report preserves the available terminal generation's subtotal; it does not
accumulate earlier generations. If earlier observations are missing, affected
capacity-time, CPU and traffic measurements are marked incomplete. The CPU
average uses the terminal generation's containing duration as its denominator,
when that duration is valid; it does not use elapsed time across restarts.
Retained content comes from the final workspace scan. The parent accepts only a
handoff bound to the terminal launch generation; missing terminal observations
remain gaps even if an earlier generation's file survives. Scheduling retries
alone do not establish that an earlier execution occurred.

Filesystem observations run in isolation with a bounded finalization wait, so a
blocked optional probe does not hold up job cleanup or resource release
indefinitely. Captured measurements are preserved, unfinished observations
become unavailable, and late results are ignored. Requested archive members and
ZIP indexing have separate decoded-data and metadata-read bounds. Remote job
stores may still transfer the whole workspace archive to answer a small query.
Query latency and temporary storage can therefore depend on the archive size,
not just the statistics size. Each server admits one archive-backed job or study
query at a time and rejects concurrent requests with `resource query busy; retry
later`. This bounds concurrency only; the [reporting
design](resource_statistics_reporting.md) describes the remaining transfer-size,
disk-space, and execution-time limitations.

## Task lifetime

With `execution_lifetime="task"`, the resident Client Job coordinates workers
whose resources and lifetimes can differ from its own. Statistics are collected
inside each worker's actual environment and rolled into the participant's
existing CPU-work, capacity-time, and traffic fields. Retained content is
scanned once at finalization. There is no public task or worker list.

For example, a 60-second coordinator with zero GPUs and 2 CPU seconds launches
one-GPU workers lasting 10 and 5 seconds, using 8 and 4 CPU seconds. The participant
reports 15 GPU-seconds, 14 CPU seconds and a 60-second duration. Average
CPU use is about 0.23 logical CPUs. It does not report two GPUs or multiply worker
capacity by the coordinator's lifetime.

Task-mode CPU uses coordinator self CPU plus disjoint worker CPU; workers are
not also added through the coordinator's waited-child counters. A reused worker
session contributes once, even when it runs several assignments. Saved-result
retransmission adds actual traffic but no compute; failed physical work still
contributes. Multi-rank adapters avoid repeating shared capacity per rank.

The runtime owns worker launch, task completion and settlement. Worker observations
contribute once after settlement and remain separate from task results, checkpoints
and receipts. Missing or invalid observations mark affected rollups incomplete
without invalidating a task result. If a worker execution remains unresolved, the
report omits participant duration and CPU average because containment in the
participant window cannot be established. Valid CPU and capacity-time totals remain
useful. Statistics does not authorize publication, retry, recovery or cleanup.

Resource statistics follows the admitted singleton task-worker path across
supported launchers. External/attach, grouped, and distributed execution require
runtime-owned execution identities, attributable measurement scopes, artifact
access, and lifecycle evidence. Rank-zero observations alone do not describe a
multi-node Slurm job: its capacity is unavailable and CPU attribution is
incomplete.

## User interface

The `nvflare resources` command inspects a finalized job, a participant, or a
study:

```bash
nvflare resources --job JOB_ID
nvflare resources --job JOB_ID --site site-1
nvflare resources --job JOB_ID --study STUDY_NAME
nvflare resources --study STUDY_NAME
nvflare resources --job JOB_ID --format json
nvflare resources --schema
```

A job query searches the default study unless `--study` specifies another study.
The command presents capacity-time and utilization per participant, including in
the selected-site view. Participant detail labels the raw snapshot **Job-process
startup capacity**. For task jobs, this describes the coordinator rather than an
inventory of its workers. GPU counts are integers; CPU quotas, average CPU use
and capacity-time may be fractional. Text uses at most two decimals without
trailing zeros, with `<0.01` for small positive values. JSON uses native numbers
and preserves the measurement quality markers; text rounding does not change
stored values.

## Related architecture

The existing [task worker lifecycle](task_worker_lifecycle.md) defines worker
ownership, coordinator/compute budgets, settlement, and result receipts.
Resource statistics must observe those contracts without changing them.
The [job launcher and handle design](job_launcher_and_job_handle.md) describes
backend lifecycle ownership. The detailed reporting page includes a complete
participant example and the stored/query contracts. The machine-readable schema,
validators and generated examples must follow the same contract.
