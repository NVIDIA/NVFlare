# Resource statistics design

## Purpose

Resource statistics provides a final, best-effort account of the resources visible
to a job, its recorded CPU work, retained file sizes, and message payloads. It
lets users inspect a participant or compare retained jobs without installing a
monitoring service or changing their applications.

Collection follows the runtime's execution boundaries and admission checks. Each
execution mode provides attributable measurements and lifecycle ownership; resource
statistics does not select or enable backends.

## Detailed design

This page introduces the contract. The following pages describe its
component interfaces, algorithms, and lifecycle decisions:

1. [Measurement and accounting](resource_statistics_measurement.md): the APIs,
   limits, execution windows and formulas used to obtain each measurement.
2. [Reports, delivery and queries](resource_statistics_reporting.md): record
   formats, validation, deduplication, archive publication and authorized reads.
3. [Runtime integration](resource_statistics_implementation.md): component
   interfaces and lifecycle behavior on success, failure, cancellation and restart.

## Measurement contract

A participant is the server or a client site. Each participant has one final
rollup, a shared `duration_seconds`, and separate capacity and utilization
measurements. `resource_capacity` is the job-process startup snapshot.
`resource_time` rolls up capacity multiplied by each execution's own duration;
it uses the same representation as job/study totals. Worker records stay internal.
The following abbreviated example for a resident job omits identity and optional
descriptions:

```json
{
  "duration_seconds": "10",
  "resource_capacity": {
    "cpu": {"groups": [{"units": "2"}]},
    "memory": {"bytes": "2147483648"},
    "gpu": {"groups": [{"kind": "cuda_device_unclassified", "count": "1"}]}
  },
  "resource_time": {
    "cpu": {"groups": [{"unit_seconds": "20"}]},
    "memory": {"byte_seconds": "21474836480"},
    "gpu": {"groups": [{"kind": "cuda_device_unclassified", "instance_seconds": "10"}]}
  },
  "resource_utilization": {
    "cpu_consumed": {"cpu_seconds": "2"},
    "retained_content": {"bytes": "1024"},
    "message_traffic": {"sent_bytes": "2048", "received_bytes": "4096"}
  }
}
```

- **CPU capacity:** allowed logical CPUs, reduced by readable CPU-time quotas.
  CPU capacity can be fractional.
- **RAM capacity:** physical RAM reduced by readable memory limits; not free
  memory or peak usage.
- **CUDA capacity:** devices visible for CUDA execution to the observed process.
  Device and MIG-instance counts are integers. Visibility does not prove use or
  exclusive allocation.
- **CPU work:** user plus system CPU time from disjoint job/worker process
  scopes and their normally waited descendants. Task workers are folded in once;
  arbitrary unwaited descendants are not guaranteed covered.
- **Duration:** one participant window, including waiting. Each execution has
  one shared CPU/capacity window internally; worker durations are not added to
  participant elapsed time.
- **Retained content:** bounded sum of regular-file lengths in the run directory
  at finalization. Symlinks and the top-level report directory are excluded.
  This is logical content size, not physical disk occupancy or peak storage.
- **Message traffic:** separate sent/received payload-byte counters from the job
  Cell, task workers and managed trainers, including control payloads and repeated
  transfers. Shared parent-process traffic and network headers are excluded.
- **Workspace filesystem:** visible filesystem capacity at finalization,
  available in participant detail; not storage reserved for the job.

`resource_utilization` groups recorded work, content, and traffic; it does not
claim GPU busy time or RAM utilization. Measurement objects retain their own
quality beside their values. For example, `retained_content.bytes` can carry a
partial-scan explanation without affecting CPU results. Numeric strings preserve
precision; field names identify units.

The collector computes capacity-time for each execution and combines it in
`resource_time`. Job/study queries sum those participant values without multiplying
startup capacity by duration again. The example yields 20 CPU-unit seconds and
10 device-seconds; average CPU use is 2/10 = 0.2 logical CPUs. Shared hardware
can be visible to several executions, so totals are observations, not unique
hardware, reservations, billing or GPU busy time. Filesystem capacity is never
summed or multiplied by duration.

Sent and received counters are summed separately, including local endpoint
traffic. A successful local send is not proof of receipt. Their sum is not a
deduplicated network-byte total. Direct in-process delivery is excluded.

## Collection and delivery

```text
Job process: prepare workspace -> observe capacity and open counters
             -> run application -> finish counters and scan files
Parent:      consume terminal handoff -> release resources -> report outcome
Server:      validate participant report -> finalize workspace archive
User:        authorized job/site/study query -> derive totals -> render
```

Collection starts after existing workspace preparation and before
application construction in the client/server job entry points. It does not
change normal Python startup, resource assignment, or process ownership. CPU
and memory probes use existing affinity and visible cgroup limits. CUDA
enumeration uses the system Driver; optional NVML only improves device
classification. No mandatory PyTorch dependency is added.

Driver enumeration can initialize CUDA in its calling process. To avoid altering
the application's CUDA initialization, it runs in a fresh helper process with a
bounded lifetime under the existing process owner. The helper inherits the
execution's finalized visibility and limits.

The parent supplies participant identity and attaches the report to the existing
`REPORT_JOB_OUTCOME` request. The server retains accepted reports until
finalization, writes participant files and then `resource_summary.json` under
`resource_stats/`, and includes them in the normal workspace archive. Queries
use existing per-job authorization. There is no separate statistics database,
periodic sender, or live resource-monitoring service.

## Quality, failure, and cost

Successful measurements omit `status`. `partial` retains usable values with a
coverage explanation; `unavailable` has no usable value. An unreadable limit
does not erase another usable observation. A missing CUDA Driver is unavailable;
a successful observation of zero devices is zero.

Reports are untrusted site self-reports. Validation checks sender/job identity,
shape, numeric limits, and duplicates, but does not attest hardware. Equivalent
duplicates do not add measurements; conflicting replacements are rejected.
Missing participants remain explicit gaps. A server crash before archive
publication or a killed job process can lose statistics. Collection failures
must not change the training outcome, delay resource release indefinitely, or
create a second recovery protocol.

Filesystem scans and requested archive members are bounded. Scan deadlines are
cooperative and cannot interrupt a blocked filesystem call. ZIP indexing has a
separate metadata-read bound. Remote job stores may still transfer the whole
workspace archive to answer a small query. Query latency and temporary storage
can therefore depend on the archive size, not just the statistics size.

## Task lifetime

With `execution_lifetime="task"`, the resident Client Job coordinates workers
whose resources and lifetimes can differ from its own. Statistics are collected
inside each worker's actual environment and rolled into the participant's existing
CPU-work and traffic fields. Retained content is scanned once at finalization.
There is no public task or worker list.

For example, a 60-second coordinator with zero GPUs and 2 CPU seconds launches
one-GPU workers lasting 10 and 5 seconds, using 8 and 4 CPU seconds. The participant
reports 15 GPU-device-seconds, 14 CPU seconds and a 60-second duration. Average
CPU use is about 0.23 logical CPUs. It does not report two GPUs or multiply worker
capacity by the coordinator's lifetime.

Task-mode CPU uses coordinator self CPU plus disjoint worker CPU; workers are
not also added through the coordinator's waited-child counters. A reused worker
session contributes once, even when it runs several assignments. Saved-result
retransmission adds actual traffic but no compute; failed physical work still
contributes. Multi-rank adapters avoid repeating shared capacity per rank.

Worker reports are optional measurement artifacts, separate from task completion,
checkpoint promotion, receipts and resource settlement. Missing observations mark
the affected rollups partial or unavailable. Statistics never authorizes publication,
retry, recovery or cleanup. If the participant window cannot be shown to contain
all included work, the report omits its duration and CPU average. Valid CPU and capacity-time totals
remain useful.

Resource statistics uses a common accounting path across supported launchers.
External/attach, grouped, and distributed execution use this path when their
runtime integrations provide execution identity, attributable measurement scopes,
artifact access, and lifecycle ownership.

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
the selected-site view. Participant detail labels the raw snapshot **Job-process startup capacity**; for task jobs
this is coordinator capacity, not a worker inventory. GPU counts are integers;
CPU quotas, average CPU use and capacity-time may be fractional. Text uses at
most two decimals without trailing zeros, with `<0.01` for small positive values.
JSON preserves precision and existing measurement quality markers.

## Related architecture

The existing [task worker lifecycle](task_worker_lifecycle.md) defines worker
ownership, coordinator/compute budgets, settlement, and result receipts.
Resource statistics must observe those contracts without changing them.
The [job launcher and handle design](job_launcher_and_job_handle.md) describes
backend lifecycle ownership. The detailed reporting page includes a complete
participant example and the stored/query contracts. The machine-readable schema,
validators and generated examples must follow the same contract.
