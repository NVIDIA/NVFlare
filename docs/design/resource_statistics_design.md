# Resource statistics design

## Purpose and status

This proposal defines a final, best-effort account of the resources visible to
a job, its recorded CPU work, retained file sizes, and messaging payload. It
lets users inspect a participant or compare retained jobs without installing
a monitoring service or changing their application.

**Status: design proposal.** This document does not add resource collection,
report schemas, or CLI commands to main. Implementation and qualification will
follow separately. The resident-job contract below establishes the measurement
semantics; **Task lifetime** describes the required extension for disposable
workers. The task-aware report layout and supported deployment matrix must be
finalized with that implementation. Each claimed execution profile needs
validation on the final integrated revision.

## Measurement contract

A participant is the server or a client site. The proposed resident-job report
has one shared `duration_seconds`, separate `resource_capacity` and `resource_utilization`
groups, and optional quality information on each measurement. This abbreviated
example omits report identity and optional hardware descriptions:

```json
{
  "duration_seconds": "10",
  "resource_capacity": {
    "cpu": {"groups": [{"units": "2"}]},
    "memory": {"bytes": "2147483648"},
    "gpu": {"groups": [{"kind": "cuda_device_unclassified", "count": "1"}]}
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
- **CPU work:** user plus system CPU time of the job process and descendants
  included by the operating system after waiting. Arbitrary unwaited descendants
  are not guaranteed covered.
- **Duration:** one monotonic elapsed window shared by CPU work and the startup
  capacity observation, including application waiting.
- **Retained content:** bounded sum of regular-file lengths in the run directory
  at finalization. Symlinks and the top-level report directory are excluded.
  This is logical content size, not physical disk occupancy or peak storage.
- **Message traffic:** separate sent/received payload-byte counters from the job
  Cell and its managed trainers, including control payloads and repeated
  transfers. Shared parent-process traffic and network headers are excluded.
- **Workspace filesystem:** visible filesystem capacity at finalization,
  available in participant detail; not storage reserved for the job.

`resource_utilization` groups recorded work, content, and traffic; it does not
claim GPU busy time or RAM utilization. Measurement objects retain their own
quality beside their values. For example, `retained_content.bytes` can carry a
partial-scan explanation without affecting CPU results. Numeric strings preserve
precision; field names identify units.

A query derives capacity-time from capacity × duration. The example yields
20 CPU-unit seconds and 10 CUDA-device seconds. Average CPU use is 2 CPU
seconds / 10 seconds = 0.2 logical CPUs. Shared hardware can appear in several
reports, so totals are observations across executions, not unique hardware,
reservations, billing, or GPU utilization. Filesystem capacity is never summed
or multiplied by duration.

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

Driver enumeration can initialize CUDA in its calling process. A bounded
fresh-process probe is the recommended approach to avoid altering application
CUDA initialization. It must inherit the execution's finalized visibility and
limits, and pass timeout, cleanup, and application lifecycle checks before
adoption.

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
workspace archive to answer a small query; representative latency, temporary
storage, and concurrency measurements are release checks.

## Task lifetime

With `execution_lifetime="task"`, a resident Client Job coordinates disposable
workers. Its CPU/GPU budget can differ from each worker's budget. A coordinator
with zero GPUs can run a worker with one GPU, release it, and continue uploading
the result. The coordinator's capacity multiplied by the job duration cannot
describe that execution.

The planned extension observes each worker in its actual execution environment
and folds its bounded report into the participant report. It preserves one
duration per execution, shared by that execution's CPU and capacity measurements.
It keeps coordinator and worker capacity distinguishable, derives worker
capacity-time using worker durations, and counts each physical attempt once.
Saved-result retransmission adds actual messaging traffic but no new compute.
Failed work remains resource use even when its result is never accepted.

CPU ownership must be explicit: local Process workers may already contribute
to the coordinator's waited-child counters, whereas native Kubernetes/Slurm
workers do not. The proposed task-mode accounting uses coordinator self CPU
plus worker CPU, without also adding those workers through coordinator child
counters. Resident-job accounting retains the self-plus-waited-descendants boundary
defined above.

Worker reports remain optional measurement artifacts, independent of successful
task completion, checkpoint promotion, receipts, and confirmed resource
settlement. Missing reports produce gaps. Statistics must never authorize
publication, retry, or cleanup. Retained content is scanned once at participant
finalization; per-task scans would count the same workspace repeatedly.

The shared task runtime is the integration point. Process CPU and Kubernetes
are the first integration targets; Process GPU and single-node Slurm coverage
depends on the corresponding runtime support and qualification. Docker task
accounting is outside this proposal's initial scope.
External-process/attach task execution, arbitrary detached descendants, and
multi-node task accounting require their own supported runtime contracts.
This extension requires a coordinated stored-report/schema, query, CLI, and
documentation update before task statistics can be advertised as supported.

## Proposed user interface and release checks

These commands specify the intended interface; they are not enabled by this
documentation change.

```bash
nvflare resources --job JOB_ID
nvflare resources --job JOB_ID --site site-1
nvflare resources --study STUDY_NAME
nvflare resources --job JOB_ID --format json
nvflare resources --schema
```

The proposed command serves people and agents. Capacity and utilization have
separate tables. Counts display as integers; CPU quotas, average CPU use, and
capacity-time can be fractional. Text uses at most two decimal places without
trailing zeros, with `<0.01` for small positive values. JSON preserves precision.

Before merging the implementation, require report/derived-total reconciliation,
missing and malformed data tests, duplicate/retry isolation, real job completion and failure paths,
concurrent-job checks, authorization/archive bounds, and documentation/example
consistency. Qualify CPU/GPU visibility, worker teardown, application CUDA
initialization, and overhead on the exact release candidate. Task-mode claims
also require distinct coordinator/worker budgets and proof of no CPU or retry
double counting. Standalone simulator collection and unqualified launchers are
not implied by this design.

## Related architecture

The existing [task worker lifecycle](task_worker_lifecycle.md) defines worker
ownership, coordinator/compute budgets, settlement, and result receipts.
Resource statistics must observe those contracts without changing them.
The [job launcher and handle design](job_launcher_and_job_handle.md) describes
backend lifecycle ownership. The implementation should add the public report
reference, generated examples, and QA acceptance guide alongside its code.
