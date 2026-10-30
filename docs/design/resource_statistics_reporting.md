# Resource statistics: reports, delivery and queries

This page defines the resource-statistics records, delivery, and query behavior
introduced in the [overview](resource_statistics_design.md). The
[measurement design](resource_statistics_measurement.md) explains how the values
are obtained.

## Records and ownership

The design defines three public record kinds. Each `kind` string in the table
is prefixed by `nvflare.resource_stats.`, and every record has a `schema_version`
field.

| Kind | Required top-level fields, in addition to version and kind | Owner |
| --- | --- | --- |
| `participant_summary` | `job_id`, `participant_name`, `reported_at`, `resource_capacity`, `resource_time`, `resource_utilization`, `workspace_filesystem`; optional `duration_seconds` | The participant's parent constructs identity and timestamp around the job process's observations. |
| `resource_summary` | `job_id`, `participants` | The root server builds the expected participant roster and copies accepted measurements into it. |
| `study_summary` | `selection`, `generated_at`, `coverage`, `jobs`, `totals` | The authorized query derives a view of accessible jobs; it is not a stored report. |

The `participant_name` is the existing logical site name, or `server` for the
server slot. The root server assigns `role: "server"|"client"` from its roster;
the child process cannot claim a role. A job-process handoff contains
measurements but does not supply trusted job or site identity. The `reported_at`,
server-owned `received_at`, and `generated_at` fields are UTC RFC 3339
timestamps. They support diagnostics and are not used to calculate durations
across hosts. Elapsed time comes from a local monotonic clock.

Job IDs match `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`. Participant names use the same
alphabet, are at most 128 characters long, and may also start with `_` or `-`.
Study names are at most 63 characters long and use lowercase alphanumeric
characters, with `_` or `-` allowed internally. They begin and end with an
alphanumeric character.
Timestamps require the `Z` UTC suffix and allow at most nine fractional-second
digits.

The following participant record uses a placeholder for the schema version:

```json
{
  "schema_version": "1.0",
  "kind": "nvflare.resource_stats.participant_summary",
  "job_id": "job-42",
  "participant_name": "site-1",
  "reported_at": "2026-10-29T12:00:10Z",
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
  },
  "workspace_filesystem": {"capacity_bytes": "107374182400"}
}
```

### Value and quality rules

Quantities are unsigned plain decimal **strings**, not JSON numbers. They may
have at most 39 integer digits and nine fractional digits; signs, exponents, NaN,
and infinity are not accepted. Byte and device counts must be integers, while
capacity-time products such as `byte_seconds` may be fractional. Canonical output
removes trailing fractional zeros, for example converting `"2.0"` to `"2"`.

Products and sums use decimal arithmetic rather than binary floating-point.
Values and sums cannot exceed `2^128 - 1`; capacity-memory, per-device-memory,
and filesystem bytes are limited to `2^64 - 1`. CPU quota ratios are rounded down
to nine fractional digits so that rounding cannot enlarge a capacity bound. At
serialization, other fractional quantities are quantized to nine digits using
round-half-even, trailing fractional zeros are removed, and the result is checked
again.

A group is a category of capacity with compatible hardware descriptions. It
preserves distinctions when observations from different executions are combined.
For example, suppose two participants report these observations:

| Participant | CPU capacity | CUDA capacity | Duration |
| --- | --- | --- | ---: |
| Site A | 4 units of CPU model A | 1 full GPU | 60 seconds |
| Site B | 8 units of CPU model B | 1 MIG compute instance | 10 seconds |

The job total keeps a separate capacity-time group for each kind of hardware:

| Hardware group | Capacity-time calculation | Result |
| --- | --- | ---: |
| CPU model A | 4 units × 60 seconds | 240 CPU-unit-seconds |
| CPU model B | 8 units × 10 seconds | 80 CPU-unit-seconds |
| Full GPU | 1 device × 60 seconds | 60 device-seconds |
| MIG compute instance | 1 instance × 10 seconds | 10 instance-seconds |

The CPU groups show which models supplied the observed capacity-time; a single
320 CPU-unit-second total hides that distinction. This grouping assumes each CPU
capacity is attributable to its model. When a shared quota cannot be attributed by
model, the report keeps an aggregate group instead. The GPU groups remain separate:
a MIG instance is not treated as equivalent to a full GPU. An operator comparing
this job with another that observed one full GPU for 10 seconds can see that this
job observed a full GPU for longer and select it for further investigation. These
values do not measure CPU or GPU busy time or prove exclusive allocation.

CPU groups may include `model` and `architecture`. GPU groups use
`kind: "full_gpu"|"mig_compute_instance"|"cuda_device_unclassified"` and may
include `model` and `memory_bytes`; only MIG groups may include `mig_profile`.
CPU groups are consolidated by `(model, architecture)`, and GPU groups by
`(kind, model, memory_bytes, mig_profile)`. A missing description remains
distinct from a known one, so the implementation does not guess the hardware
class. A successful observation of zero visible GPUs is represented as
`"gpu": {"groups": []}`, rather than as unavailable or as a group with a zero
count.

Successful measurement objects omit `status` and `issues`; explicit success
strings such as `reported` or `accepted` are not serialized. Incomplete or
unavailable measurements use the following representations:

| Measurement state | Representation |
| --- | --- |
| Complete | The required value fields are present and quality fields are omitted. A measured zero is valid. |
| Partial | `status: "partial"` accompanies usable value fields and nonempty `issues`. |
| Unavailable | `status: "unavailable"` and nonempty `issues` are present; usable value fields are omitted. |

These rules apply to participant measurements, including copies in the job's
participant list. Derived job and study **utilization totals** use `status`
without `issues`; the explanations for their quality remain in participant
measurements and coverage. Aggregated `resource_time` retains `issues`. An
aggregate utilization object cannot be copied back into a participant report
without the required source-quality fields.

For example, a bounded file scan that stopped early can retain its subtotal:

```json
{"status": "partial", "issues": ["observation_incomplete"], "bytes": "1024"}
```

The issue codes are `observation_incomplete`, `attribution_incomplete`,
`unsupported`, and `malformed_source`. Partial values use the first two codes;
partial traffic measurements use `observation_incomplete`. Each validator accepts
only codes meaningful for its measurement. Issues must not contain raw exception
text or environment details.

Quality is tracked independently for `resource_capacity`, `resource_time`, and
each of `cpu_consumed`, `retained_content`, and `message_traffic`. Quality is not
attached to the `resource_utilization` container itself. Complete capacity and time
objects contain CPU, memory, and GPU dimensions; partial objects retain the known
dimensions. For example, an unavailable GPU probe does not discard known CPU or
RAM values. Filesystem capacity appears only in participant detail and is either
complete or unavailable; it has no partial subtotal. When no valid containing
participant window exists, `duration_seconds` is omitted, although CPU and
capacity-time values may still be useful without an average.

## Terminal transport and validation

The `REPORT_JOB_OUTCOME` request carries an optional `resource_report` field
with the value `{"participant_summary": <UTF-8 JSON bytes>}`. The outcome reply
carries a separate `resource_report_status`. Rejecting statistics must not reject
or change the training outcome. The
[client outcome path](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/private/fed/client/client_executor.py)
and [server handler](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/private/fed/server/fed_server.py)
remain the transport and authentication boundaries.

The parent reads one bounded terminal handoff, validates individual measurements,
adds the job and site identity, and serializes the participant record once.
Invalid child measurements become `unavailable` with `malformed_source`;
malformed identity or envelope data is not repaired to produce an accepted
participant. The private handoff is `resource_stats/staging/terminal_handoff.json`
with internal version `"4"`. It is consumed and removed before public archival.

The server derives the client name from the authenticated, registered client,
not from the payload or an arbitrary message-origin header. It creates per-job
state with the server and deployed client roster. Reports cannot create jobs,
participants, filesystem paths, or role assignments. Statistics acceptance runs
before the ordinary duplicate-outcome gate, allowing an outcome retry to obtain
a statistics result without applying the job outcome twice.

Validate in this order:

1. Bind the request to the active job and expected, non-disabled sender slot.
2. Check the wrapper and raw byte limit before decoding. Decode UTF-8 JSON,
   rejecting duplicate keys, nonfinite constants, wrong kind/version and excess
   nesting. Unknown fields are rejected, not silently ignored.
3. Validate identifiers, units, quality/value combinations, group uniqueness and
   numeric bounds; require body job/site identity to match the bound slot.
4. Normalize decimal spellings, object keys and set-like group/issue ordering.
   Recheck the canonical byte limit and compare with any accepted record.
5. Under the per-job lock, recheck that acceptance is still open and the slot is
   eligible. Retain immutable canonical bytes and server-owned receipt time.

Hardware descriptions are controlled labels rather than host identifiers.
Reports must not emit hostnames, addresses, serials, UUIDs, PCI identifiers, CUDA
visibility masks, raw CPU topology, or raw cgroup paths. Validation limits
disclosure and rejects malformed input, but it cannot attest to the accuracy of
a site's self-report.

### Limits and retries

The following implementation bounds are not additional configuration
options. A producer must stop before exceeding them, and a receiver must enforce
them independently.

| Bound | Limit |
| --- | --- |
| Participant JSON, before and after canonicalization | 1 MiB |
| Accepted canonical participant bytes retained per job | 64 MiB |
| Job or study summary JSON | 64 MiB |
| JSON nesting | 32 levels |
| Expected participants per job; selected jobs per study query | 10,000 each |
| CPU or GPU groups per object; visible GPU instances | 4,096 each |
| Total CPU capacity units in a snapshot | 1,048,576 |
| ZIP metadata reads while constructing the member index | 4 MiB |

One immutable accepted record occupies a `(job_id, participant_name)` slot:

| Condition | `resource_report_status` | Effect |
| --- | --- | --- |
| First valid record while open | `accepted` | The server retains the record once. |
| Semantically equivalent retry, including after closure | `duplicate` | The retry adds no values or receipt entry. |
| Different valid record for an accepted open slot | `conflict` | The server keeps the original. |
| New or changed record after closure | `too_late` | The archive snapshot is unchanged. |
| Invalid content while open | `invalid` | The server keeps any accepted record; otherwise, it records a gap that a valid arrival may repair before closure. |
| No statistics on an otherwise valid outcome | `not_provided` | Ordinary outcome processing continues. |
| Unknown/disabled slot or no current job ledger | `not_expected` | The server does not allocate state for the request. |
| Capacity or internal acceptance failure | `server_error` | The job outcome and accepted subtotal are unchanged. |

The normal terminal-outcome mechanism may retransmit the same bytes, including
the original `reported_at`. Rebuilding a record with a new timestamp is not an
equivalent retry. There is no separate statistics sender, retry journal, or
acknowledgement protocol. Older peers may omit the report or its reply status;
this indicates absent statistics rather than an outcome failure. An older archive
without the bundle is unavailable, not a zero measurement.

## Server ledger and archive publication

The coordinator maintains state and a lock for each job, plus a registry lock
held briefly for lookup. Parsing occurs outside the per-job lock, and acceptance
rechecks closure afterward. Slow finalization of one job must not block unrelated
jobs through a global filesystem-I/O lock. The ledger retains only bounded
canonical report bytes, rather than an additional decoded copy for every
participant.

```text
OPEN: accept one record per trusted slot
  -> CLOSED: freeze roster and accepted bytes; later changes are too_late
  -> FINALIZED: participant files and summary marker written
  -> ARCHIVED: existing workspace archive saved
```

At closure, each roster entry contains either an accepted value or a gap. An
accepted entry contains `participant_name`, server-owned `role` and `received_at`,
optional `duration_seconds`, and copies of `resource_capacity`, `resource_time`,
and `resource_utilization`. It omits a success status. The `reported_at` and
`workspace_filesystem` fields remain only in participant detail. A gap contains
the name, role, and `status: "missing"|"invalid"|"disabled"`; an invalid gap also
has `received_at`. Here, `disabled` means the parent knows the participant was
not launched; it is not a new collection opt-out feature. Missing measurements
must not be filled with zeros or inferred from requested resources.

Publish under the server-owned run directory:

```text
resource_stats/
  participants/server.json
  participants/site-1.json
  resource_summary.json
```

Publication uses descriptor-relative, no-follow directory operations where
supported and rejects symlinked or replaced reserved directories rather than
following a job-provided path. On platforms without these operations, it rejects
existing symlinks but cannot claim equivalent protection against path races.

Publication writes only accepted participant bytes through temporary files and
atomic rename, then writes `resource_summary.json` last as the bundle commit
marker. It verifies the reserved-directory inventory before the existing
[job finalization path](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/private/fed/server/job_runner.py)
saves the normal `WORKSPACE` archive. The summary contains the self-contained
roster and measurements, without a separately trusted totals cache. Repeated
finalization returns the same committed bundle and does not accept newer
measurements.

If publication fails, the incomplete statistics bundle is excluded or discarded
and the normal job and archive failure policy continues. An optional statistics
failure must not change the training result. A crash before publication can lose
the in-memory ledger; the protocol provides neither server restart adoption nor
durable delivery before archive publication. Unpublished child files are not an
authoritative alternative to a committed server bundle.

## Authorized reads and derived totals

The Session API exposes `Session.get_job_resources(job_id, site=None)`
and `Session.get_study_resources()`. A job response contains `resource_summary`
and, when a site is selected, `participant_summary`; a study response contains
`study_summary` for the session's active study. These methods extend the existing
[Session](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/fuel/flare_api/flare_api.py) and
[admin job commands](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/private/fed/server/job_cmds.py)
rather than introducing an HTTP service. The server authenticates the caller and
applies existing job-ID, study, and submitter authorization before fetching a
workspace. A study query also checks the resources-read right for each job and
excludes denied jobs without exposing their IDs or counts.

Only finalized jobs have archive-backed data. The reader stages the normal
`WORKSPACE` through `JobDefManagerSpec.get_storage_for_download`; the design adds
neither a parallel storage database nor a sidecar-storage API. It reads the summary
member and reads a participant detail only when requested and accepted in the
roster. For each requested member, it requires exactly one matching ZIP member,
rejects encrypted, directory, or oversized members, and bounds actual decoded
reads. It also cross-checks detail identity and copied values against the summary
without extracting or scanning unrelated workspace contents.

The ZIP index has its own limit because a small JSON member does not imply small
archive metadata. Remote job stores may still download the entire workspace before
the reader opens it. The design reuses existing download cleanup and timeout
policy; it does not promise low network cost for a small query or cancellation of
server I/O when the CLI times out.

Job totals are derived from accepted entries independently for each measurement.
They sum stored `resource_time` and utilization without multiplying startup
capacity by participant duration again. CPU groups and GPU/MIG kinds retain their
grouping keys. Startup capacity, filesystem capacity, and participant durations
are not summed. Any expected gap makes a usable job subtotal partial; if there are
no usable contributors, the result is unavailable rather than zero. Average CPU
use is a participant's CPU seconds divided by its positive containing duration,
not a percentage of host capacity.

A study row contains `job_id`, `job_name`, `job_status`, and
`resource_data: "included"|"unavailable"|"nonterminal"`. Only included rows have
derived `totals`. Coverage records `selected_jobs`, `included_jobs`,
`unavailable_jobs`, and `nonterminal_jobs` as numeric strings. The study
`selection` contains `study_name`, and `generated_at` is the query time. Study
totals sum only included-job totals and preserve their per-field quality.
Unavailable and nonterminal jobs remain explicit coverage gaps, so the
**included-jobs subtotal** is not a claim about all selected jobs.

Numeric overflow is never clamped or represented as infinity. During collection,
valid independent subtotals are retained and gaps are marked. If a requested job
or study sum exceeds the contract, the derived result is rejected rather than
published as a misleading complete total.

The CLI uses the existing success/error envelope. Job JSON data contains
`selection: {job_id, site}`, `summary`, and an optional `participant`; study JSON
data contains `selection: {study}` and `summary`. The CLI rejects `--site` without
`--job`; a job query may also specify its study. The `--schema` option describes
the command interface, not the report-format schema. Authorization,
nonterminal-job, and unavailable-data errors remain distinguishable. Text output
rounds only for display; JSON retains the numeric strings and quality markers
described above.

## Compatibility and failure boundary

The report does not provide a fallback that reinterprets startup capacity as a
task rollup, and unsupported formats are not silently upgraded. Unsupported or absent
archives are reported as unavailable; deployment may still complete normally when
a peer does not send statistics.

The server trusts authenticated identity and validates report structure, but it
cannot verify the physical truth of reported numbers. Atomic files prevent partial
publication; they do not provide hardware attestation or recovery after loss of
the in-memory ledger. Final reports are immutable observations of completed work.
Scheduling, live monitoring, billing, restart adoption, and result or checkpoint
acceptance remain outside this protocol.
