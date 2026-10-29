# Resource statistics: reports, delivery and queries

Public resource reports use **schema format 4.0**. This page defines the record
and delivery contract introduced in the [overview](resource_statistics_design.md).
The [measurement design](resource_statistics_measurement.md) defines how values
are obtained; the [implementation design](resource_statistics_implementation.md)
maps the components to their lifecycle hooks.

## Records and ownership

There are three public record kinds. The `kind` strings below are prefixed by
`nvflare.resource_stats.`; each record has `schema_version: "4.0"`.

| Kind | Required top-level fields, in addition to version and kind | Owner |
| --- | --- | --- |
| `participant_summary` | `job_id`, `participant_name`, `reported_at`, `resource_capacity`, `resource_time`, `resource_utilization`, `workspace_filesystem`; optional `duration_seconds` | The participant's parent constructs identity and timestamp around the job process's observations. |
| `resource_summary` | `job_id`, `participants` | The root server builds the expected participant roster and copies accepted measurements into it. |
| `study_summary` | `selection`, `generated_at`, `coverage`, `jobs`, `totals` | The authorized query derives a view of accessible jobs; it is not a stored report. |

`participant_name` is the existing logical site name, or `server` for the server
slot. The root server assigns `role: "server"|"client"` from its roster; the child
cannot claim a role. A job-process handoff contains measurements, not trusted
job/site identity. `reported_at`, server-owned `received_at`, and `generated_at`
are UTC RFC 3339 timestamps. They are for diagnostics, not cross-host duration
arithmetic; elapsed time comes from a local monotonic clock.

Job IDs match `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`. Participant names are at most
128 characters from the same alphabet and may also start with `_` or `-`.
Study names are lowercase alphanumeric with internal `_`/`-`, at most 63
characters, beginning and ending alphanumeric. Timestamps allow at most nine
fractional-second digits and require the `Z` UTC suffix.

An example complete participant record is:

```json
{
  "schema_version": "4.0",
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

Quantities are unsigned plain decimal **strings**, not JSON numbers. Accept at
most 39 integer digits and nine fractional digits, no sign, exponent, NaN or
infinity. Byte counts and device counts must be integer-valued; capacity-time
products such as `byte_seconds` may be fractional. Canonical output turns
`"2.0"` into `"2"`. Compute products and sums with decimal arithmetic, not binary
floating-point. Values and sums cannot exceed `2^128 - 1`; capacity-memory,
per-device-memory and filesystem bytes are limited to `2^64 - 1`.
Round CPU quota ratios down to nine fractional digits so rounding cannot enlarge
a capacity bound. At serialization, quantize other fractional quantities to nine
digits using round-half-even, remove trailing zeros, and check the result again.

CPU groups may carry `model` and `architecture`. GPU groups use
`kind: "full_gpu"|"mig_compute_instance"|"cuda_device_unclassified"` and may carry
`model`, `memory_bytes`, and, only for MIG, `mig_profile`. Consolidate CPU groups
by `(model, architecture)` and GPU groups by `(kind, model, memory_bytes,
mig_profile)`. Missing descriptions are distinct from known ones; do not guess a
hardware class. Zero visible GPUs is `"gpu": {"groups": []}`, not an unavailable
observation or a group with count zero.

Successful measurement objects omit `status` and `issues`. Explicit success
strings such as `reported` or `accepted` are not serialized. Exceptions use:

| Measurement state | Representation |
| --- | --- |
| Complete | Required value fields, no quality fields. A measured zero is valid. |
| Partial | `status: "partial"`, usable value fields, and nonempty `issues`. |
| Unavailable | `status: "unavailable"` and nonempty `issues`; no usable value fields. |

This table describes participant measurements, including those copied into the
job's participant list. Derived job/study **utilization totals** use `status`
without `issues`; their source explanations remain in participant measurements
and coverage. Aggregated `resource_time` retains `issues`. Do not copy an
aggregate utilization object back into a participant report without its required
source-quality fields.

For example, a bounded file scan that stopped early can retain its subtotal:

```json
{"status": "partial", "issues": ["observation_incomplete"], "bytes": "1024"}
```

Issue codes are `observation_incomplete`, `attribution_incomplete`, `unsupported`
and `malformed_source`. Partial values use the first two; traffic partials use
`observation_incomplete`. Each validator admits only codes meaningful for that
measurement. Do not store raw exception text or environment details as issues.

Quality belongs to `resource_capacity` and `resource_time` independently, and to
each of `cpu_consumed`, `retained_content` and `message_traffic`, not their
`resource_utilization` container. Complete capacity/time objects have CPU, memory
and GPU dimensions; partial ones keep the known dimensions. An unavailable GPU
probe therefore does not discard known CPU/RAM values. Filesystem capacity is a
detail-only observation and supports complete or unavailable, not a subtotal.
Omit `duration_seconds` when no valid containing participant window exists;
CPU/capacity-time values may still be useful without an average.

## Terminal transport and validation

The `REPORT_JOB_OUTCOME` request carries an optional `resource_report` field.
Its value is `{"participant_summary": <UTF-8 JSON bytes>}`. The outcome
reply carries a separate `resource_report_status`; statistics rejection must not
reject or change the training outcome. The [client outcome path](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/private/fed/client/client_executor.py)
and [server handler](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/private/fed/server/fed_server.py)
remain the transport and authentication boundaries.

The parent reads one bounded terminal handoff, validates individual measurements,
adds its job/site identity, and serializes a participant record once. Invalid
child measurements become `unavailable` with `malformed_source`; malformed
identity/envelope data is not repaired into an accepted participant. The private
handoff is `resource_stats/staging/terminal_handoff.json`, with internal version
`"4"`; it is consumed and removed before public archival.

The server derives the client name from its authenticated, registered client,
not the payload's name or an arbitrary message-origin header. Its per-job state
is created with the server and the deployed client roster. Reports cannot create
new jobs, participants, filesystem paths or role assignments. Acceptance runs
before the ordinary duplicate-outcome gate so an outcome retry can obtain a
statistics result without applying the job outcome twice.

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

Hardware descriptions are controlled labels, not host identifiers. Do not emit
hostnames, addresses, serials, UUIDs, PCI identifiers, CUDA visibility masks,
raw CPU topology or raw cgroup paths. Validation limits disclosure and malformed
input; it cannot attest the truth of a site's self-report.

### Limits and retries

These are format-4 implementation bounds, not additional configuration
options. A producer must stop before exceeding them; a receiver must enforce them
independently.

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
| First valid record while open | `accepted` | Retain it once. |
| Semantically equivalent retry, including after closure | `duplicate` | No additional values or receipt entry. |
| Different valid record for an accepted open slot | `conflict` | Keep the original. |
| New or changed record after closure | `too_late` | Do not alter the archive snapshot. |
| Invalid content while open | `invalid` | Keep any accepted record; otherwise record a gap that a valid arrival may repair before closure. |
| No statistics on an otherwise valid outcome | `not_provided` | Ordinary outcome processing continues. |
| Unknown/disabled slot or no current job ledger | `not_expected` | Do not allocate state for the request. |
| Capacity or internal acceptance failure | `server_error` | Do not change the job outcome or accepted subtotal. |

The normal terminal-outcome mechanism may retransmit the same bytes, including
the original `reported_at`; rebuilding a record with a new timestamp is not an
equivalent retry. There is no separate statistics sender, retry journal or
acknowledgement protocol. Old peers
may omit the report or its reply status. Treat that as absent statistics, not an
outcome failure. An old archive without this bundle is unavailable, not a zero
measurement. Do not interpret an unsupported version as format 4.0.

## Server ledger and archive publication

The coordinator has one per-job state and lock, plus a short registry
lock for lookup. Parsing occurs outside the per-job lock; acceptance rechecks
closure after parsing. A slow finalization for one job must not block unrelated
jobs through a global filesystem-I/O lock. Keep only bounded canonical report
bytes in the ledger, not an additional decoded copy for every participant.

```text
OPEN: accept one record per trusted slot
  -> CLOSED: freeze roster and accepted bytes; later changes are too_late
  -> FINALIZED: participant files and summary marker written
  -> ARCHIVED: existing workspace archive saved
```

At closure, each roster entry is either an accepted value or a gap. An accepted
entry contains `participant_name`, server-owned `role` and `received_at`, optional
`duration_seconds`, and copies of `resource_capacity`, `resource_time` and
`resource_utilization`. It omits success status. `reported_at` and
`workspace_filesystem` remain in participant detail only. Gaps contain name,
role, and `status: "missing"|"invalid"|"disabled"`; invalid also has `received_at`.
Here `disabled` means the parent knows that participant was not launched, not a
new collection opt-out feature. Missing measurements must never be filled with
zeros or inferred from requested resources.

Publish under the server-owned run directory:

```text
resource_stats/
  participants/server.json
  participants/site-1.json
  resource_summary.json
```

Use descriptor-relative, no-follow directory operations where supported. Reject
symlinked/replaced reserved directories rather than following a job-provided
path. On platforms without these operations, reject existing symlinks but do not
claim equivalent protection against path races.

Write only accepted participant bytes through temporary files and atomic rename,
then write `resource_summary.json` last as the bundle commit marker. Verify the
reserved-directory inventory before the existing
[job finalization path](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/private/fed/server/job_runner.py) saves the
normal `WORKSPACE` archive. The summary contains the self-contained roster and
measurements, **not a separately trusted totals cache**. Repeated finalization
returns the same committed bundle; it does not accept newer measurements.

If publication fails, exclude/discard the incomplete statistics bundle and
continue the normal job/archive failure policy. Do not turn an optional
statistics failure into a different training result. A crash before publication
can lose the in-memory ledger; the protocol provides neither server restart
adoption nor durable delivery before archive publication. Unpublished child files
are never an authoritative alternative to a committed server bundle.

## Authorized reads and derived totals

The Session API exposes `Session.get_job_resources(job_id, site=None)` and
`Session.get_study_resources()`. A job response contains `resource_summary` and,
when a site is selected, `participant_summary`; the study response contains
`study_summary`. They extend the existing
[Session](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/fuel/flare_api/flare_api.py) and
[admin job commands](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/private/fed/server/job_cmds.py), not a new HTTP
service. Authenticate and apply the existing job-ID/study/submitter authorization
before fetching a workspace. A study query also checks the resources-read right
for each job; exclude denied jobs without exposing their IDs or counts.

Only finalized jobs have archive-backed data. Stage the normal `WORKSPACE`
through `JobDefManagerSpec.get_storage_for_download`. Do not add a parallel
storage database or a new sidecar-storage API. Read exactly the summary member;
read one participant detail only when requested and accepted in that roster.
Require one matching ZIP member, reject encrypted/directory/oversized members,
bound actual decoded reads, and cross-check detail identity and copied values
against the summary. Do not extract or scan unrelated workspace contents.

The ZIP index has its own limit because a small JSON member does not imply small
archive metadata. Remote job stores may still download the entire workspace
before this reader opens it. Reuse existing download cleanup and timeout policy;
this design does not promise small-query network cost or cancellation of server
I/O when the CLI times out.

Derive job totals from accepted entries, independently per measurement. Sum
stored `resource_time` and utilization; never multiply startup capacity by
participant duration again. CPU groups and GPU/MIG kinds retain their grouping
keys. Do not sum startup capacity, filesystem capacity or participant durations.
Any expected gap makes usable job subtotals partial; no usable contributors means
unavailable, not zero. Average CPU use is per-participant CPU seconds divided by
its positive containing duration; it is not a percentage of host capacity.

A study row has `job_id`, `job_name`, `job_status` and
`resource_data: "included"|"unavailable"|"nonterminal"`; only included rows have
derived `totals`. Coverage records `selected_jobs`, `included_jobs`,
`unavailable_jobs` and `nonterminal_jobs` as numeric strings. Study `selection`
contains `study_name`; `generated_at` is the query time. Sum included-job totals
only, preserving their per-field quality. Unavailable/nonterminal jobs remain
explicit coverage gaps but do not change an **included-jobs subtotal** into a
claim about all selected jobs.

Numeric overflow is never clamped or represented as infinity. During collection,
retain valid independent subtotals and mark gaps. If a requested job/study sum
itself exceeds the contract, reject that derived result rather than publish a
misleading complete total.

The CLI uses the existing success/error envelope. Job JSON data contains
`selection: {job_id, site}`, `summary` and optional `participant`; study JSON data
contains `selection: {study}` and `summary`. Reject `--site` without `--job`;
`--job` may also specify its study. `--schema` describes the command interface,
not the report-format schema. Authorization, nonterminal-job and unavailable-data
errors remain distinguishable. Text rendering rounds only for display; JSON
retains the numeric strings and quality markers above.

## Compatibility and failure boundary

Format 4.0 requires `resource_time` on every participant, including resident
jobs. There is no optional fallback that reinterprets startup capacity as a task
rollup. Unsupported formats are not silently upgraded. Unsupported or absent
archives are reported as unavailable; deployment may still complete normally
when a peer does not send statistics.

The server trusts authenticated identity and validates the report's structure,
not the physical truth of its numbers. Atomic files prevent partial publication;
they do not provide hardware attestation or recovery after losing the in-memory
ledger. Final reports are immutable observations of completed work. Scheduling,
live monitoring, billing, restart adoption and result/checkpoint acceptance stay
outside this protocol.
