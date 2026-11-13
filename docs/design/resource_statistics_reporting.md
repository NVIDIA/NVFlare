# Resource statistics: reports, delivery and queries

This page defines the resource-statistics records, delivery, and query behavior
introduced in the [overview](resource_statistics_design.md). The
[measurement design](resource_statistics_measurement.md) explains how the values
are obtained.

## Records and ownership

The design defines three public record kinds. Each `kind` string in the table is
prefixed by `nvflare.resource_stats.`, and every record has a `schema_version`
field. The public schema version is `"1.0"`; private handoff metadata is
versioned separately.

| Kind | Required top-level fields, in addition to version and kind | Owner |
| --- | --- | --- |
| `participant_summary` | `job_id`, `participant_name`, `reported_at`, `resource_capacity`, `resource_time`, `resource_utilization`, `issues`; optional `duration_seconds` | The participant's parent constructs identity and timestamp around the job process's observations. |
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

The following example shows a complete participant record:

```json
{
  "schema_version": "1.0",
  "kind": "nvflare.resource_stats.participant_summary",
  "job_id": "job-42",
  "participant_name": "site-1",
  "reported_at": "2026-10-29T12:00:10Z",
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
    "message_traffic": {
      "sent_bytes": 2048,
      "received_bytes": 4096
    }
  },
  "issues": []
}
```

### Value and quality rules

Quantities use native JSON numbers. Bytes and GPU counts are integers;
CPU units, durations, CPU work and capacity-time may be fractional. Numbers must
be finite and nonnegative; boolean values and numeric strings are not measurements.
Integer products and sums remain Python integers. Mixed arithmetic uses floating
point and rejects nonfinite results rather than clamping them. The wire format
adds no fixed decimal precision or artificial integer-width limit.

Capacity and capacity-time are scalar objects:
`cpu.units`, `memory.bytes`, `gpu.count`, and respectively `cpu.unit_seconds`,
`memory.byte_seconds`, `gpu.instance_seconds`. There are no CPU-model or GPU-type
groups, device-memory fields, or hardware descriptions. An observed zero is
`"gpu": {"count": 0}`; zero capacity-time is `"gpu": {"instance_seconds": 0}`.
GPU counts follow NVFlare's shared inventory and integer-mask selection rules.
`gpu.instance_seconds` records GPU-seconds, not GPU utilization or exclusive allocation.

Successful metrics omit `status`; exceptional metrics retain their own
`status: "partial"` or `"unavailable"`. Explanations appear once in the owning
scope's `issues` list as `{metric, code}` entries, sorted and deduplicated by
metric then code. A complete scope has `issues: []`.

| Measurement state | Representation |
| --- | --- |
| Complete | Value fields are present and `status` is omitted. Zero is a valid observation. |
| Partial | `status: "partial"` accompanies usable values; the owning scope names the metric's issues. |
| Unavailable | `status: "unavailable"` replaces usable values; the owning scope names the metric's issues. |

Each participant report and accepted participant row owns one source issue list.
Worker records and terminal handoffs follow the same scope rule. Derived job
and per-study-job totals own aggregate issue lists, and study totals own their
list. Aggregate explanations describe aggregate coverage; they do not copy all
participant source details. The persisted job summary and study response add no
duplicate root list around their existing rows or totals.

The following abbreviated scope fragment shows incomplete CPU attribution
with a usable subtotal. Other measurements and record identity are omitted:

```json
{
  "resource_utilization": {
    "cpu_consumed": {
      "status": "partial",
      "cpu_seconds": 12.5
    }
  },
  "issues": [
    {
      "metric": "cpu_consumed",
      "code": "attribution_incomplete"
    }
  ]
}
```

The issue codes are `observation_incomplete`, `attribution_incomplete`,
`unsupported`, and `malformed_source`. Partial values use the first two codes;
partial traffic measurements use `observation_incomplete`. Issues must not contain raw exception
text or environment details.

Quality is tracked independently for `resource_capacity`, `resource_time`, and
each of `cpu_consumed` and `message_traffic`. Quality is not
attached to the `resource_utilization` container itself. Complete capacity and time
objects contain CPU, memory, and GPU dimensions; partial objects retain the known
dimensions. For example, an unavailable GPU probe does not discard known CPU or
RAM values. When no valid containing
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

The parent reads one bounded terminal handoff, checks its launch generation,
adds the job and site identity, and serializes the participant record once.
Local producers and reducers do not repeat the server's measurement checks.
Measurements remain self-reported; validating a local file cannot establish their
authenticity. Each launch owns a private
`resource_stats/staging/terminal_handoff.<generation>.json` file with internal
version `"4"`. Its bounded generation token is checked before consumption and
never enters public records. Cleanup removes only that generation's file, so an
older writer or failed launch cannot replace or remove a newer handoff. Private
staging must be empty before public archival.

The parent binds each handoff to the launch generation it owns, establishing the
association before launch. It rejects stale or unverifiable handoffs, even when a
prior file remains in the workspace. If the terminal generation produces no valid
handoff, its observations are gaps; an earlier generation's report is never used
as a substitute.

The server derives the client name from the authenticated, registered client,
not from the payload or an arbitrary message-origin header. It creates per-job
state with the server and deployed client roster. Reports cannot create jobs,
participants, filesystem paths, or role assignments. Statistics acceptance runs
before the ordinary duplicate-outcome gate, allowing an outcome retry to obtain
a statistics result without applying the job outcome twice.

Acceptance checks the input at the authenticated reporting boundary:

1. Check the wrapper and raw byte limit before decoding.
2. Bind the request to the active job and expected, non-disabled sender slot.
3. Decode UTF-8 JSON, rejecting duplicate keys, nonfinite constants, wrong
   kind/version, excess nesting, and unknown fields. Validate identifiers, units,
   statuses, required numeric fields, and finite nonnegative values; require body job/site
   identity to match the bound slot.
4. Normalize object keys and sort/deduplicate issue tags without converting
   numbers to strings. Recheck the canonical byte limit and compare with any accepted record.
5. Under the per-job lock, recheck that acceptance is still open and the slot is
   eligible. Retain immutable canonical bytes and server-owned receipt time.

Hardware descriptions are not collected. Reports must not emit hostnames,
addresses, serials, UUIDs, PCI identifiers, CUDA visibility masks, raw CPU
topology, or raw cgroup paths. The reader accepts only these report fields, but cannot attest to the accuracy
of a site's self-report.

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

The client removes its persisted attempt marker and in-memory launch entry after
an `accepted` or `duplicate` acknowledgement. Unacknowledged attempts retain their
history so a later execution does not claim complete coverage of lost work.

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

After the server job process exits, aborted and failed jobs allow up to 20 seconds
for outstanding client outcomes, capped by a shorter configured
`client_outcome_wait_timeout`. Client cancellation starts before this wait. The
completion loop continues processing other jobs and closes the report as soon as
all expected outcomes arrive. At expiry, absent reports remain gaps; they do not
change the abort or failure outcome. Final status, archive availability, and release
of the scheduler's job slot follow closure, so a queued job can wait for this grace
when the concurrency limit is reached. Successful jobs retain the normal outcome
wait policy.

At closure, each roster entry contains either an accepted value or a gap. An
accepted entry contains `participant_name`, server-owned `role` and `received_at`,
optional `duration_seconds`, and copies of `resource_capacity`, `resource_time`,
and `resource_utilization`, plus its scope-level `issues` list. It omits a success
status. The `reported_at` field remains only in participant detail. A gap contains the name, role, and
`status: "missing"|"invalid"|"disabled"`, with no issue list; an invalid gap also
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

Publication requires descriptor-relative, no-follow directory operations and
rejects symlinked or replaced reserved directories rather than following a
job-provided path. If these operations are unavailable, reporting is unavailable;
job execution continues.

Publication writes only accepted participant bytes through temporary files and
atomic rename, then writes `resource_summary.json` last as the bundle commit
marker before the existing
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

The Session API exposes `Session.get_job_resources(job_id, site=None)` and
`Session.get_study_resources()`. A job response contains `resource_summary`, a
query-only `job_name`, and, when a site is selected, `participant_summary`. The
study method returns the `study_summary` record directly for the session's
active study. These methods extend the existing
[Session](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/fuel/flare_api/flare_api.py)
and [admin job
commands](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/private/fed/server/job_cmds.py)
rather than introducing an HTTP service. The server authenticates the caller and
applies existing job-ID, study, and submitter authorization before fetching a
workspace. A study query also checks the resources-read right for each job and
excludes denied jobs without exposing their IDs or counts.

Each server admits one archive-backed resource query at a time, shared by job and
study queries. A concurrent request receives `resource query busy; retry later`
immediately instead of entering a queue. The admission slot is released when the
server-side query finishes, including failure paths.

Only finalized jobs have archive-backed data. The reader stages the normal
`WORKSPACE` through `JobDefManagerSpec.get_storage_for_download`; the design adds
neither a parallel storage database nor a sidecar-storage API. It reads the summary
member and reads a participant detail only when requested and accepted in the
roster. For each requested member, it requires exactly one matching ZIP member,
rejects encrypted, directory, or oversized members, and bounds actual decoded
reads. It checks detail identity against the summary slot
without extracting or scanning unrelated workspace contents.

Study queries process archives sequentially and remove each archive's temporary
staging directory before proceeding to the next job. The ZIP index has its own
limit because a small JSON member does not imply small archive metadata. Remote
job stores may still download the entire workspace before the reader opens it.

Admission bounds concurrent queries; it does not bound the transfer size, disk
space, or execution time of one query. Transfer-size, disk-space, and hard
execution-time limits remain deferred until statistics can be fetched independently
of the workspace archive. Existing download cleanup and timeout policy still
apply. A client timeout does not cancel server storage I/O or release the admission
slot while that query continues on the server.

Job totals are derived from accepted entries independently for each measurement.
They sum stored `resource_time` and utilization without multiplying startup
capacity by participant duration again. All resource quantities are scalar totals.
Startup capacity and participant durations
are not summed. Any expected gap makes a usable job subtotal partial; if there are
no usable contributors, the result is unavailable rather than zero. Average CPU
use is a participant's CPU seconds divided by its positive containing duration,
not a percentage of host capacity.

A study row contains `job_id`, `job_name`, `job_status`, and
`resource_data: "included"|"unavailable"|"nonterminal"`. Only included rows have
derived `totals`. Coverage records `selected_jobs`, `included_jobs`,
`unavailable_jobs`, and `nonterminal_jobs` as integers. The study
`selection` contains `study_name`, and `generated_at` is the query time. Study
totals sum only included-job totals and preserve their per-field quality.
Unavailable and nonterminal jobs remain explicit coverage gaps, so the
**included-jobs subtotal** is not a claim about all selected jobs.

Nonfinite arithmetic results are rejected rather than clamped or represented as
infinity. Collection preserves valid independent subtotals and marks gaps.
Job and study queries mark an overflowing utilization total unavailable. An
overflowing capacity-time dimension is omitted, leaving the other dimensions
partial, or the measurement unavailable if none remain. The aggregate issue list
identifies the affected metric with `observation_incomplete`. Source reports and
independent measurements remain available.

The CLI uses the existing success/error envelope. Job JSON data contains
`selection: {job_id, site}`, `summary`, and an optional `participant`; study JSON
data contains `selection: {study}` and `summary`. The CLI rejects `--site` without
`--job`; a job query may also specify its study. The `--schema` option describes
the command interface, not the report-format schema. Authorization,
nonterminal-job, and unavailable-data errors remain distinguishable. Text output
rounds only for display; JSON retains native numbers, per-metric status, and
scope-level issues described above.

## Compatibility and failure boundary

Public records use a strict schema: readers reject unknown fields and
unsupported versions. Adding a field, including an optional field, or changing
permitted enum values, units, or field meanings requires a new schema version.
Populating an optional field already defined by the schema does not change the
version.

Mixed-version peers exchange statistics only when the reader supports the
producer's version. There is no implicit conversion or version negotiation. An
unsupported participant report remains a statistics gap without changing the job
outcome; an unsupported archived report is unavailable to the query.

The report does not provide a fallback that reinterprets startup capacity as a
task rollup, and unsupported formats are not silently upgraded. Unsupported or
absent archives are reported as unavailable; deployment may still complete
normally when a peer does not send statistics.

The server trusts authenticated identity and validates report structure, but it
cannot verify the physical truth of reported numbers. Atomic files prevent partial
publication; they do not provide hardware attestation or recovery after loss of
the in-memory ledger. Published reports describe completed work.
Scheduling, live monitoring, billing, restart adoption, and result or checkpoint
acceptance remain outside this protocol.
