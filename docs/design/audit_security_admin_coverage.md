# Audit log improvements, part 2: security and administration coverage

**Status: planning draft; implementation has not started.**

**Dependency:** [Part 1, pull request #5806](https://github.com/NVIDIA/NVFlare/pull/5806),
introduces UTC timestamps and JSONL records. This follow-up depends on that change.
Begin implementation only after #5806 merges, then update this branch from
NVIDIA/NVFlare's current `main`. This document describes planned behavior, not
capabilities already available.

## Recording contract

UTC means Coordinated Universal Time. JSONL stores one JSON object, a collection
of named values, per line. Retain exactly the eight part-1 fields:
`event_id`, `reference`, `timestamp`, `user`, `action`, `job`, `scope`, and `message`.
Their meanings follow the [auditing documentation](../user_guide/admin_guide/security/auditing.rst).

- Keep one request record for each command currently audited. Use its registered
  command name instead of raw arguments. Add dispatch and handling-result records
  only for sensitive operations; successful routine polling gains no records.
- New action names use periods between category, operation, and observed result.
  Put fixed reason codes and approved resource identifiers in `message`.
- Use verified administrator identity; before verification use `unknown`.
  Client records distinguish identity attributed by a trusted server from locally
  verified identity. Rejected origins cannot supply trusted actors or references.
- Link server outcomes to their request through `reference`; link client receipts
  to the server request, and client outcomes to their own receipt. Preserve the
  original request identifier.
- Include only established job identifiers. Use `scope` for actual job scope,
  and put study identifiers in `message`. Omit unavailable context.
- Exclude credentials, session secrets, raw arguments, request bodies, file
  contents, private paths, and exception text from new records.

Add optional keyword-only `job_id` and `scope_name` arguments to the existing
event-writing methods. These populate existing fields, allowing a record to
contain both its actor and its job or scope; existing callers retain their behavior.

Request-local context means values isolated to one request. Use it for attribution
through synchronous calls and reset it on every exit. Functions invoked later
receive an immutable copy of approved values, without retaining connections or
sessions.

## Events and capture boundaries

A capture boundary is the point where the responsible code knows a fact occurred.
A command-handling function returning does not establish business completion.

| Area | Planned records and required meaning |
| --- | --- |
| Login and sessions | Login success, rejection, and processing error; session creation, restoration, rejection, and actual termination. Distinguish logout, inactivity, and certificate expiry. |
| Permission checks | Policy denial, policy-evaluation error, explicit role denial, and preprocessing rejection. Record failures before the ordinary command-audit filter without changing filter order. Emit one rejection per decision. |
| Command handling | Sensitive-command dispatch, handler return, and processing error. Distinguish failures before, during, and after handler execution. |
| Client administration | Rejected origins, client-side permission denials/errors, sensitive-request receipt, and immediate handling results. Legitimate child-status notifications gain no additional administrative records. |
| Job deletion and abort | Record completed manager deletion even if later bookkeeping fails. Distinguish abort before execution, already finished, starting/retry required, signal sent, and error. |
| Study and client access | Study creation/removal and user/site membership changes; client enable/disable/remove. Record per-target results and unchanged state; preserve an applied change separately from later cleanup failure. |
| Logging configuration | Local application or error and remote reported results for site/job logging changes, excluding configuration contents and paths. |
| Shutdown and restart | Rejection, scheduling, and remote acknowledgement where observed; never report that a program stopped or restarted without that evidence. |
| Protected downloads | Job, component, and job-log downloads: setup rejection and the transfer service's terminal status and reason, without claiming confirmed receiver receipt or successful local extraction. |

For study changes, distinguish successful file persistence followed by failed live
configuration reload. For deletion and client policy changes, capture the applied
fact before later cleanup, then emit outside shared locks and cleanup-sensitive
error paths.

For protected downloads, use the existing transfer-outcome callback, a function
invoked with the service's aggregate result. A cleanup notification alone is not
a successful transfer. The reported outcome does not establish receiver-confirmed
receipt when the existing receiver-confirmation setting is disabled.

Preserve permission decisions, replies, retries, and cleanup. Do not infer success
from response text or a general "OK" value. Task execution, workflow results, and
model-artifact coverage are later increments.

## Audit-write failures

Add the shared administrative helper `nvflare/fuel/sec/admin_audit.py`; leave
failure behavior unchanged for unrelated audit callers.

- Protect context preparation, record construction, and writing against ordinary
  exceptions. Return an empty identifier on failure; do not replay operations or
  invent references.
- Treat a missing writer, closed writer, or unexpected empty return from an enabled
  writer as a recording failure.
- Give the simulator an explicit internal no-operation marker. Intentional
  simulator silence and excluded polling records produce no warning.
- Continue the operation and use independent diagnostics, limited to one warning
  per minute per running program. Count failed recording attempts without claiming
  exact lost-record counts.
- Protect diagnostics themselves, with a fixed, protected standard-error fallback.
  Never recursively audit an audit failure.

## Validation and delivery

Use an isolated development checkout and environment, preserving existing
workloads. Verify the exact source revision and follow the
[test placement and authoring rules](../../tests/README.md).

Component tests must verify security decisions, actor attribution, rejection
counts, references, the eight permitted fields, string values, optional omissions,
and absence of secrets. Exercise missing/closed writers, unexpected empty returns,
construction/write/flush/diagnostic failures, and intentional simulator silence.

Cover mixed client results, unchanged operations, changes applied before cleanup
errors, persistence before reload failure, abort/shutdown scheduling, and download
completion/failure/cancellation/timeout without overstating receipt evidence.

Extend the existing live administrator-authorization test to inspect actual
server/client records directly and through a relay, an intermediate participant
that forwards communication. Add bounded real-transfer coverage for download
records and explicitly select it in the existing automated test runner.

After #5806 merges, implement on `improving_audit_coverage` from NVIDIA's refreshed
`main`. Keep this pull request in draft until implementation, affected tests,
style/license checks, event-catalog documentation, and independent review are
complete. Publish signed commits and sign-offs.
