# Audit log improvements, part 2: security and administration coverage

**Status: implemented; part 1 has merged and this follow-up is based on NVIDIA's refreshed `main`.**

**Dependency:** [Part 1, pull request #5806](https://github.com/NVIDIA/NVFlare/pull/5806)
introduced UTC timestamps and JSONL records and has merged. This follow-up builds
on that change and includes NVIDIA/NVFlare's refreshed `main`. Required review
and validation remain prerequisites for merging this follow-up.

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

| Area | Records and required meaning |
| --- | --- |
| Login and sessions | Login success, rejection, and processing error; session creation, restoration, rejection, and actual termination. Distinguish logout, inactivity, and certificate expiry. |
| Permission checks | Policy denial, policy-evaluation error, explicit role denial, and preprocessing rejection. Record failures before the ordinary command-audit filter without changing filter order. Emit one rejection per decision. |
| Command handling | Sensitive-command dispatch, handler return, and processing error. Distinguish failures before, during, and after handler execution. |
| Client administration | Rejected origins, client-side permission denials/errors, sensitive-request receipt, and immediate handling results. Legitimate child-status notifications gain no additional administrative records. |
| Job deletion and abort | Record completed manager deletion even if later bookkeeping fails. Distinguish abort before execution, already finished, starting/retry required, abort requested, and error. |
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

The implementation is on `improving_audit_coverage`, with signed commits and
sign-offs, and includes NVIDIA's refreshed `main` after #5806 merged. Run affected
tests, style/license checks and independent review before considering it for merge.


## Implemented event catalog

An action is the name in the existing `action` field. In the table, braces list
alternative final words; they are table notation, not literal action characters.
A period separates the category, operation, and observed result. The existing
`message` field contains approved named details: an equals sign separates each
detail's name and value, and spaces separate details. `reason` is a fixed reason
code; `command`, `site`, `study`, `component`, and `target` identify approved
resources; `stage`, `status`, `operation`, and `identity_source` qualify the fact.
These details do not add JSON object fields.

| Action or action family | Observation and source |
| --- | --- |
| Registered command name | One existing request record, without arguments; `fuel/hci/server/audit.py`. |
| `admin.login.{succeeded,rejected,error}` | Identity/study validation and login handling; `fuel/hci/server/login.py`. Before verification the actor is `unknown`. |
| `admin.session.{created,restored,rejected,ended}` | Session storage, verification failure, or actual removal; `fuel/hci/server/sess.py` and `login.py`. End reasons distinguish `logout`, `inactivity`, and `certificate_expired`. No session identifier or token is recorded. |
| `admin.permission.{denied,rejected,error}` | Role denial, failed preprocessing, policy denial or evaluation error; `fuel/hci/server/authz.py` and `private/fed/server/cmd_utils.py`. |
| `authorization.role.denied` | Explicit study-role denial; `private/fed/server/study_cmds.py`. The shared rejection marker prevents an additional preprocessing rejection for the same decision. |
| `admin.command.{rejected,dispatched,handler_returned,error}` | Resolution failure or sensitive-command handling; `fuel/hci/server/reg.py`. Error stages are `before_handler`, `handler`, and `after_handler`. A returned handler is not business completion. |
| `client.origin.rejected` | The origin failed the existing trusted-server/own-child checks; `private/fed/client/admin.py`. Actor is `unknown`; actor, job, and reference headers are ignored. |
| `client.command.{received,rejected,handler_returned,error}` | Sensitive receipt, unknown topic, and immediate handling; `private/fed/client/admin.py`. Error stages distinguish `pre_handler`, `handler`, and `post_handler`. |
| `client.permission.{denied,rejected,error}` | Site policy or authorization-service decision/error; `private/fed/client/admin.py`. A policy event error records the observed error without changing the existing permission decision. |
| `job.delete.{applied,rejected,error,cleanup_error}` | Configured manager deletion completed, precondition rejected, or later bookkeeping failed; `private/fed/server/job_cmds.py`. Applied deletion is recorded before later bookkeeping. |
| `job.abort.{before_execution,unchanged,retry_required,requested,error}` | Status update before execution, already finished, starting/retry, a handled abort request, or error; `private/fed/server/job_cmds.py`. A handled abort request does not establish that a signal was dispatched or execution stopped. |
| `study.{register,remove,site_add,site_remove,user_add,user_remove}.{applied,unchanged,rejected,error,persisted,reload_error}` | Registry mutation and each requested membership target; `private/fed/server/study_cmds.py`. `persisted` followed by `reload_error` distinguishes saved file state from failed live reload. Records are emitted outside the registry lock. |
| `client.{enable,disable,remove}.{applied,unchanged,error,cleanup_error}` | Actual policy persistence/removal and later cleanup, where applicable; `private/fed/server/client_manager.py` and `server_engine.py`. Per-target mixed results remain separate. |
| `client.policy.rejected` | Invalid client-policy request; `private/fed/server/training_cmds.py`. |
| `logging.{site,job}.{applied,error,rejected,remote_reported}` | Local application at its owner, setup rejection, or explicit remote response; server `sys_cmd.py`, `job_cmds.py`, `server_commands.py`, and client `sys_cmd.py`, `admin_commands.py`, `training_cmds.py`. `remote_reported` describes explicit acknowledgement/error/no reply/unknown outcome, not confirmed application. |
| `admin.{shutdown,restart}.rejected` | Command preconditions rejected; `private/fed/server/training_cmds.py`. |
| `server.{shutdown,restart}.{rejected,scheduled,error}` | Existing server preconditions or submission to the executor; `private/fed/server/server_engine.py`. |
| `client.{shutdown,restart}.{scheduled,error,cleanup_error,acknowledged,reported_error,no_reply,reply_received}` | Local scheduling or immediate remote reply; `private/fed/client/client_engine.py` and server `training_cmds.py`. No action claims that the program stopped or restarted. |
| `download.setup.rejected` | Protected download setup could not proceed; `private/fed/server/job_cmds.py` and `fuel/hci/server/binary_transfer.py`. |
| `download.transfer.{completed,failed,aborted,error}` | Transfer service's terminal aggregate outcome, through its outcome callback; `fuel/hci/server/binary_transfer.py`. `error` is reserved for an unknown service outcome. |
| `download.job_log.{prepared,error}` | The direct job-log command prepared its reply or encountered an error; `private/fed/server/job_cmds.py`. |

All source paths in this table are relative to `nvflare/`. Action families show
the available result vocabulary; not every operation has every result.
Reason codes are fixed at the capture sites; arbitrary reply/error text is never
copied into a new administrative record. Transfer reasons are limited to
`all_receivers_succeeded`, `receiver_failed`, `no_objects`, `timeout`, `deleted`,
`unknown_receiver_count`, `unknown_done_status`, `computation_failed`, or
`unknown_outcome`.

## Attribution and capture limits

The server's request record keeps its original identifier. Later server records
reference that request. A trusted client receipt references the server request;
client handling and immediate owner records reference their local receipt.
`identity_source=locally_verified` means this program verified the administrator;
`identity_source=trusted_server` means the trusted server supplied that identity.
Rejected origins cannot supply attribution. Successful internal child-status
notifications produce no administrative record.

Job children receive some logging changes through an existing parent/child
command path that does not carry administrator attribution. Their local logging
application records use actor `unknown` and omit unavailable references; the
parent's immediate remote-result record retains its established actor/reference.
No extra transport fields or session objects are introduced to infer attribution.

Job and component archive downloads have service terminal records. Service
`completed` is not evidence of confirmed receiver receipt: receiver confirmation
can be disabled. It also says nothing about local archive extraction. Logout or
expiry can cancel a bound transfer; its later terminal record still uses the
immutable approved values captured when it was set up. Cancellation and timeout
remain distinct service reasons.

The direct `get_job_log` command returns data in its ordinary command reply and
has no transfer-service terminal callback. Its `prepared` record establishes only
that reply preparation finished. The surrounding handling record does not prove
that the administrator received or saved that data.

Only explicit sensitive command names in `admin_audit.py` gain dispatch/handling
records. Successful routine polling adds none. Existing unrelated job/task audit
callers retain their behavior; task execution, workflow results and model-artifact
coverage remain later increments.

For site logging and remote shutdown/restart, retain an audit-only copy of the
selected client tokens and approved names before sending. Correlate returned
entries by token without recording the token itself. A selected target omitted
from the returned list receives one `no_reply` observation, as does a target with
an explicit absent reply. Unknown site-logging reply codes are `unknown_outcome`;
only an explicit built-in success code establishes acknowledgement.

The existing job-child command helper represents both an ordinary empty child
response and some failed transport responses with the same absent value. Parent
job-logging observations therefore use `unknown_outcome` for falsy results;
truthy errors remain remote errors. The original command responses are unchanged.
An owner-side `logging.job.applied` record is a separate local fact.

For a running job, `job.abort.requested` means the existing stop request returned
without an error. A job can remain tracked while its server process is already
absent, so this does not establish signal dispatch. The existing administrator
reply is preserved; new audit records do not infer evidence from that reply.

## Administrative recording failures

`AdminContext` holds immutable strings isolated to the current request.
`audit_context` resets them on exit; `snapshot` copies only approved values for
deferred work. `prepare` protects extra audit-only preparation; it must never wrap
the administrative operation itself. `record` validates and writes through the
existing auditor. `reject` marks an already recorded decision so preprocessing
does not duplicate an explicit role rejection.

Missing/closed writers, ordinary construction/write/flush errors, and unexpected
empty results count as failed recording attempts and return an empty identifier.
An explicit simulator marker and excluded polling are intentional silence.
Independent diagnostics warn at most once per minute per program. Diagnostic
failure uses protected fixed standard-error text. Counts describe attempts;
after a flush error a line may already have reached the operating system, so they
are not exact counts of lost records. Recording never retries the administrative
operation or invents an identifier/reference.

## Validation coverage

Component suites exercise the helper and actual JSONL files, security decisions,
attribution/reference relationships, optional field omissions, string values,
secret exclusion, error boundaries and partial changes. They include actual
writer failures and assertions that recording happens outside shared locks.

The existing `peer_admin_authorization_test.py` runs a real administrator, server,
and clients directly and through a relay. It inspects actual server/client audit
files and verifies rejected origins and reference chains. The new
`admin_download_audit_test.py` uses real communication endpoints for completed,
failed, cancelled, and timed-out bounded transfers. Both are explicitly selected
by `ci/run_process_contracts.py`, the shared automated process/transport test
runner. Test workers and communication endpoints are cleaned up on failure.
