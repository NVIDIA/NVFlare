# Audit log improvements, part 3: task execution and workflow results

This increment builds on the merged changes from
[part 1, pull request #5806](https://github.com/NVIDIA/NVFlare/pull/5806) and
[part 2, pull request #5823](https://github.com/NVIDIA/NVFlare/pull/5823).
It covers task execution and workflow results. Model saving/loading,
algorithm-specific aggregation decisions and external application protocols are
later increments.

## Record contract

UTC means Coordinated Universal Time. JSON is a collection of named values;
JSONL stores one such object per line. Retain exactly the existing eight possible
top-level fields: `event_id`, `reference`, `timestamp`, `user`, `action`, `job`,
`scope`, and `message`. Every value is a string; unavailable optional context is
omitted. Existing combined task-name/task-identifier actions and the existing
audit transport header remain unchanged.

New task/workflow actions use periods between category, operation and observed
result. Reasons, stages and status values come from fixed code vocabularies.
Only built-in return codes are retained; an application-defined code becomes
`custom_return_code`, and an absent code becomes `unspecified`. Task data, result
contents, model values, credentials, session secrets, private paths, exception
text, callback representations and arbitrary application-defined messages are
excluded from new records.

The existing `message` string contains fixed named details. An equals sign
separates a name from its value; spaces separate details. Task names, task
identifiers, workflow identifiers, attempt identifiers and site names are quoted
strings: quotes surround the value and special characters are escaped. These
details are not new top-level JSON fields. Accepted task/scope names can contain
spaces or non-English characters; validation must preserve their existing
contracts rather than apply command-name restrictions to every field.

Task processing normally has no verified administrator actor. New records use
`user=unknown`; they do not infer an administrator from job submitter metadata.
Use the local established job and its actual effective scope. Study names are
not scopes. Before peer/job/assignment validation, a rejection includes only
local context and a fixed reason. It cannot acquire task identity, site or
reference from the rejected request.

## Correlation and evidence

The existing audit identifier header and combined legacy records retain their
behavior. New observations only reference identifiers issued by the local
writer. The client links validated ordinary assignments and their execution to
its locally written legacy receipt. The server links matched results and their
handling to its locally written legacy receipt. These receipts continue to
carry the existing legacy links between processes. Direct auxiliary execution
and unmatched result handling omit any unchecked reference. Buffered events
never invent an identifier or replace a transport header.

Task identity used after filters comes from the original approved assignment.
Result preparation and publication can reference the locally written legacy
result record using the identifier captured before before-send callbacks.
Workflow lifecycle records can reference the locally written initialization
record. Later functions receive immutable approved strings. Unavailable
identifiers are omitted. No transport fields are added.

The following are separate observations:

- An assignment was prepared for transport.
- The client validated the assignment and invoked its executor.
- The executor returned, raised, observed abort, or was interrupted.
- A result was prepared for publication.
- Transport acknowledged a legacy publication, or an explicit result receipt
  confirmed complete receipt. Lost-acknowledgement recovery can obtain a receipt
  through task checking without another upload.
- The owner claimed a complete result, replayed a duplicate receipt, or allowed
  a first late result.
- Result filtering, manager handling, a result callback or an unknown/late-result
  hook returned or failed.
- A task left the standing-task collection with a particular status. A later
  cleanup/done callback can fail independently.
- Workflow initialization, control handling and finalization returned or failed.

Receipt does not establish successful callback processing, aggregation, model
quality or training completion. A callback returning does not establish those
application outcomes either. An observed abort signal does not prove every
worker stopped. Finalization remains a separate observation after control-flow
errors, and the end-workflow notification is not evidence of successful cleanup.

## Capture owners

All paths below are relative to `nvflare/`. Result words in braces are
alternatives in this table, not literal characters in an action.

| Owner | Captured facts |
| --- | --- |
| `private/fed/client/client_runner.py` | Validated task receipt/rejection; input/output filter outcomes; executor invocation, return, error, abort and interruption; result preparation; one terminal publication observation across retries, distinguishing explicit receipt from a legacy transport acknowledgement. |
| `private/fed/server/server_runner.py` | Assignment preparation; result rejection/retry and filter/handling boundaries; workflow initialization, control return/error/abort observation, and finalization. |
| `apis/impl/wf_comm_server.py` | Matched assignment receipt, duplicate/rejected/late decisions, actual result-manager/callback/hook outcomes, task retirement and later done-callback errors. Finalization's standing-task clearing is covered as well as monitor retirement. |
| `apis/impl/task_controller.py` | Direct auxiliary-task preparation/dispatch, validated target reply/no-reply, filters and callback outcomes, rejection and handling errors. |

The implemented action catalog is:

| Action family | Observation |
| --- | --- |
| `task.assignment.{prepared,received,rejected,error}` | Locally prepared outgoing assignment, validated client receipt, fixed rejection or preparation error. Preparation is not confirmed delivery. |
| `task.execution.{started,returned,error,aborted,interrupted}` | The actual executor boundary; interruption retains the existing worker-recovery exception. A returned executor need not report a successful application code. |
| `task.filter.{error,aborted}` | Client data resolution/input/output filtering failed or observed abort. The stage identifies which boundary. |
| `task.processing.error` | An existing outer client-processing error path. |
| `task.result.prepared` | The outgoing result, including an error result, is ready for publication. |
| `task.publication.{received,acknowledged,closed,aborted,error}` | One final publication observation across retries. `received` requires explicit receipt; `acknowledged` is the weaker legacy transport observation. The send decision distinguishes task closure from stopping locally. Abort/error do not establish that the parent lacks a copy. |
| `task.result.{rejected,retry_required,duplicate,received,abort_requested}` | Actual result-admission decision, complete receipt, or fatal-code panic request. Duplicate receipt does not repeat application work; abort requested does not prove the job stopped. |
| `task.result.{filter_error,handling_returned,handling_error}` | Server filter or immediate communicator-handling boundary. The communicator returning does not prove a swallowed callback succeeded. |
| `task.result_manager.{returned,error}` | The matched assignment's result manager returned or raised. |
| `task.result_callback.{returned,error}` | Actual result callback returned or raised, independently of receipt. |
| `task.{late_result,unknown_result}.{returned,error}` | The existing late/unknown-result hook and its surrounding notifications. Unmatched legacy hooks omit unavailable assignment identity/reference. |
| `task.retired` | The task left standing-task collections, with its observed fixed completion status and monitor/finalization reason. |
| `task.done_callback.{returned,error}` | Later task-done callback outcome, separate from retirement. |
| `task.cleanup.error` | Finalization resource release failed after removal; earlier retirement facts remain. |
| `workflow.execution.{initialized,error}` | Workflow became active after initialization, or initialization/control handling raised at the recorded stage. |
| `workflow.control.{returned,abort_observed}` | Control function returned and, independently, the abort signal was observed. Neither is a training-success claim. |
| `workflow.finalization.{returned,error}` | Stop-controller/finalize handling returned or failed, separate from control handling. |
| `task.auxiliary.{prepared,dispatched,outcome_reported,no_reply,rejected,filter_error,callback_returned,callback_error,handling_error}` | Direct auxiliary send invocation, reported per-target outcome or absent result entry, and filter and callback handling. Callback stages distinguish result and task-done callbacks. |

Direct auxiliary requests have a local task name and known targets but do not
issue the ordinary per-recipient task/attempt identifiers. Their records omit
those identifiers and unavailable references. A returned entry can represent a
response or a status generated by the local transport, such as a timeout.
The reported outcome makes no claim that the target sent a reply. A no-reply
record means the sender returned no entry for that target.

The client-to-client receipt-owner implementation in
`private/fed/client/communicator.py` is not expanded in this increment. Client
execution shared with `ClientRunner` can still produce execution records; that
does not establish complete coverage of peer-workflow admission or external
protocols. Model persistence and algorithm-specific result/aggregation decisions
are explicitly excluded.

Successful heartbeat, readiness, no-task and try-again polling gains no new
records. Publication retries produce one terminal observation rather than a
record per retry. Actual duplicate result submissions can still produce distinct
duplicate-decision records while preserving once-only application processing.

## Recording outside shared locks

The shared helper is `fuel/sec/task_audit.py`. `TaskAuditContext` is an immutable
collection of approved strings. Its builder reads only local job/scope by
default; task/peer/reference information must be supplied explicitly after
validation. Scoped context is reset on every exit.

A collector temporarily stores immutable facts while an operation owns shared
locks. Nested collectors join the outer operation. The outermost collector
restores its caller's context and detaches its pending facts before invoking any
writer or diagnostic output. This keeps new audit writes and diagnostics outside
the server workflow lock, communicator controller/task locks and task callback
locks. Each collector is bounded to 1,024 facts; overflow counts as failed
recording attempts without interrupting the operation. A collector must never
span an entire workflow lifetime. Preparation failures are deferred too.

Existing audit timestamps are generated by the writer. Buffered records therefore
carry writing time, which can be later than observation time. Records from
different operations can interleave. Neither file order nor timestamps establish
a total execution order across workers/sites; use the documented identifiers and
observed meanings instead.

## Recording failures and compatibility

New task/workflow recording catches ordinary preparation, construction and
write/flush exceptions, detects missing/closed writers and unexpected empty
returns, and returns an empty identifier on failure or deferral. It never
replays an executor, callback, transport operation or workflow operation.
Reentrant writer hooks cannot recursively create new task audit writes.
The simulator's explicit no-operation marker remains intentionally silent.

Task/workflow failures share part 2's protected process-wide diagnostics and
one-warning-per-minute limit. Counts describe failed recording attempts, not
exact numbers of lost records: a flush failure may follow a completed write.
The fixed message and failure count go directly to standard error, independently
of the site's logging settings. Errors writing the diagnostic are contained.
No audit failure is recursively audited.

Existing legacy task records, helper calls, audit-header values, return codes,
receipts, retries and operation cleanup keep their behavior. In particular,
pre-existing legacy writer calls can still propagate failures. The new helper's
continuation guarantee applies to the newly added records and does not globally
change `AuditService` or unrelated callers.

## Validation and delivery

Component tests write actual audit files with controlled boundaries. They cover
the eight allowed string fields and optional omissions; task/scope names with
spaces and escaped characters; absent/untrusted context and references; no
payload/secret/custom-code text; executor/filter/abort/interruption paths;
explicit receipt and legacy acknowledgement; duplicate and late results;
callback errors after receipt; task retirement and finalization; workflow
initialization/control/finalization errors; missing/closed/empty/failing writers;
bounded nested collectors, context reset, reentrant writers and lock exclusion.

The live administrator-authorization suite already selected by
`ci/run_process_contracts.py` runs one real training round directly and through a
relay. It additionally checks server-job and client-job audit files at the
workspace-selected locations, validates task/result/workflow reference chains,
and plants private task/result markers to prove the new records exclude them.
Bounded readiness accounts for final audit flushing after job status changes.
Failure/duplicate/interruption permutations are controlled component cases,
not claims made by the successful live round.

Develop and validate in an isolated environment on the approved machine while
preserving other workloads. Run affected tests, project style/license and
documentation-link checks; obtain independent quality and plan-compliance
review. Publish signed commits with sign-offs on
`codex/audit-task-workflow-coverage`. The third pull request targets NVIDIA's
current `main` and stays draft pending review.
