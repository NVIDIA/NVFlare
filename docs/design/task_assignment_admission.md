# Task assignment identity and result receipt

The server gives each client assignment a task ID and a separate attempt ID.
Resending an assignment preserves both. This change supports ordinary job-lifetime
executors and the future task-worker path without changing executor configuration.

## Client contract

The client keeps its completed result until it receives an identity-bound reply:

| Reply | Client action |
| --- | --- |
| `RECEIVED` | Stop resending: the server received the complete payload. |
| `TASK_CLOSED` | Stop resending: this assignment is no longer wanted; get the next task. |
| `RETRY` | Keep the result and retry its submission. |

A lost, malformed, or mismatched reply leaves receipt unresolved. Retry submission
of the same saved result; never rerun the executor to recreate it. A readiness
check can recover a lost `RECEIVED` ACK without uploading the result again.
Closure can race readiness and upload; the submission reply then reports
`TASK_CLOSED`.

`TASK_CLOSED` does not say whether an earlier upload of this attempt was received.
An ACK can be lost before the workflow ends or its history entry is evicted. The
client stops resending in either case, but closure is not evidence of non-receipt.

Receipt says nothing about aggregation acceptance. Failed results, filter errors,
callback rejection, and application exceptions still receive `RECEIVED` once the
server owns the complete result. Aggregators keep their existing local decisions
and callback signatures. A receipt is not a promise to aggregate or persist a
result, nor a durable recovery guarantee across server restart.

The client context exposes `TASK_RESULT_RECEIPT`, whether transport was ever
attempted, and send success. For fenced assignments, send success means confirmed
receipt. Closure stops retries with send success false and a `TASK_CLOSED` receipt.
Cancellation without a terminal server reply leaves receipt unknown.

## Identity and compatibility

Assignment data carries task ID, attempt ID, required-attempt marker, and workflow
in reserved headers and cookies. Clients echo the assignment cookies. If attempt
header and cookie are both present, they must agree. The server validates the
authenticated site and job, workflow, task, assignment, and attempt before filters,
callbacks, aggregation, or fatal handling. ACKs and readiness receipts carry the
task ID, attempt ID, and original workflow; clients verify all three.

Task-data and result filters may replace Shareables. Trusted replacements receive
the original assignment identity in headers and cookies; application cookies stay
intact. Forwarding task data creates a per-client copy with a new assignment.
Incoming conflicts are rejected before trusted replacements can rebind them.

Existing clients that echo task cookies remain compatible and may ignore the new
receipt. Unfenced older servers retain legacy transport behavior with receipt
unknown. Legacy readiness checks retain their active-assignment behavior.
Unfenced custom communicators and genuinely unknown legacy results keep their
established hooks; custom authorities issuing attempt IDs must implement receipt
claims. No boolean aggregation-admission ACK is exposed.

Readiness uses `WFCommSpec.process_task_check`: the communicator validates the
requested task name, attempt, and authenticated peer, then reports the receipt
through `FLContextKey.TASK_RESULT_RECEIPT`. The runner does not inspect the
communicator's task records or private receipt markers.

## Ownership, retries, and lifetime

FOBS resolves streamed data before command dispatch. An unresolved forwarding
reference receives `RETRY`; receiving its envelope alone cannot confirm receipt.
The scheduling authority records receipt before filters or application side
effects. The workflow's processing-finished timestamp is updated separately, so
a receipt claim cannot prematurely complete a broadcast.

A repeated publication for the same assignment returns `RECEIVED` without repeating
filters, callbacks, aggregation, late hooks, or fatal effects, even if its payload
differs. A first recognized late result within its issuing workflow still follows
the established unknown-task hook. Exceptions after receipt do not repeat effects.
Fatal results record receipt before panic handling.

There is one workflow assignment history, with identity and receipt metadata,
never result payloads. Active assignments hold their own receipt marker; retired
assignments enter a bounded LRU history. Configure `task_result_history_size` in
the server application config (positive integer, default 10,000). An evicted
fenced assignment returns `TASK_CLOSED` and cannot enter a legacy unknown-task hook.

Workflow finalization clears this history. After the workflow advances, submissions
and readiness checks for the original workflow return `TASK_CLOSED`. There is no
second job-level receipt cache and no need to replay aggregation decisions across
workflows. After eviction an unknown ID with all fencing fields stripped remains
indistinguishable from an unfenced custom protocol.

In a client hierarchy, the parent client confirms complete receipt from each
assigned child independently of its local aggregation decision. Readiness and
submission replies use the forwarded assignment identity. Each running parent
assignment holds its children's receipt markers; retries do not repeat result
events. When the parent task ends or aborts, its child assignments return
`TASK_CLOSED`. These markers end with the parent assignment and need no cache.

Task-worker artifact cleanup must wait for terminal receipt or closure and for
outstanding transfer readers to release the source. Worker supervision, launchers,
declared-state promotion, adapters, and recipes remain separate work in #5352.
Its integration must consume this receipt contract and decide state promotion
locally; it cannot gate promotion on a server aggregation-admission boolean.
In particular, `TASK_CLOSED` cannot be interpreted as proof that an earlier result
was not received or used; closure alone cannot decide state promotion.
