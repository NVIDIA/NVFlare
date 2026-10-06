# Task assignment identity and result admission

The server assigns each `ClientTask` an assignment ID and a separate physical
attempt ID. Resending an assignment preserves both IDs. This strengthens ordinary
job-lifetime execution; it does not change executor lifetime or configuration.

## Wire compatibility

Task data carries `TASK_ID`, `TASK_ATTEMPT_ID`, and `TASK_ATTEMPT_REQUIRED` in
reserved headers and cookies. The workflow cookie identifies the issuing workflow.
Existing clients that echo the received cookie jar remain compatible with the new
server. Updated clients also echo the attempt header and include the attempt,
task name, and workflow in readiness checks. Legacy readiness checks without an
attempt retain their active-assignment behavior. Unfenced custom communicators
and genuinely unknown legacy results retain their established handling.

An attempt may arrive in either the header or cookie. If both are present they
must agree. A required attempt cannot be missing; malformed values are rejected.
An issued assignment cannot be downgraded to an unfenced result.
Retained assignment IDs stay fenced even if the sender omits the attempt and
changes the site or task name; only genuinely unknown IDs use legacy handling.
The server checks the authenticated peer's site and job, workflow, task name,
assignment, and attempt
before result filters, callbacks, aggregation, or fatal return-code handling.

Server task-data filters may replace a Shareable. The command layer restores the
issuing identity and workflow from private context. Client filters may also replace
task data or results; the client preserves the original assignment cookies and
binds the final outgoing reply. Forwarding a received Shareable as new task data
rebinds reserved cookies on a protected per-client copy and preserves application
cookies. Incoming conflicts remain invalid.

## Receipt versus admission

Transport success reports delivery of the reply, independently of application
admission. The result ACK includes the assignment and attempt IDs and a boolean
admission decision. Clients use this decision only when the ACK identity matches
their submitted result. Older servers without this ACK leave admission unknown.
The client context separately records send success, whether submission transport
was ever attempted, and acknowledged admission.

A successful standing result is admitted by default, retaining compatibility with
callbacks that return nothing. Literal `False` vetoes admission, including when a
callback consumes the result. Failure return codes, callback errors, and task
errors cannot acknowledge successful admission. Built-in training callbacks return
their real admission decision. Received failures still reach their ordinary error
handling callbacks.

A recognized first late result from an authenticated assignment reaches the
existing unknown-task hook, before or after task sweep. This preserves
ScatterAndGather aggregation, FedAvg's existing training-result publication, and
CrossSiteModelEval storage. A void late hook does not imply an accepted ACK; a
custom hook can explicitly set `TASK_RESULT_ACCEPTED` to `True` for successful
admission. A first authenticated late fatal result retains job-abort behavior.
Forged or already-decided attempts cannot cause fatal effects.

## Replay and retention

Once processing claims a result, its admission decision is recorded. An exact
retry replays that decision without repeating filters, callbacks, late hooks, or
aggregation, even when its payload differs. A hook or manager failure records a
rejection so a retry cannot repeat partially applied effects. Job-local receipts
also survive workflow transitions and teardown. Receipt replay is authenticated
and scoped to the original site, job, workflow, task, assignment, and attempt.

Retention is bounded and in memory. Each workflow retains up to 10,000 retired
assignments (including assignments awaiting their first late result); the server
runner retains up to 10,000 result receipts. Access refreshes retention order.
Eviction or server-job exit ends replay availability. An evicted result that still
carries an attempt is rejected rather than passed to the legacy unknown-task hook.
After eviction, an unknown ID without fencing fields is indistinguishable from an
unfenced custom protocol; rejection of stripped attempts requires retained authority.
Pending late assignments are only recognized by their issuing workflow; this does not route a
new late result into a subsequent workflow. These caches do not provide durable
recovery or authorize another execution attempt.

Task-lifetime workers, supervision, launchers, declared state, placement,
capabilities, client-script adapters, and recipes are separate changes. Existing
executors and their configuration paths remain supported.
