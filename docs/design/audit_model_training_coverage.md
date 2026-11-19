# Audit log improvements, part 4: model initialization and saving

A **persistor** is a component that supplies initial model state and stores
updated state. A **checkpoint** is saved model state intended for later reuse.
This increment adds model observations after the
[record format](https://github.com/NVIDIA/NVFlare/pull/5806),
[administration coverage](audit_security_admin_coverage.md), and
[task and workflow coverage](audit_task_workflow_coverage.md) increments.

## Record contract

Timestamps remain in Coordinated Universal Time. JSONL stores one JSON object,
a collection of named values, per line. The only fields are the existing
`event_id` (record identifier), `reference` (approved earlier record identifier),
`timestamp` (writing time), `user` (actor), `action` (observed fact), `job`
(established job), `scope` (actual job scope), and `message` (fixed details).
Values are strings. Unavailable optional values are omitted. The actor for
these operations is `unknown`.

The model helper copies the local job and scope before invoking model code. It
keeps a parent reference only from the existing approved task or workflow
context, when job and scope match. All facts in the operation use that snapshot,
including facts written later. An omitted reference can reflect an unavailable
parent record. Model records do not create identifiers for linking start and end
observations, so concurrent operations need not have uniquely paired records.

Model values, accompanying information (metadata), filenames, file paths,
component names, credentials, and exception text never enter the new records.
The existing logs, storage formats, and transport messages keep their behavior.

## Event catalog

Action names have three parts separated by periods: category `model`, operation
`load`, `save`, or `best_save`, and an outcome from this table.

| Outcome | Observed fact |
| --- | --- |
| `started` | The selected operation began. |
| `stage_returned` | The step named by `stage` returned. |
| `stage_error` | That specific step raised an ordinary exception. Existing code may recover and continue. |
| `returned` | The selected operation returned. A load has status `value_returned` or `no_value`. |
| `error` | An ordinary exception propagated from the selected operation. The stage is included when established. |
| `skipped` | A save was explicitly skipped because it had no configured destination. |

In `message`, an equals sign separates each fixed detail name from its value,
and spaces separate details. For example, `stage=replace` on a
`model.save.stage_returned` record means the destination replacement call
returned. The replacement remains an observed fact if directory synchronization
later raises. These records do not establish durability, model validity,
training success, or best-model quality.

## Capture points and limits

| Entry point | Captured steps |
| --- | --- |
| Shared model persistor load/save | The delegated implementation and the optional pre-save, post-load and post-save transformations. A delegate may initialize in memory, return no model, or return from a save without writing. |
| Built-in NumPy persistence | Explicit source and previous-file reads, fallback value construction, and array writes. Includes the client-controlled best-model save when triggered by its existing event. |
| Built-in PyTorch checkpoint persistence | The actual source or preloaded-checkpoint read, in-memory initialization, serialization to a temporary file, destination replacement and directory synchronization. A lazy initial load reached by an observed save is a separate load. The existing best-model save has its own observed entry point. |
| Built-in averaging controller direct-file training | Preparation, initialization when the model is configured in memory or a file is unavailable, serialization calls, and skipping a save when no destination is configured. The PyTorch version observes parameters and metadata separately. |
| Base model controller without a persistor | Empty initialization, publication into the local job context, and the explicit save skip. |

The stronger storage evidence comes from individual calls by the component
performing the observed load or save. Unrelated reads that application code
initiates during a save do not become stages of that save.
Preparation and logging errors cannot be mistaken for read errors. A stage
error can be followed by recovery, such as a previous-file read error followed
by NumPy fallback initialization. An observed write also remains a fact if a
later transformation, separate metadata write, or synchronization fails.

Named-model retrieval for evaluation, learner-local files, arbitrary
application-specific storage, and custom implementations that bypass these
entry points are outside this increment.

## Recording behavior and verification

The existing bounded collector temporarily holds facts and diagnostic counts.
The selected client-controlled gathering, submission permission, timeout,
finalization, and task-setting entry points collect outside their shared locks. The outermost collector detaches
and writes after those locks are released. These guarantees cover the selected
framework entry points and do not extend to application-owned locks. Timestamps
are writing times, and records from different processes can interleave.

Audit preparation and ordinary failures in writing do not alter model return
values, exceptions, or side effects. A missing writer, closed writer, or enabled
writer returning no identifier counts as a failed recording attempt. Diagnostics
use the shared one-warning-per-minute limit and protected standard-error
fallback. The simulator's explicit no-operation writer remains silent.

Component tests cover storage and transformation failures, partial application,
privacy, and lock release. The existing automated direct and relayed training
suite verifies a packaged NumPy source at both clients, the saved server model
in the archived workspace, and the new records in the server job audit file.
