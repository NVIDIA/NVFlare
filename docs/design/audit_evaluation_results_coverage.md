# Audit log improvements, part 6 — evaluation execution and result handling

This increment follows the
[named-model lookup coverage](audit_model_evaluation_coverage.md).
It records selected client evaluation calls and server handling of the
resulting replies. The record describes the step observed by the component
that performed it.

## Record format and attribution

The existing eight string-valued fields and Coordinated Universal Time
timestamps apply. The actor is `unknown`. Each operation copies the locally
established job and scope before application code runs. It carries an approved
local parent record identifier when the active task or workflow context has
the same job and scope. Unavailable optional fields are omitted. There is no
new identifier linking operations across sites.

The records exclude model names, model values, evaluation measurements and
their names, site names, payloads, paths, application-defined return codes,
and exception text. An operation can emit several records that refer to the
same earlier task record. They describe writing time and may interleave with
other operations.

## Event catalog

An action has three parts separated by periods. The category is `evaluation`,
the operation is `execution`, `result`, or `output`, and the outcome is
one of the following. The optional `message` contains fixed `stage`,
`reason`, or `status` details, with an equals sign between a detail
name and its value.

| Outcome | Observation |
| --- | --- |
| `started` | A covered client evaluation handler, server result handler, or summary writer began. |
| `stage_returned` | The named step returned. |
| `stage_error` | The named step raised or detected invalid entries, as described below. |
| `returned` | The selected handler or writer returned at its recorded boundary. |
| `observations_limited` | A result contained more entries than the per-entry storage observation limit. |
| `rejected` | The existing handler discarded an input or reply, or prepared an error reply for it. |
| `aborted` | A covered NumPy validation branch observed its existing abort signal. |
| `error` | An exception propagated from the operation, or a covered client handler caught one and followed its existing error path. |

| Owner | Captured facts |
| --- | --- |
| Model learner executor | The learner's explicit validation call (`learner`), preparation of its reply (`encode_reply`), and rejection or error paths. |
| Legacy learner executor | The explicit validation call (`learner`) and whether a reply was prepared from its return value. |
| NumPy validator | The selected validation branch, its input rejection, abort checks, error path, or reply preparation. Its shared delay and abort before task selection are outside this observation. |
| Legacy cross-site evaluation controller | Handling of validation replies, including the existing route for replies with no matching outstanding task. A non-success reply or invalid result can be rejected. Invalid nested entries can be reported separately from valid entries. Each `store_result` stage describes the existing per-entry storage method returning or raising. |
| Modern cross-site evaluation controller | The result callback's tracking call (`track_result`) and its individual result-file call (`file_write`). Earlier reply checks and conversion are covered by the existing task audit where applicable. |
| Modern summary writer | Its serialization step (`serialize_summary`), write call (`file_write`), and return after file closure. Serialization still follows opening the destination. A write return remains visible if closing the file later raises. |

Client rejection reasons distinguish an unusable input (`invalid_input`), an
unusable return value (`invalid_reply`), and a learner returning an application
error code (`learner_code`). The legacy server distinguishes a reply with a
non-success code (`non_ok_reply`) from a reply it could not decode
(`invalid_result`). An invalid nested entry is recorded as `invalid_entries`
at the `decode_result` stage. NumPy records whether it observed the abort
before or after its calculation with `before_evaluation` and `after_evaluation`.

The `reply_prepared` status describes the reply kept by the client handler.
It can include an existing error reply and does not establish receipt.
The `processed` status describes a server result handler reaching its end.
A nested result can be partially processed when individual entries are discarded.
The legacy server records successful storage calls for the first 64 entries of
one result. Additional entries still follow the original storage path. One
`observations_limited` record names the `store_result` stage and the
`record_limit` reason. A later storage exception still produces its stage error.
This limit leaves room for the result and task handling outcomes in the shared collector.
A returned file call establishes that the call returned. Durability, evaluation
quality, and the meaning of measurements are outside these observations.

## Boundaries and validation

The existing bounded collector defers records and recording diagnostics until
the covered framework locks are released. The normal server callback route
uses the communicator's collector. The model learner executor collects outside
its own lock. Ordinary recording failures continue to use the shared diagnostic
limit and preserve the original operation. The simulator remains intentionally
silent.

Pre-training validation, calls that bypass the selected executors and
controllers, model lookup and task transport have separate boundaries.
This increment introduces no additional records for those calls.

Component tests cover reply preparation, explicit rejection, abort and
error handling, partial results, file-write and close failures, capture after
lock release, and recording failures. The direct and relayed live cases
verify that both clients evaluate the expected saved model. The downloaded
server audit and both client workspace audit files describe the observed
handling.
