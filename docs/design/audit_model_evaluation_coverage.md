# Audit log improvements, part 5 — named model retrieval for evaluation

A named model lookup asks a configured component or workflow for a saved
model using a name or location. This increment follows the
[training-model coverage](audit_model_training_coverage.md).

The same eight string-valued audit fields and Coordinated Universal Time
timestamps apply. The actor is `unknown`. Each lookup snapshots the locally
established job and scope before calling model code. It retains an approved
parent record identifier only when both match the active task or workflow
context. Unavailable optional values are omitted.

## Records and evidence

An action consists of the category `model`, operation `retrieve`, and an
outcome, separated by periods. The `message` field can identify a fixed
step in `stage` or a fixed result in `status`. An equals sign separates each
detail name from its value.

| Outcome | Observed fact |
| --- | --- |
| `started` | The selected lookup began. |
| `stage_returned` | The named step returned. |
| `stage_error` | The named step raised. Existing code can catch the error and continue. |
| `returned` | The lookup returned. The status is `no_value` for the Python value `None` (no value) or `value_returned` for any other value. |
| `error` | An exception propagated from the selected lookup. The record includes the step when established. |

The `implementation` step is the delegated lookup call. The `post_get_filter`
step is the shared model persistor's optional transformation after lookup.
A persistor is a component that reads or stores model state. The `file_read`
step records a file-reading call at the component or workflow that owns it.
A returned read remains observable when subsequent model conversion or
filtering fails. A miss in a built-in inventory has no file-read record.
An unreadable saved model can produce a step error followed by a return with
no value when existing code catches the read exception.

| Entry point | Captured work |
| --- | --- |
| Shared model persistor | Delegated named lookup and optional post-lookup filter. |
| Built-in cross-site evaluation controller | Its direct call to the persistor's raw lookup or its own saved-file read. The raw path keeps its existing behavior and does not invoke the shared filter. Conversion for dispatch and sending evaluation tasks are outside this observation. |
| Built-in NumPy model locator | Alias lookup and the individual array read when attempted. |
| Built-in NumPy, PyTorch and TensorFlow named persistors | The individual array, saved model state, or model-weights read when reached from a covered lookup. |

These records establish lookup and storage-call observations. Evaluation
dispatch and results, model validity, and model quality are separate facts.
Standalone inventory queries, raw lookup calls outside the covered workflow,
learner-local work, and custom storage that bypasses these entry points are
outside this increment. Names, aliases, inventory keys, component names,
paths, model contents, accompanying metadata, credentials, and exception
text are excluded from the new records.

The existing bounded collector and recording-failure diagnostic protect
application behavior. An audit failure does not cause a retry or change the
lookup return value or exception. Intentional simulator silence remains silent.

Component tests exercise matching and missing names, read and transformation
failures, and audit-record attribution and privacy. The automated direct and
relayed federation tests use the trained server output in a subsequent
evaluation and inspect the server's archived retrieval records.
