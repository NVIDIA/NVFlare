# PyTorch CIFAR client state continuity

This slice builds on merged P4 (`94c5b0aebd3de9a6733926d5fc285144bfd335da`).
It is independent of the separate FedOpt recipe metadata patch. It closes the
native training-state integration for the shipped PyTorch CIFAR FedAvg, FedOpt
and FedProx client loops; it does not declare the overall task-execution feature
complete.

Previously, fresh workers reset SGD momentum, cosine-scheduler progress and
random state. The FedAvg and FedOpt scripts also treated evaluation as training.
The three clients now restore and save explicit native state through
`flare.system_info()["task_checkpoint"]`, and all three handle evaluation
without optimizer steps or checkpoint candidates.

Application source changes are explicit: the three shipped clients opt into an
example-local checkpointable DataLoader and call the shared restore/save helper
around each training assignment. This is not automatic capture of arbitrary
unchanged Python programs. Users of the updated examples select lifetime through
their job entry point with `--execution_lifetime task` (omission retains job
lifetime), or through the common recipe configuration:

```python
recipe.add_client_config({"execution_lifetime": "task"})
```

The three job entry points explicitly bundle the shared `src` directory,
including the checkpoint and data-worker modules. Exported jobs do not rely on
the author's `PYTHONPATH` to supply these files at the client site.

The native file contains optimizer/scheduler state, sampler/worker state, and
Python/NumPy/PyTorch CPU RNG state. The helper validates job, site, prior-round and
configuration compatibility before restoring. Missed rounds can reuse the most
recent earlier local state. The server still provides global model weights.
Checkpoint promotion remains owned by P4 after validated completion and process
settlement; it is separate from server receipt and algorithm acceptance.

The selected CIFAR clients consume complete local epochs. At that boundary every
prefetched batch has been consumed. The loader captures each worker's RNG after
collating its last batch, allowing a fresh worker to resume the same augmentation
stream while job lifetime retains its persistent workers. Default PyTorch
sampler/model RNG coupling is retained: the adapter reproduces the normal
iterator seed draw for job execution and avoids an extra main-process seed draw
when a task recreates previously persistent workers. A direct native-DataLoader
comparison checks batches and RNG progress with zero and two workers.

Other callers of `create_data_loaders()` retain its existing default behavior;
only the three clients opt in. The adapter covers these map-style datasets at
fully consumed epoch boundaries, not arbitrary dataset-owned mutable objects or
mid-epoch recovery. There is no generic worker component graph or per-task
guardian.

## Validation

The focused native checkpoint checks cover optimizer momentum, enabled/disabled
schedulers, all three RNG sources, missed rounds, invalid scope/version/rounds,
and unchanged job-mode file behavior. Existing FedProx evaluation/submission
checks remain included.

The paired Process tests run the actual three shipped client loops with a small
offline model/dataset. They exercise stochastic augmentation and dropout,
shuffle, two local epochs, three federated rounds, two sites, zero/two DataLoader
workers, global evaluation, native checkpoint promotion and fresh-worker
identities. The client files deployed in both lifetimes must match source bytes.
Every round's input, output, metrics, steps, optimizer, scheduler and RNG states,
as well as final model/evaluation/checkpoint states, are compared exactly.

Final local qualification on macOS / Python 3.14.4 / PyTorch 2.12.1:

| Check | Result |
| --- | --- |
| Native checkpoint, original PyTorch resident-trajectory comparison, existing FedProx dispatch checks | 15 passed |
| Adjacent recipe, export/task-contract and FedProx regressions, including those 15 checks | 214 passed, 6 skipped (TensorFlow unavailable) |
| Real entry-point CLI/default lifetime and complete shared-source export, all three algorithms | 6 passed |
| Actual client loops, job/task pairs, zero/two persistent data workers | 6 passed in 484.43 seconds; 12 jobs, two sites, three rounds |
| Full project Python style and whitespace | Passed; notebook formatting skipped without Black's Jupyter dependencies |

Every paired model/state/evaluation comparison passed exactly, including the
native checkpoint payloads. The original-job reference checks match unmodified
PyTorch's batch values and model-side RNG progress. The final runtime/test source
was frozen before the paired run and verified unchanged afterward; the separate
authoring/export changes passed their real CLI/export checks afterward.

A one-second process observer started during the final run captured 275
test-owned descendant lifetimes, bound to PID creation times. None remained when
the suite exited. This is sampled normal-cleanup evidence, not crash-recovery
qualification.

The tested Python-code manifest SHA-256 is
`880a8e987ac6316092f83f7f6d09abff7713a8131f0b785d4025d58c1267e090`.
It hashes the sorted path-to-SHA-256 mapping for all 13 changed Python files.
The evidence bundle contains the full mapping, source patch, test XML and logs.

Reproduce the focused cases with:

```bash
python -m pytest -q tests/unit_test/examples/cifar_task_checkpoint_test.py tests/unit_test/examples/cifar10_fedprox_client_test.py tests/unit_test/examples/cifar_task_export_test.py
python -m pytest -q tests/integration_test/fast/cifar_task_state_test.py
./runtest.sh -s --skip-install
```

Retained failures distinguish an initial fixture packaging error (the `data`
directory was flattened), a preliminary adapter regression caught by the native
resident-trajectory test (corrected seed handling), and a process-observer setup
failure (system Python lacked psutil; the existing qualified venv succeeded).
No unexpected product failure was suppressed or marked xfail.

Offline model/data substitution is a bounded regression fixture, not full
CIFAR-10, PoC/production, GPU, performance or all-algorithm qualification.

SCAFFOLD, local-model submission, Lightning restoration, TensorFlow application
state and per-task external trainers remain required follow-up work. These are
unfinished acceptance cases, not approved omissions from the release scope.
