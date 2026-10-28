# Async with research reproducibility

This is the research-reproducibility companion to the
[asynchronous CIFAR-10 example](../pt_async_cifar10/README.md). It keeps the
`pt_async_cifar10` naming and places the extended components in a separate
`pt_async_cifar10_reproducible` package.

`ResearchRecipe` owns fixed arrival replay, logical clients over fewer physical
workers, assignment seeds and append-only disk checkpoints. Its workflow reuses
the core numerical engine, Collab calls and lifecycle. These research options
are example-level APIs; production `AsyncRecipe` always uses actual arrivals and
direct base-model references. Both paths share update/aggregation arithmetic.

## Run

Download the example and enter the parent of its importable package:

```bash
nvflare examples get collab-pt-async-reproducible
cd collab-pt-async-reproducible/collab
python -m pt_async_cifar10_reproducible.job --num-clients 2 --run-seed 10
```

From the repository root:

```bash
cd examples/advanced/collab
python -m pt_async_cifar10_reproducible.job --num-clients 2 --run-seed 10
```

The dataset-free smoke job reserves K=3 logical assignments over two physical workers. It accepts
IDs `[2, 0, 1, 3]` with exact bases `[0, 0, 0, 1]` and applies seeded random
NumPy deltas. The run directory keeps every artifact on disk. No dataset or PyTorch is needed.

| Artifact | Contents |
| --- | --- |
| `run_config.json` | K/B/O, seed, arrival and logical assignment sequences. |
| `assignment_bases/<version>.pkl` | Immutable assignment base models. |
| `assignment_results/<id>.pkl` | Raw model/metrics/metadata results, retained after acceptance. |
| `published_models/<version>.npz` | Initial model and every global publication. |
| `schedule.jsonl` | Dispatch/accept records appended immediately, including training seeds. |
| `final_weights.npy` | Final NumPy weights. |

A small `DiskStore` writes bases once per version and results once per assignment
ID. Assignment handles are file paths; acceptance reloads raw results. Reads
retain every file, without an in-memory model cache or reference counts. Model arrays are materialized only while needed for calls
and aggregation; the current model/accumulator still live in RAM. All checkpoints
remain available for inspection even after failure. Use a fresh workspace for
each experiment; existing artifacts are never reused silently. Keep enough disk
space and remove old experiment directories when they are no longer needed.
The pickle checkpoints are produced and loaded by the same trusted job; they are
local storage, not a new Collab wire format or automatic restart mechanism.

For a smoke check, repeat with the same seed and `--num-clients 1`, using a
different `--workspace-root`. The accepted ID/base/seed trace and final array
should match. Physical dispatch order and worker names may differ. This tests
the research composition through the real Collab simulator, rather than adding
a separate research unit-test matrix.

## Import and compose

With this directory's parent on the Python path:

```python
from pt_async_cifar10_reproducible.replay import ReplayScheduler
from pt_async_cifar10_reproducible.seed_policy import SeedPolicy, SeededTrainArgs
from pt_async_cifar10_reproducible.workflow import ResearchRecipe

recipe = ResearchRecipe(
    job_name="experiment", application=MyApplication(), client=SeededClient(),
    model_adapter=MyAdapter(), num_updates=4, num_active_jobs=3,
    scheduler=ReplayScheduler([2, 0, 1, 3], ["alice", "bob", "alice", "carol"]),
    train_kwargs=SeededTrainArgs(SeedPolicy(10)),
)
```

`job.py` explicitly bundles this package for server and client export. Do the
same when using these helpers in another job. They can be copied or adapted
independently; they are example APIs rather than installed SDK guarantees.
See the [core contracts](../../../../docs/design/async_recipe.md) for
numerical contracts, ownership, deployment and failure behavior. The
[component scope table](../../../../docs/design/async_recipe.md#component-scope)
identifies what belongs in the core recipe, core engine, model helpers and this
research workflow.

## Replay semantics

`arrival_sequence` must permute the complete `num_updates * buffer_size` ID
budget and be compatible with K/O. Early results wait for their acceptance turn;
errors still stop the workflow immediately. Impossible orders raise an error.
`logical_sequence` names every assignment. Jobs receive their base at reservation
time, before a worker is available; physical workers can be reused while results
wait. K may exceed the physical pool.
Clients must keep persistent optimizer/data state by logical identity, or reset
it per assignment. The demonstration uses stateless local updates.

Record acceptance IDs in trace order, logical names/assignment metadata in ID
order, and K/B/O with the run configuration. Acceptance events report the current
version before the update. `assignment_metadata` can supply recorded training
seeds. This is fixed-order replay, not FedRevive's sampled phase simulator.

## Seed ownership

`SeedPolicy(run_seed)` derives stable 63-bit seeds from the versioned
`nvflare.seed.v1` JSON identity using SHA-256. Inputs include a named stream and
optional assignment ID/logical name; physical workers and Python `hash()` are
excluded. `python_rng` and `numpy_rng` return fresh caller-owned generators.
Retain one to advance its stream; requesting another restarts it. Use separate
stream names for training, data, delays, probes and KD so ablations do not
consume unrelated randomness.

`SeededTrainArgs` supplies `train_seeds={physical_site: seed}` through the research
workflow and records the seed in assignment metadata. An explicit
`metadata["train_seed"]` overrides derivation. The published client must accept
and apply that argument. Importing a helper alone enables no behavior.

For PyTorch clients, optional helpers remain in this example:

```python
from pt_async_cifar10_reproducible.pt_seed import seed_worker, torch_seed_scope

loader = DataLoader(
    dataset, shuffle=True, num_workers=2, persistent_workers=False,
    generator=torch.Generator().manual_seed(SeedPolicy(seed).seed("data")),
    worker_init_fn=seed_worker,
)
with torch_seed_scope(seed):  # Pass cuda_devices=[index] for CUDA training.
    train(model, loader)
```

The scope restores Python/NumPy/Torch RNG states after success/failure. CPU mode
does not initialize CUDA; list every GPU used explicitly. Cooperating scopes
serialize access to process-global RNGs; unrelated threads must not use those
RNGs concurrently. Recreate loader workers when assignment seeds change.
Seeds and replay do not enforce deterministic kernels or equality across
devices/software versions, nor restore optimizer state after restart.

## FedRevive follow-up

`ReplayScheduler` reserves logical jobs, dispatches `ScheduledCall` records and
selects ready results in the recorded order. `ResearchWorkflow` accepts them
through the shared engine. It selects `ResearchEngine` through the protected
`_engine_class` attribute and reuses core engine construction, Collab calls and
lifecycle. The research engine adds disk bases; numerical acceptance stays in core.

Sampled download/train/upload phases, virtual clocks and time horizons belong to
the later FedRevive migration. Implement those policies in this example when
needed. The current replay stops at the configured publication budget. Teacher/KD
components attach through `update_transform`.

FedRevive's reference linear training-seed mapping differs from this example's
hash mapping. Preserve its recorded seed values and generator algorithms during
migration, or document intentional changes. Sampled scheduling, tensor offload,
and full experiment migration remain tracked in Issue #5572.
