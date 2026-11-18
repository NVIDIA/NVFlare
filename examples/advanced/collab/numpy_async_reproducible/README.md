# Async with reproducible simulation

This dataset-free companion to the [basic NumPy FedBuff example](../numpy_async/README.md)
shows the complete `FedBuffRecipe` + `SimEnv(reprod=True, plan=...)` path. It uses
small NumPy models so replay can be checked without downloading a dataset or
installing PyTorch.

`job.py` constructs an ordinary `FedBuffRecipe`. Its workflow has no simulation
arguments or branches: the shared async execution port in `FedBuffWorkflow`
connects assignment reservation, accepted results, and model publication to
the simulator. `make_plan()` supplies fixed acceptance order or virtual timing
only to `SimEnv`. The same recipe can execute in `ProdEnv` with live arrivals.
No example-specific recipe subclass or scheduler is required.

## Run

From the repository root:

```bash
cd examples/advanced/collab
python -m numpy_async_reproducible.job --num-clients 2 --run-seed 10
python -m numpy_async_reproducible.job --num-clients 1 --run-seed 10 \
  --workspace-root /tmp/nvflare/collab-replay-one-worker
python -m numpy_async_reproducible.job --plan-mode timed --num-clients 2 \
  --workspace-root /tmp/nvflare/collab-replay-timed
```

The first two commands use four logical assignments with acceptance order
`[1, 0, 2, 3]`. They start with `alice` and `bob`; `bob` is assigned again only
after her first result is accepted, and likewise for `alice`. This preserves
the active-logical-client eligibility rule even when one physical worker
executes both clients. Assignment bases and final weights should match across
the two runs; physical dispatch records may differ. The timed command uses
explicit logical starts and constant download, train, and upload delays on a
virtual clock. Its horizon is 8, after all four planned uploads.

For a downloaded example, enter the parent of the package and run the same
module:

```bash
nvflare examples get collab-numpy-async-reproducible
cd collab-numpy-async-reproducible/collab
python -m numpy_async_reproducible.job --num-clients 2 --run-seed 10
```

## Compose

The complete program is in [job.py](job.py). Its essential wiring is:

```python
from numpy_async_reproducible.job import make_plan, make_recipe
from nvflare.recipe import SimEnv

recipe = make_recipe(run_seed=10)  # Returns FedBuffRecipe.
run = recipe.execute(SimEnv(
    num_clients=2,
    reprod=True,
    plan=make_plan(run_seed=10, mode="fixed"),
))
```

`make_plan()` lists logical names, acceptance IDs, and per-assignment training
seeds. In timed mode it replaces acceptance IDs with starts, delay profiles,
and a horizon. `SeedPolicy` derives independent seed streams from the run seed,
assignment ID, and logical name. `SeededNumpyClient.train()` applies the supplied
plan seed; without a plan it derives the same seed locally. A client with
persistent data or optimizer state must separate or reset it by logical name.
The NumPy model operations are example code and can be replaced for another
task. The recipe and plan can be imported independently for experiments.

The simulator stores the plan in the workspace and writes `assignments.jsonl`,
`dispatch.jsonl`, `accepted.jsonl`, retained `models/` and `results/`, and
`run.jsonl` under
`<workspace-root>/collab_numpy_async_reproducible/server/simulate_job/simulation/`.
Timed runs also write `phases.jsonl`. The observer saves `final_weights.npy` in
the server run directory. Use a fresh workspace root for independent runs.
These artifacts are run evidence; they are not restart checkpoints.

The [reproducible simulation guide](../../../../docs/user_guide/data_scientist_guide/reproducible_simulation.rst)
describes plan validation, timing, evidence, and determinism limits. The
[FedBuff recipe guide](../../../../docs/user_guide/data_scientist_guide/fedbuff_recipe.rst)
describes the client protocol and model operations. A fixed plan controls
acceptance, not client training kernels; matching weights also requires
deterministic client computations and data loading.

## PyTorch seed helpers

For a PyTorch client, `pt_seed.py` offers `torch_seed_scope()` and
`seed_worker()`. The client must call them with the assignment's `train_seed`;
importing the helpers alone has no effect. The scope restores Python, NumPy,
and PyTorch RNG states after training. CPU mode does not initialize CUDA; list
every GPU used explicitly. Recreate loader workers when assignment seeds change.
These helpers do not guarantee deterministic kernels or equality across
devices and software versions.
