# Reproducible asynchronous CIFAR-10

This companion to [asynchronous CIFAR-10](../pt_async_cifar10/README.md)
trains the same ModerateCNN on Dirichlet CIFAR-10 shards with a standard
`FedBuffRecipe`. The shared execution port in `FedBuffWorkflow` exposes
reservation, acceptance, and publication to the simulator. This example
supplies the fixed-order or timed plan only to `SimEnv`; the exported recipe
has no replay arguments or simulator branches.

The folder includes its CIFAR model, data splitter, evaluator, and trainer so
the catalog download works independently of the basic example. Its trainer
accepts the plan's per-assignment seed, selects data by logical client name,
and creates a fresh loader and optimizer for each assignment. That reset makes
the result independent of which physical worker runs a logical assignment.
It differs from the basic example's persistent per-site optimizer and cosine
scheduler. The learning rate here follows a deterministic assignment-index
cosine schedule.

## Run

Install the same requirements as the basic example:

```bash
python -m pip install "nvflare[PT]"
python -m pip install -r requirements.txt
```

From the repository checkout:

```bash
cd examples/advanced/collab/pt_async_cifar10_reproducible
./prepare_data.sh
cd ..
python -m pt_async_cifar10_reproducible.job --num-rounds 1 --aggregation-epochs 1
```

For the catalog copy:

```bash
nvflare examples get collab-pt-async-reproducible
cd collab-pt-async-reproducible/collab/pt_async_cifar10_reproducible
./prepare_data.sh
cd ..
python -m pt_async_cifar10_reproducible.job --num-rounds 1 --aggregation-epochs 1
```

The CLI prepares eight logical CIFAR shards by default and runs them on two
physical simulator clients with eight logical assignments in flight. The
fixed plan swaps the first two accepted IDs; later IDs are accepted in order.
To compare physical worker counts, use separate workspace roots:

```bash
python -m pt_async_cifar10_reproducible.job --num-clients 1 --num-rounds 1 \
  --aggregation-epochs 1 --workspace-root /tmp/nvflare/cifar-replay-one
python -m pt_async_cifar10_reproducible.job --num-clients 2 --num-rounds 1 \
  --aggregation-epochs 1 --workspace-root /tmp/nvflare/cifar-replay-two
```

For a timed plan, use `--plan-mode timed`. Assignments start in logical
batches, with one virtual unit each for download, training, and upload.
`--horizon 2` stops before the first upload; the observer still saves the
initial model as the final model. All started physical calls are drained.

## Compose

```python
from pt_async_cifar10_reproducible.job import make_plan, make_recipe, define_parser, prepare_splits
from nvflare.recipe import SimEnv

args = define_parser().parse_args([])
args.train_idx_root = prepare_splits(args.num_logical_clients, args.alpha, args.split_seed)
recipe = make_recipe(args)
run = recipe.execute(SimEnv(num_clients=2, reprod=True, plan=make_plan(args)))
```

`make_plan()` supplies `num_rounds * buffer_size` logical assignments and
training seeds. The fixed order is compatible with the active logical-client
rule: a client is assigned again only after its earlier result is accepted.
The same `recipe` can run in `ProdEnv`, where the client derives the seed
from assignment ID and logical name. Production sites must have the prepared
shards at the configured path.

The simulator writes `assignments.jsonl`, `dispatch.jsonl`,
`accepted.jsonl`, retained `models/` and `results/`, and `run.jsonl`
under `<workspace-root>/<job>/server/simulate_job/simulation/`. Timed runs
also write `phases.jsonl`. The CIFAR observer writes model snapshots,
accuracy history, TensorBoard data, and `final_model/global_model.pt` in
the server run directory.

CPU runs on the same software stack are the intended comparison. Seeds and
server acceptance order do not guarantee identical GPU kernels, results
across library versions, or restart recovery. See the
[simulation guide](../../../../docs/user_guide/data_scientist_guide/reproducible_simulation.rst)
for the exact replay boundary.
