# Asynchronous FL with FedBuff and the Collab API

This example implements online buffered asynchronous federated learning on
CIFAR-10. Client jobs remain active across global-model updates, and the
server aggregates each full update buffer using FedBuff.

Download the example and enter its preserved Python package layout:

```bash
nvflare examples get collab-pt-async
cd collab-pt-async/collab/pt_async_cifar10
```

From an NVFlare source checkout, use
`cd examples/advanced/collab/pt_async_cifar10` instead.

## NVIDIA FLARE Installation

Add the PT extra to the same stable or nightly NVFlare distribution that
provided the example, then install its remaining dependencies:

```bash
# Stable installation
python -m pip install "nvflare[PT]"

# Nightly installation
python -m pip install "nvflare-nightly[PT]"

python -m pip install -r requirements.txt
```

For a source checkout, install that checkout from its repository root with
`python -m pip install -e ".[PT]"` before returning to this directory.

## Code Structure

```text
pt_async_cifar10/
├── async_aggregator.py  # CIFAR model, evaluation, and artifact hooks
├── trainer.py           # Published client-side local training
├── model.py             # FedAvg ModerateCNN, state and arithmetic helpers
├── cifar10_data.py      # Local CIFAR-10 download and Dirichlet splitter
├── prepare_data.sh      # Standalone CIFAR-10 download entry point
├── job.py               # FedBuffRecipe CLI and simulator wiring
└── requirements.txt
```

## Scheduling Parameters

The execution policy has three independent parameters:

| CLI argument | Meaning |
|---|---|
| `--concurrency` | Maximum concurrent client assignments |
| `--buffer-size` | Returned updates required to create a global model version |
| `--min-open-slots` | Free client slots required before replacement work is dispatched |

Aggregation happens before dispatch when a response reaches both the buffer
size and refill threshold. Replacement work therefore receives the newly
published model.

| Configuration | Behavior |
|---|---|
| `min_open_slots=1` | FedBuff: refill each completed slot immediately |
| `min_open_slots=buffer_size<concurrency` | Aggregate and refill in batches while other clients remain active |
| `min_open_slots=buffer_size=concurrency` | Synchronous FedAvg-style rounds |

The [FedBuff paper](https://proceedings.mlr.press/v151/nguyen22b/nguyen22b.pdf)
distinguishes concurrency from buffer size and uses K for buffer size. This
example uses full parameter names. The configurable refill threshold is an
implementation extension beyond immediate refill.

`concurrency` also sets the initial client-training parallelism. It may
be smaller than `--num-clients`, but cannot exceed available physical clients.
`--num-rounds` counts global aggregations, not dispatches. Defaults are eight
physical clients, eight active clients, buffer size four, minimum open slots
one and 100 aggregations. This matches the referenced eight-client, 50-round
FedAvg baseline's 400 accepted-update budget.

## Data

Use the included preparation script to download the dataset to `/tmp/cifar10`:

```bash
./prepare_data.sh
```

`job.py` then calls the included `split_and_save()` function. The defaults
create eight non-overlapping Dirichlet shards with
alpha 0.5 and seed 0 at:

```text
/tmp/cifar10_splits/cifar10_fedavg_8sites_alpha0.50_seed0/
├── site-1.npy
├── site-2.npy
└── ...
```

Use `--alpha` and `--split-seed` to change that partition. Keeping their
defaults makes the Collab and existing FedAvg jobs consume the exact same
eight shards.

## Model

The workflow uses the same `ModerateCNN` as the existing CIFAR-10 FedAvg
example. Every client trains from the exact global-model snapshot attached to
its assignment. An update can therefore be stale when it returns after newer
global versions have already been created.

## Client Code

`Cifar10Trainer.initialize()` uses `@collab.init` to load the prepared dataset
once. `Cifar10Trainer.train()` uses `@collab.publish`, which exposes local
training to the server's matching `collab.get_clients(...).train(...)` call.
Each response contains the trained model, metrics, assignment ID, and base
model version.

## Server-Side Workflow

`FedBuffRecipe` runs the shared `FedBuffWorkflow.execute()` entry point, marked
with `@collab.main`. `Cifar10ModelLoader` supplies the initial model, and
`Cifar10Observer` supplies evaluation and artifact hooks. The shared workflow:

1. fills the configured active-client slots;
2. invokes selected clients with a nonblocking Collab group call;
3. handles successful responses in arrival order;
4. refills when at least `min_open_slots` client slots are open;
5. creates a new global version whenever `buffer_size` updates are buffered; and
6. leaves other jobs active on the model versions they originally received.

The server computes each client delta relative to that assignment's base
model, weights updates by local training steps, and applies the FedBuff average
with `--server-lr` to match the existing FedAvg example.

See the [FedBuff recipe guide](../../../../docs/user_guide/data_scientist_guide/fedbuff_recipe.rst)
for configuration, client contracts, observer hooks, and component scope.

## Job Recipe Code

`job.py` constructs a `FedBuffRecipe` with a published client trainer,
`Cifar10ModelLoader`, `PTModelOps`, and a CIFAR evaluation/artifact observer. The recipe owns
assignment tracking, client concurrency, buffering and refill, and step-weighted buffered aggregation.
`async_aggregator.py` contains the model loader and observer. `recipe.execute(SimEnv(...))` starts the requested physical
simulator clients. Each physical client retains its optimizer, learning-rate
scheduler, data loader, and corresponding `site-<N>` shard across assignments.
The cosine schedule reaches its minimum after `--total-client-rounds`
assignments and remains there if asynchronous refill gives a faster client
additional assignments.

## Run Job

Run the default FedBuff configuration with eight active clients and buffer size four:

```bash
python job.py \
    --num-clients 8 \
    --alpha 0.5 \
    --split-seed 0 \
    --concurrency 8 \
    --buffer-size 4 \
    --min-open-slots 1 \
    --num-rounds 100
```

## Synchronous FedAvg Comparison

Set `concurrency=buffer_size=min_open_slots` to give this Collab workflow the same scheduling boundary as the
[standard CIFAR-10 FedAvg example](../../cifar10/pt/cifar10-sim/cifar10_fedavg/README.md):

The following configurations consume the same number of accepted client
updates, which makes them a useful first comparison:

| Workflow | Active clients | Buffer size | Minimum open slots | Global aggregations | Accepted updates |
|---|---:|---:|---:|---:|---:|
| FedBuff | 8 | 4 | 1 | 100 | 400 |
| Synchronous | 8 | 8 | 8 | 50 | 400 |

The shared splitter creates the same eight shards used by the existing
example. The Collab workflow also uses the same `ModerateCNN`, fixed
site-to-shard mapping, augmentation and normalization, batch size 64, four
local epochs, SGD with momentum, cosine learning-rate schedule, and
step-weighted aggregation.

```bash
# FedBuff: 4 updates/version x 100 versions = 400 updates.
python job.py \
    --num-clients 8 \
    --alpha 0.5 \
    --split-seed 0 \
    --concurrency 8 \
    --buffer-size 4 \
    --min-open-slots 1 \
    --num-rounds 100 \
    --total-client-rounds 50 \
    --workspace-root /tmp/nvflare/collab-fedbuff

# Synchronous: 8 updates/round x 50 rounds = 400 updates.
python job.py \
    --num-clients 8 \
    --alpha 0.5 \
    --split-seed 0 \
    --concurrency 8 \
    --buffer-size 8 \
    --min-open-slots 8 \
    --num-rounds 50 \
    --total-client-rounds 50 \
    --workspace-root /tmp/nvflare/collab-sync
```

In synchronous mode, all eight jobs receive one version, all eight must
complete to fill the buffer, and only then are eight jobs dispatched with the
next version.

### Evaluation for direct comparison

For direct comparison under different settings, every global model version
must be scored on the same evaluation population. Evaluating on whichever
client happens to receive an asynchronous version is not comparable: different
versions can be evaluated by different sites, and broadcasting every version
to every site would introduce a synchronization barrier into FedBuff.

The workflow therefore uses the union CIFAR-10 test set on the server. During
training, the server only snapshots each global version; it does not run
evaluation concurrently with client training. After training is complete, the
server evaluates every saved version on that common test set. Deferring
evaluation prevents evaluation work from changing client completion order,
open-slot timing, staleness, or the asynchronous participation sequence.

## Seeds and arrival timing

`--split-seed` controls the prepared shards; `--setup-seed` controls initial
model weights. `--run-seed` sets the server-side RNG after model setup. Client
training uses ordinary persistent loaders, optimizer and scheduler state; this
example does not impose per-assignment RNG scopes or replay.

Actual arrivals depend on training and RPC timing. `schedule.jsonl` records the
observed assignment IDs, physical/logical clients and base versions. For fixed
replay and assignment seed helpers, see the separate
[async research example](../pt_async_cifar10_reproducible/README.md). Its client contract can
be adopted explicitly when building a reproducible training experiment.

## Outputs

The default simulation workspace is
`/tmp/nvflare/collab/collab_pt_async_cifar10`. The server run directory
contains:

- `eval_snapshots/model_version_<N>.pt`
- `accuracy_history.json`
- `schedule.jsonl`
- `final_model/global_model.pt`
- `tensorboard/`

Additional round checkpoints are optional and controlled by
`--checkpoint-interval`.

View metrics with:

```bash
tensorboard --logdir /tmp/nvflare/collab/collab_pt_async_cifar10
```
