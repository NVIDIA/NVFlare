# Asynchronous FL with the Collab API

This example demonstrates asynchronous PyTorch training on CIFAR-10 with the
Collab API. It combines a ResNet-18 model, logical-client rotation, streamed
response aggregation, server evaluation, round checkpoints, and TensorBoard
metrics.

The data lifecycle follows the current standard workflow:

1. `prepare_data.py` downloads CIFAR-10 once and writes deterministic logical
   client index files.
2. `pt_async_cifar10.py` builds a `CollabRecipe`; clients only load prepared
   data and never download or repartition it during execution.

From the `examples/collab` directory, enter this example first:

```bash
cd pt_async_cifar10
```

## Install

Install the example dependencies:

```bash
python -m pip install -r requirements.txt
```

## Prepare data

We prepare for 10 clients for illustration:

```bash
python prepare_data.py \
    --num-clients 10 \
    --data-root /tmp/nvflare/datasets/cifar10
```

The output contains the torchvision CIFAR-10 cache, a `manifest.json`, and one
index file per logical client:

```text
/tmp/nvflare/datasets/cifar10/
├── cifar-10-batches-py/
├── manifest.json
└── splits/
    ├── site-1.npy
    ├── site-2.npy
    └── ...
```

Logical-client subsets use Dirichlet class skew and may overlap. This models a
large logical population whose participants independently sample local data
from the smaller CIFAR-10 training pool.

## Run the Collab recipe

The following runs two physical simulator clients. Each round assigns those
workers to two of the ten prepared logical clients:

```bash
python -m pt_async_cifar10 \
    --data-root /tmp/nvflare/datasets/cifar10 \
    --num-clients 2 \
    --num-rounds 3
```

To scale the simulation to 1,000 logical clients and 100 physical workers:

```bash
python prepare_data.py \
    --num-clients 1000 \
    --subset-size 350 \
    --data-root /tmp/nvflare/datasets/cifar10 \
    --overwrite

python -m pt_async_cifar10 \
    --data-root /tmp/nvflare/datasets/cifar10 \
    --num-clients 100 \
    --clients-per-round 100 \
    --num-rounds 10
```

Use `--overwrite` on `prepare_data.py` when intentionally regenerating an
existing split set.

## Outputs

The default simulation workspace is
`/tmp/nvflare/collab/collab_pt_async_cifar10`. The server run directory
contains:

- `round_models/model_round_<N>.pt`
- `final_model/global_model.pt`
- `tensorboard/`

View metrics with:

```bash
tensorboard --logdir /tmp/nvflare/collab/collab_pt_async_cifar10
```

## Collab API features

| Feature | Usage in this example |
|---|---|
| Client initialization | `@collab.init` loads the prepared CIFAR-10 data |
| Published client training | `@collab.publish` exposes the local training method |
| Server workflow | `@collab.main` drives sampling, training, aggregation, and evaluation |
| Targeted client groups | `collab.get_clients(names)` selects physical workers for a round |
| In-time aggregation | A response callback accumulates each update as it arrives |
| Runtime context | `collab.workspace` stores checkpoints and TensorBoard events |
| Standard execution | `CollabRecipe(...).execute(SimEnv(...))` runs the simulation |
