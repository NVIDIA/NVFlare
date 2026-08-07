# PyTorch CIFAR-10 with the Collab API

This example demonstrates how to use NVIDIA FLARE's Collab API with PyTorch to
train a CIFAR-10 image classifier. It includes FedAvg, FedProx, and SCAFFOLD
workflows that use direct Python calls between the server and clients.

It is recommended to create a virtual environment before running the example.

## NVIDIA FLARE Installation

For complete installation instructions, see the
[NVIDIA FLARE installation guide](https://nvflare.readthedocs.io/en/main/installation.html).
From the root of this repository, install NVFlare and the example dependencies:

```bash
python -m pip install -e .
python -m pip install -r examples/advanced/collab/pt_cifar10/requirements.txt
```

## Code Structure

First get the example code from GitHub:

```bash
git clone https://github.com/NVIDIA/NVFlare.git
cd NVFlare
git switch <release branch>
cd examples/advanced
```

The example follows the client/server/job organization used by
[hello-pt](../../../hello-world/hello-pt/README.md):

```text
collab/pt_cifar10/
├── aggregation.py             # Reuses NVFlare weighted aggregation
├── prepare_data.py            # Standard CIFAR-10 Dirichlet splitter
├── data/                      # Splitter dependencies from cifar10/pt
│   ├── cifar10_data_utils.py
│   └── cifar10_dataset.py
├── loader.py                  # Loads each site's prepared split
├── model.py                   # PyTorch model definition
├── requirements.txt           # Example dependencies
├── fedavg/
│   ├── client.py              # Local training
│   ├── server.py              # Synchronous round loop
│   └── job.py                 # CollabRecipe and SimEnv
├── fedprox/
│   ├── client.py              # FedAvg client with proximal loss
│   └── job.py                 # Reuses the FedAvg server
└── scaffold/
    ├── client.py              # Local control variate
    ├── server.py              # Model and control aggregation
    └── job.py                 # CollabRecipe and SimEnv
```

Shared code stays at the package root. Each algorithm folder contains only the
behavior that differs.

## Data

This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
dataset. The preparation script and its supporting modules are copied from the
existing [`cifar10/pt`](../../cifar10/pt/README.md) example. It downloads the
dataset and creates disjoint client partitions with Dirichlet sampling.

From `examples/advanced`, prepare the two client splits used by the jobs:

```bash
python collab/pt_cifar10/prepare_data.py \
    --split_dir_prefix /tmp/cifar10_splits/pt_cifar10 \
    --num_sites 2 \
    --alpha 0.5
```

This writes the split to
`/tmp/cifar10_splits/pt_cifar10_2sites_alpha0.50_seed0`. The `--num_sites`
value must match `NUM_CLIENTS` in the job files. Lower `--alpha` values create
more heterogeneous client data.

## Model

The [model.py](model.py) file defines `SimpleNetwork`, the same small
convolutional network used by `hello-pt`. The server initializes this model,
and each client loads the global weights before local training. PyTorch uses an
available NVIDIA GPU and otherwise runs on the CPU.

## Client Code

The [FedAvg client](fedavg/client.py) is ordinary PyTorch training code with
three Collab interactions:

1. `@collab.init` initializes the model, optimizer, and prepared site data.
2. `@collab.publish` exposes `train` to the server.
3. `collab.site_name` selects the matching `site-N.npy` partition.

```python
# @collab.publish exposes train so the server can call it through collab.clients.
@collab.publish
def train(self, global_weights):
    self.model.load_state_dict(global_weights)
    num_steps = self._local_train()
    return {"weights": get_model_state(self.model), "num_steps": num_steps}
```

FedProx subclasses this client and changes only loss computation. SCAFFOLD
subclasses it to apply and update the local control variate.

## Server-Side Workflow

The [FedAvg server](fedavg/server.py) defines the synchronous workflow directly
with the Collab API:

1. Initialize the global model.
2. Call the published `train` method on all clients.
3. Aggregate the returned weights with NVFlare's existing
   `WeightedAggregationHelper`.
4. Evaluate the updated global model.

```python
# @collab.main marks the single server entry point that drives the workflow.
@collab.main
def run(self):
    ...
    # This calls every client's published train method and returns results by site.
    client_results = collab.clients.train(global_weights)
```

The SCAFFOLD server follows the same round loop and also exchanges global and
local control variates.

## Job Recipe Code

Each `job.py` connects its client and server objects with `CollabRecipe` and
runs the recipe with `SimEnv`. For example, FedAvg uses:

```python
recipe = CollabRecipe(
    job_name="collab_cifar10_fedavg",
    server=FedAvgServer(num_rounds=NUM_ROUNDS),
    client=FedAvgClient(),
    min_clients=NUM_CLIENTS,
)

run = recipe.execute(SimEnv(num_clients=NUM_CLIENTS))
```

The job files intentionally expose only `NUM_CLIENTS` and `NUM_ROUNDS`, keeping
the example focused on the Collab API.

## Algorithm Variants

| Algorithm | Difference from FedAvg |
|---|---|
| FedAvg | Trains locally and averages client model weights |
| FedProx | Adds a proximal term to the client loss and reuses the FedAvg server |
| SCAFFOLD | Exchanges control variates to correct client drift |

## Run Job

From `examples/advanced`, run any of the three jobs:

```bash
python -m collab.pt_cifar10.fedavg.job
python -m collab.pt_cifar10.fedprox.job
python -m collab.pt_cifar10.scaffold.job
```

Each job prints round accuracy, final job status, and the simulation result
location.
