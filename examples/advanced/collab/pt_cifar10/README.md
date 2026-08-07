# Synchronous CIFAR-10 with the Collab API

This example shows how a server coordinates PyTorch clients with direct Python
calls through the NVFlare Collab API. It implements FedAvg, FedProx, and
SCAFFOLD while keeping the communication points separate from ordinary model
training code.

## Installation

Install NVFlare from this repository, then install the example dependencies:

```bash
python -m pip install -r examples/advanced/collab/pt_cifar10/requirements.txt
```

## Code structure

The layout follows the client/server/job organization used by
[hello-pt](../../../hello-world/hello-pt/README.md):

```text
pt_cifar10/
├── aggregation.py          # Reuses NVFlare weighted aggregation
├── data.py                 # CIFAR-10 DataLoader
├── model.py                # hello-pt SimpleNetwork
├── fedavg/
│   ├── client.py           # Local training
│   ├── server.py           # Synchronous round loop
│   └── job.py              # CollabRecipe and SimEnv
├── fedprox/
│   ├── client.py           # FedAvg client plus proximal loss
│   └── job.py              # Reuses the FedAvg server
└── scaffold/
    ├── client.py           # Local control variate
    ├── server.py           # Model and control aggregation
    └── job.py              # CollabRecipe and SimEnv
```

Shared code stays at the package root. Algorithm folders contain only the
responsibilities that differ.

## Data

The example uses the same simple CIFAR-10 loading approach as hello-pt:
`torchvision.datasets.CIFAR10` downloads data under `/tmp/nvflare/data`.
For teaching purposes, every simulated client uses the same training dataset.
In a real deployment, each client would load its own local data.

Pre-download CIFAR-10 when several processes should not download it
concurrently.

## Key Collab interactions

A client publishes the method that the server may call:

```python
# @collab.publish exposes train so the server can call it through collab.clients.
@collab.publish
def train(self, global_weights):
    ...
```

The server marks one method as the workflow entry point:

```python
# @collab.main marks the single server entry point that drives the workflow.
@collab.main
def run(self):
    ...
```

Inside that entry point, this line invokes the published `train` method on
all clients and returns results keyed by site:

```python
client_results = collab.clients.train(global_weights)
```

The server passes the returned tensor dictionaries to NVFlare's existing
`WeightedAggregationHelper`; the example does not define another aggregation
framework.

## FedAvg

The FedAvg client loads the global model, runs ordinary PyTorch training, and
returns its model weights and local step count. The server calls every client,
aggregates their weights, and evaluates the new global model.

`fedavg/job.py` connects those objects:

```python
recipe = CollabRecipe(
    job_name="collab_cifar10_fedavg",
    server=FedAvgServer(num_rounds=NUM_ROUNDS),
    client=FedAvgClient(),
    min_clients=NUM_CLIENTS,
)
```

## FedProx

FedProx changes only client-side loss computation. Its client subclasses the
FedAvg client and adds the proximal term; its job reuses `FedAvgServer`.

## SCAFFOLD

SCAFFOLD adds a persistent local control variate to each client. Its published
`train` method receives both global model weights and global controls. The
SCAFFOLD server aggregates both returned dictionaries with the same NVFlare
weighted aggregation helper.

## Run

Run the examples from `examples/advanced` so the `collab` package is importable:

```bash
python -m collab.pt_cifar10.fedavg.job
python -m collab.pt_cifar10.fedprox.job
python -m collab.pt_cifar10.scaffold.job
```

The job files intentionally expose only `NUM_CLIENTS` and `NUM_ROUNDS`.
Training constants live next to the client training code, keeping the example
focused on the Collab API.
