# Advanced Collab API Examples

These examples demonstrate custom workflows built with the Collab programming
model. Start with
[Hello NumPy Collab](../../hello-world/hello-collab/README.md) for the smallest
introduction.

## Run

Each entry point builds a `CollabRecipe` and executes it with `SimEnv`:

```bash
cd examples/advanced

python -m collab.hello_fedavg.hello_fedavg
python -m collab.simple_split_learning.simple_split_learning
python -m collab.async_aggregation.async_aggregation
python -m collab.swarm.swarm --num-clients 3
python -m collab.pt_cifar10.fedavg.job
python -m collab.pt_cifar10.fedprox.job
python -m collab.pt_cifar10.scaffold.job
```

## Examples

| Example | Demonstrates |
|---|---|
| `hello_fedavg` | The Collab API in one file: `@collab.main`, `@collab.publish`, and `collab.clients.train(...)` |
| `simple_split_learning` | Direct activation and gradient exchange for split learning |
| `async_aggregation` | In-time aggregation with a response callback |
| `swarm` | Decentralized client-to-client calls |
| [`pt_cifar10/fedavg`](pt_cifar10/README.md#fedavg) | Synchronous PyTorch FedAvg with direct client calls |
| [`pt_cifar10/fedprox`](pt_cifar10/README.md#fedprox) | FedAvg client training extended with a proximal loss |
| [`pt_cifar10/scaffold`](pt_cifar10/README.md#scaffold) | Model and control-variate exchange with SCAFFOLD |
| [`pt_async_cifar10`](pt_async_cifar10/README.md) | Asynchronous PyTorch training with logical-client shards |

Every server object or module defines one `@collab.main` entry point. Client
operations callable by the workflow use `@collab.publish`. Related algorithm
variants may share small model, data, or aggregation modules at their suite
root.

To use another deployment mode, execute the same recipe with `PocEnv` or
`ProdEnv` from `nvflare.recipe`. Paths configured in a recipe must already
exist on the system where each server or client runs.

The NumPy examples run in a base installation. PyTorch examples require
PyTorch; the CIFAR-10 examples also require torchvision. The asynchronous
CIFAR-10 example additionally requires TensorBoard and its documented
prepared-data workflow.

These examples run against an NVFlare installation from this repository. Their
requirements files list only extra framework dependencies.

See the [Collab API design](../../../docs/design/collab_api_design.md) and
[migration tutorial](../../../docs/design/collab_api_migration_tutorial.md) for
more detail.
