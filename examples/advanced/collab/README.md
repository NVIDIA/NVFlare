# Collab API Examples

The Collab API lets you write federated workflows as Python function calls
between sites. Start with the small examples below, then choose a workflow
that matches your application.

## Start here

New to Collab? Run [Hello FedAvg](../../hello-world/hello-collab/README.md)
first. It trains a small PyTorch model on synthetic data with two clients and
three rounds. It runs on a CPU and needs no dataset download.

From the NVFlare repository root, in an activated virtual environment:

```bash
python -m pip install -e ".[PT]"
python examples/hello-world/hello-collab/job.py
```

These examples use the Collab implementation in this repository; the command
above installs it together with PyTorch and torchvision. See the
[installation guide](https://nvflare.readthedocs.io/en/main/installation.html)
for environment setup.

Look for three rounds of client losses, `Job Status: FINISHED:COMPLETED`, and
`Results at:` with the simulation workspace. The synthetic data demonstrates
the API; for a model that learns to classify images, continue with
[Simple asynchronous MNIST](simple_async/README.md).

Read the Hello FedAvg [job.py](../../hello-world/hello-collab/job.py) in this
order: `Trainer.train()`, `FedAvg.run()`, then `make_recipe()`.

| API | What it does |
|---|---|
| `@collab.publish` | Exposes a function or method for another site to call. |
| `@collab.main` | Marks the single server entry point that coordinates the workflow. |
| `collab.clients.train(...)` | Calls the clients in parallel and waits for the group call to finish. |
| `collab.clients(blocking=False).train(...)` | Returns immediately; iterating the results waits for each next successful response. |
| `CollabRecipe` + `SimEnv` | Packages the application and runs the server and clients locally. |

Classes are optional: `simple_async` and `simple_split_learning` use plain
module-level functions. Call options such as `blocking=False` go on `clients`,
before the method name. Group results contain successful responses; inspect
`results.failures` for errors after consuming the result stream.

## Choose your next example

### Learn with small models

| Example | What you'll learn | Data and setup |
|---|---|---|
| [Hello FedAvg](../../hello-world/hello-collab/README.md) | Synchronous rounds, model averaging, and per-client configuration | Synthetic PyTorch data; no download |
| [Simple asynchronous MNIST](simple_async/README.md) | Nonblocking calls and immediately giving a finished client more work | MNIST downloaded automatically; CPU |
| [Simple split learning](simple_split_learning/simple_split_learning.py) | Exchanging activations and gradients between client and server | MNIST downloaded automatically; CPU |

### Explore coordination patterns

These use small NumPy arrays to focus on communication and control flow.

| Example | What you'll learn |
|---|---|
| [In-time aggregation](async_aggregation/async_aggregation.py) | Processing results with response callbacks within each round |
| [Swarm learning](swarm/swarm.py) | Decentralized training with client-to-client calls |

For the split-learning and NumPy examples, start from the repository root
and choose a module entry point below. Running from `examples/advanced` makes
the `collab` example package available to Python:

```bash
cd examples/advanced
python -m collab.simple_split_learning.simple_split_learning
python -m collab.async_aggregation.async_aggregation
python -m collab.swarm.swarm --num-clients 3
```

### Train with larger workflows

Follow each example's README for its dependencies, data preparation, and run
commands before launching training.

| Example | What you'll learn | Additional setup |
|---|---|---|
| [CIFAR-10: FedAvg, FedProx, and SCAFFOLD](pt_cifar10/README.md) | Comparing synchronous algorithms and their client/server state exchange | Prepare CIFAR-10 client shards |
| [Asynchronous CIFAR-10](pt_async_cifar10/README.md) | Buffered aggregation (FedBuff), client scheduling, and a synchronous comparison | Prepare logical-client shards; TensorBoard |
| [CIFAR-10 SplitNN](../vertical_federated_learning/cifar10-splitnn/README.md) | Two-party split learning with activation and gradient exchange | Prepare data; TensorBoard and PSI |
| [LLM supervised fine-tuning](pt_llm_sft/README.md) | Full-parameter Hugging Face training with server-side FedAvg | Prepare text data; Hugging Face dependencies and model resources |

## Adapt an example

Each example keeps its helpers in its own directory so it can be copied and
adapted independently. Use one `@collab.main` entry point per server and call
any workflow stages from it. The same recipe can run with `PocEnv` or `ProdEnv`
from `nvflare.recipe` after preparing the target environment and site data.

- [Migration tutorial](../../../docs/design/collab_api_migration_tutorial.md): adapt local training to Collab, with class-based and function-based versions.
- [Collab API design](../../../docs/design/collab_api_design.md): call semantics, context, lifecycle, and configuration.
