# Collab API Examples

The Collab API lets a server or client call ordinary published Python methods
at other sites. These self-contained examples progress from the basic
programming model to custom training algorithms and collaboration topologies.

## Start here

1. **Learn the basic calls:** [Hello FedAvg](../../hello-world/hello-collab/README.md)
   introduces `@collab.main`, `@collab.publish`, `collab.clients.train(...)`, and
   `CollabRecipe` with synchronous FedAvg on synthetic PyTorch data.
2. **Learn asynchronous execution:** [Simple async FL](simple_async/README.md)
   shows `collab.clients(blocking=False).train(...)`, arrival-driven model
   updates, and immediate client refill in one class-free MNIST example.
3. **Choose a direction below:** apply a training algorithm, explore another
   topology, or move to a larger model.

For the API concepts and call options, see the
[Collab API guide](../../../docs/user_guide/data_scientist_guide/collab_api.rst).

## Asynchronous calls and aggregation

Start with the small examples before moving to the full CIFAR-10 workflow.
A response callback by itself does not make a training loop asynchronous:
`blocking=False` releases the calling thread, while the workflow determines
whether clients can keep progressing across global-model updates.

| Example | What to learn | Setup |
|---|---|---|
| [Simple async FL](simple_async/README.md) | Nonblocking dispatch, model updates as results arrive, and immediate refill of the finished client while others keep training | PyTorch and torchvision, CPU; downloads MNIST on first run |
| [In-time aggregation](async_aggregation/avg_intime.py) | `process_resp_cb` processes results during a **blocking** call; training still has round boundaries | NumPy, synthetic updates, CPU |
| [CIFAR-10 FedBuff](pt_async_cifar10/README.md) | Buffered asynchronous FL with active-job limits, refill thresholds, model versions, and evaluation | PyTorch, torchvision, TensorBoard; prepare CIFAR-10 first |

## Federated training algorithms and larger models

The CIFAR-10 variants share a dataset and model so you can focus on the algorithm.
Read FedAvg first, then compare the client or server changes in FedProx and SCAFFOLD.

| Example | What to learn | Setup |
|---|---|---|
| [CIFAR-10 FedAvg](pt_cifar10/README.md#algorithm-variants) ([server](pt_cifar10/fedavg/server.py)) | Synchronous training and server aggregation using direct client calls | PyTorch and torchvision; prepare CIFAR-10 first |
| [CIFAR-10 FedProx](pt_cifar10/README.md#algorithm-variants) ([client](pt_cifar10/fedprox/client.py)) | Extend FedAvg client training with a proximal loss | Same CIFAR-10 setup |
| [CIFAR-10 SCAFFOLD](pt_cifar10/README.md#algorithm-variants) ([server](pt_cifar10/scaffold/server.py)) | Exchange model parameters and control variates | Same CIFAR-10 setup |
| [LLM supervised fine-tuning](pt_llm_sft/README.md) | Full-parameter Hugging Face SFT with PyTorch tensor exchange and server-side FedAvg | Hugging Face dependencies; prepare data first |

## Split learning and decentralized collaboration

These examples change who holds each part of the model or who coordinates the work.

| Example | What to learn | Setup |
|---|---|---|
| [Simple split learning](simple_split_learning/simple_split_learning.py) | Client-side images and bottom model, server-side labels and top model, with direct activation/gradient exchange | PyTorch and torchvision; downloads MNIST on first run |
| [CIFAR-10 SplitNN](../vertical_federated_learning/cifar10-splitnn/README.md) | Two-party split learning with direct activation/gradient return values | PyTorch, torchvision, TensorBoard, and PSI; prepare data first |
| [Swarm learning](swarm/swarm.py) ([workflow](swarm/swarm_algo.py)) | Decentralized coordination with client-to-client calls | NumPy, synthetic updates, CPU |

## Run the examples

The examples use an NVFlare installation from this repository. Install the
checkout from the repository root:

```bash
python -m pip install -e .
```

Install the MNIST example dependencies and run it directly:

```bash
python -m pip install -r examples/advanced/collab/simple_async/requirements.txt
cd examples/advanced/collab/simple_async
python job.py
```

The other small examples use package entry points. From the repository root:

```bash
cd examples/advanced   # makes the collab package available to module imports

python -m collab.async_aggregation.async_aggregation
python -m collab.swarm.swarm --num-clients 3
# Install PyTorch and torchvision before running the MNIST example:
python -m collab.simple_split_learning.simple_split_learning
```

For `pt_cifar10`, `pt_async_cifar10`, `pt_llm_sft`, and CIFAR-10 SplitNN, follow
that example's README for dependencies, data preparation, and launch commands.

## Shared conventions

- Each entry point constructs a `CollabRecipe` and runs it with the standard
  `SimEnv`. Use `PocEnv` or `ProdEnv` from `nvflare.recipe` for another deployment
  mode; Collab has no separate runner or environment abstraction.
- Each server object or module defines exactly one `@collab.main` entry point.
  A workflow with multiple stages calls them from that single entry point.
- Each example keeps its entry point, trainers, strategies, and utilities in
  its own directory. Helpers are intentionally local so an example can be
  copied or adapted independently.
