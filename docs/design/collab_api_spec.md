# FLARE Collab API -- User Guide and Specification

## 1. Overview

The **Collab API** (code name Fox -- FLARE Object Exchange) is NVIDIA FLARE's collaborative API that simplifies federated learning development. It allows researchers to write natural Python code while the framework handles distribution, communication, and orchestration.

### Design Goals

1. **Simplicity:** Write federated learning code as if it were local Python.
2. **Flexibility:** Support in-process simulation, multi-process POC, and distributed production.
3. **Transparency:** Hide infrastructure complexity from data scientists.
4. **Compatibility:** Minimal code changes when moving from simulation to production.
5. **Extensibility:** Plugin architecture for Collab Backends, tracking, and execution environments.

### Two API Patterns

The Collab API supports two client-side patterns:

| Pattern | Client Code | Use Case |
|---------|------------|----------|
| **Collab API** | `@collab.publish` decorator; server pushes data to client methods | Custom algorithm development, rapid prototyping, single-GPU in-process experiments |
| **Client API** | Standard `flare.receive()` / `flare.send()` loop | Multi-GPU (DDP), multi-node, production deployment, independent client/server development |

Both patterns use the **same server-side code** (`@collab.main`) and the **same `CollabRecipe`**. The Collab API is designed to help researchers **develop new federated learning algorithms** and package them into production-ready recipes.

### Key Concepts

| Concept | Description |
|---------|-------------|
| `@collab.main` | Decorator for the server-side main algorithm method (called once per job) |
| `@collab.publish` | Decorator for client-side methods published for remote invocation |
| `collab.clients` | Built-in proxy list; calls methods on all clients in parallel |
| `collab.site_name` | The current client's site name (injected by framework) |
| `CollabRecipe` | Job configuration -- ties server and client together |
| `SimEnv` | Unified simulation environment. Behind the scenes, uses the standard FLARE simulator for standard recipes, or the Collab simulation backend (pure function calls, no simulator overhead) for Collab recipes. |
| `PocEnv` | Multi-process local environment (CellNet) |

---

## 2. Usage Patterns

### 2.1 In-Process -- Single File (Single GPU)

The simplest pattern: server and client logic in a single Python file. Everything runs in-process via `SimEnv`, which for Collab recipes uses the Collab simulation backend (pure function calls, no simulator overhead).

**Single file with classes:**

```python
# fedavg_train.py -- server + client + execution in one file
from nvflare.collab import collab
from nvflare.recipe.sim_env import SimEnv
from nvflare.collab.sys.recipe import CollabRecipe

# --- Client --- (SimpleModel defined elsewhere or above; loss computed in training loop)
class Trainer:
    @collab.publish
    def train(self, weights=None):
        model = SimpleModel()
        if weights: model.load_state_dict(weights)
        # ... standard PyTorch training loop (computes loss) ...
        return model.state_dict(), loss.item()

# --- Server ---
class FedAvg:
    def __init__(self, num_rounds=5):
        self.num_rounds = num_rounds

    @collab.main
    def fed_avg(self):
        global_weights = None
        for r in range(self.num_rounds):
            client_results = collab.clients.train(global_weights)  # parallel call to all clients
            global_weights, avg_loss = weighted_avg(client_results)  # user-defined aggregation
        return global_weights

# --- Execute ---
if __name__ == "__main__":
    recipe = CollabRecipe(job_name="fedavg", server=FedAvg(num_rounds=5), client=Trainer(), min_clients=5)
    run = recipe.execute(SimEnv(num_clients=5))
```

**Single file with standalone functions (no classes):**

```python
# fedavg_no_class.py -- no classes needed!

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from nvflare.collab import collab
from nvflare.recipe.sim_env import SimEnv
from nvflare.collab.sys.recipe import CollabRecipe


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


@collab.publish
def train(weights=None):
    model = SimpleModel()
    if weights is not None:
        model.load_state_dict(weights)
    # ... training loop (computes loss) ...
    return model.state_dict(), loss.item()


NUM_ROUNDS = 5

@collab.main
def fed_avg():
    global_weights = None
    for r in range(NUM_ROUNDS):
        client_results = collab.clients.train(global_weights)
        global_weights, avg_loss = weighted_avg(client_results)  # user-defined aggregation
    return global_weights


if __name__ == "__main__":
    # CollabRecipe auto-detects @collab.main and @collab.publish in the caller's module
    recipe = CollabRecipe(job_name="fedavg", min_clients=5)
    run = recipe.execute(SimEnv(num_clients=5))
```

### 2.2 In-Process -- Separate Files (Single GPU)

For larger projects, split server and client into separate modules:

```
project/
  server.py       # @collab.main
  client.py       # @collab.publish
  model.py        # shared model definition
  job.py          # CollabRecipe ties them together
```

**model.py:** Standard PyTorch model definition.

**client.py:**

```python
from nvflare.collab import collab
from model import SimpleModel

@collab.publish
def train(weights=None):
    model = SimpleModel()
    if weights: model.load_state_dict(weights)
    # ... standard PyTorch training loop (computes loss) ...
    print(f"  [{collab.site_name}] Loss: {loss.item():.4f}")
    return model.state_dict(), loss.item()
```

**server.py:**

```python
from nvflare.collab import collab

@collab.main
def fed_avg():
    global_weights = None
    for r in range(5):
        client_results = collab.clients.train(global_weights)
        global_weights = weighted_avg(client_results)  # user-defined aggregation
    return global_weights
```

**job.py:**

```python
import server as server_module
import client as client_module
from nvflare.recipe.sim_env import SimEnv
from nvflare.collab.sys.recipe import CollabRecipe

recipe = CollabRecipe(
    job_name="fedavg_split",
    server=server_module,    # auto-wrapped with ModuleWrapper
    client=client_module,    # auto-wrapped with ModuleWrapper
    min_clients=5,
)
run = recipe.execute(SimEnv(num_clients=5))
```

### 2.3 Using Job Recipe (Recipe API)

The `CollabRecipe` extends the base `Recipe` class, so it supports all Recipe API features including execution environments:

```python
from nvflare.collab.sys.recipe import CollabRecipe
from nvflare.recipe.sim_env import SimEnv
from nvflare.recipe.poc_env import PocEnv

# Simulation (for Collab recipes: pure function calls, fast iteration)
recipe = CollabRecipe(job_name="fedavg", server=server, client=client, min_clients=5)
run = recipe.execute(SimEnv(num_clients=5))

# POC (multi-process, CellNet communication, more realistic)
recipe = CollabRecipe(job_name="fedavg", server=server, client=client, min_clients=5)
run = recipe.execute(PocEnv(num_clients=5))

# Export job for production deployment
recipe.export("/path/to/job_output")
```

---

## 3. Hybrid Patterns: Collab Server + Client API

### 3.1 Collab API on Server + Client API on Client (Single GPU)

This pattern uses the Collab API (`@collab.main`) on the **server** and the standard NVFlare Client API (`import nvflare.client as flare`) on the **client**.

**Why?** The Client API (`flare.receive()` / `flare.send()`) is the standard NVFlare pattern used in production. By using it on the client side, the training script is **portable** -- it works with Collab API, standard FLARE controllers, and production deployments with zero code changes.

#### Bridge Architecture: CollabClientAPI

Since the Collab server uses a **push** model (`collab.clients.execute(fl_model)`) and the Client API uses a **pull** model (`flare.receive()`), a bridge adapter (`CollabClientAPI`) connects the two:

```
Server (@collab.main)                       Client (Client API)
========================                    ========================

collab.clients.execute(fl_model)            flare.receive()
         |                                       ^
         v                                       |
  +-------------------------------------------------+
  |              CollabClientAPI (bridge)             |
  |                                                   |
  |  execute() is @collab.publish:                    |
  |    1. puts fl_model in _call_queue                |
  |    2. blocks on _result_queue                     |
  |                                                   |
  |  receive():                                       |
  |    gets fl_model from _call_queue                 |
  |                                                   |
  |  send(result):                                    |
  |    puts result in _result_queue                   |
  |    -> unblocks execute() -> returns to server     |
  |                                                   |
  |  stop():                                          |
  |    sets _stopped=True                             |
  |    -> is_running() returns False                  |
  +-------------------------------------------------+
         |                                       |
         v                                       v
collab.clients.stop()                       flare.is_running() -> False
```

**Two execution modes:**

| Mode | When | How the bridge works |
|------|------|---------------------|
| **In-process** (`launch_external_process=False`) | Single GPU, `SimEnv` | `CollabRecipe` sets `train_script` on `CollabClientAPI`. On first `execute()` call, the script is launched in a background thread. The script calls `flare.receive()` / `flare.send()` which route to the `CollabClientAPI` instance via `contextvars`. Queue-based handshake between `execute()` and `receive()`/`send()`. |
| **Subprocess** (`launch_external_process=True`) | Multi-GPU (torchrun DDP) | `SubprocessLauncher` spawns the training script as a child process (e.g. via `torchrun`). Communication between the `CollabClientAPI` (in the parent FLARE process) and the client script (in the subprocess) uses CellNet channels. Only rank 0 communicates with the server; other ranks sync via `dist.broadcast`. |

**How `CollabRecipe` wires it up:**

When you specify `train_script="client.py"` in `CollabRecipe`, internally:

1. `CollabRecipe` creates a `CollabClientAPI()` instance.
2. It calls `client_api.set_train_script("client.py", ...)` to register the script path.
3. The `CollabClientAPI` is passed to the executor as the client object.
4. When the server calls `collab.clients.execute(fl_model=...)`, this invokes the `CollabClientAPI.execute()` `@collab.publish` method.
5. `CollabClientAPI.execute()` starts the training script (first call only) and bridges `fl_model` to `flare.receive()` via queues.
6. The training script's `flare.send(result)` puts the result back in the queue, which `execute()` returns to the server.

This means the client training script (`client.py`) is **completely unaware** of the Collab API -- it only uses standard `import nvflare.client as flare`, making it portable to any NVFlare deployment.

**Server (server.py) -- uses `@collab.main`:**

```python
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.collab import collab

class FedAvg:
    def __init__(self, num_rounds=3):
        self.num_rounds = num_rounds

    @collab.main
    def run(self):
        global_weights = None
        for round_num in range(self.num_rounds):
            input_model = FLModel(params=global_weights, current_round=round_num + 1, total_rounds=self.num_rounds)
            client_results = collab.clients.execute(fl_model=input_model)  # dispatches to all clients
            global_weights = self._aggregate(client_results)  # user-defined; omitted for brevity
        collab.clients.stop()  # signals flare.is_running() -> False
        return global_weights
```

**Client (client.py) -- uses standard Client API:**

```python
import nvflare.client as flare  # SAME import as standard NVFlare!
from nvflare.app_common.abstract.fl_model import FLModel

def training_loop():
    flare.init()

    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break

        # Load global weights, train locally (SimpleModel defined elsewhere)
        model = SimpleModel()
        if input_model.params: model.load_state_dict(input_model.params)
        # ... standard PyTorch training loop ...

        flare.send(FLModel(params=model.state_dict(), metrics={"loss": loss.item()}))
```

**Job (job.py) -- uses `CollabRecipe` with `train_script`:**

```python
from server import FedAvg
from nvflare.recipe.sim_env import SimEnv
from nvflare.collab.sys.recipe import CollabRecipe

recipe = CollabRecipe(
    job_name="fedavg_client_api",
    server=FedAvg(num_rounds=3),
    train_script="client.py",   # Client API pattern (not client= object)
    min_clients=2,
)
run = recipe.execute(SimEnv(num_clients=2))
```

**Key points:**

- `train_script="client.py"` tells `CollabRecipe` to use the Client API pattern.
- The client script uses `import nvflare.client as flare` -- the same code that works in standard NVFlare production.
- The server calls `collab.clients.execute(fl_model=...)` which triggers the client's `flare.receive()`.
- The server calls `collab.clients.stop()` which causes `flare.is_running()` to return `False`.

### 3.2 Collab API on Server + Client API on Client + Multi-GPU (DDP)

For multi-GPU training, use `launch_external_process=True` with a `torchrun` command. The client script uses standard PyTorch DDP + NVFlare Client API.

**Server (server.py) -- identical to 3.1 (no changes for multi-GPU).** The server is GPU-agnostic.

**Client (client.py) -- standard DDP + NVFlare Client API:**

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    flare.init(rank=str(rank))

    net = SimpleModel().to(rank)  # SimpleModel defined elsewhere
    ddp_model = DDP(net, device_ids=[rank])  # wrap once, outside the loop

    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break

        # Load global weights (rank 0), broadcast to all ranks
        # Note: loading into net (unwrapped) is fine here since DDP wraps
        # the same object; in production, use ddp_model.module.load_state_dict()
        if rank == 0 and input_model.params:
            net.load_state_dict(input_model.params)
        for param in net.parameters():
            dist.broadcast(param.data, src=0)

        # Train with ddp_model (already wrapped above)
        # ... training loop with DistributedSampler ...

        # Only rank 0 sends result back
        if rank == 0:
            flare.send(FLModel(params=net.state_dict(), metrics={"loss": loss.item()}))

    dist.destroy_process_group()
```

**Job (job.py) -- uses `launch_external_process` for torchrun:**

```python
from server import FedAvg
from nvflare.recipe.sim_env import SimEnv
from nvflare.collab.sys.recipe import CollabRecipe

recipe = CollabRecipe(
    job_name="fedavg_ddp",
    server=FedAvg(num_rounds=3),
    train_script="client.py",
    min_clients=2,
    launch_external_process=True,                # subprocess mode
    command="torchrun --nproc_per_node=4",       # 4 GPUs per client
    subprocess_timeout=600.0,                    # 10 min timeout for DDP
)
run = recipe.execute(SimEnv(num_clients=2))
```

**Key differences from 3.1:**

| | Single GPU (3.1) | Multi-GPU DDP (3.2) |
|---|---|---|
| `launch_external_process` | `False` (default) | `True` |
| `command` | Not needed | `"torchrun --nproc_per_node=4"` |
| Client imports | `import nvflare.client as flare` | Same + `torch.distributed` |
| Client training | Standard single-GPU loop | `DistributedDataParallel` + `DistributedSampler` |
| Model sync | Not needed | `dist.broadcast()` or checkpoint-based sync |
| Result send | `flare.send(result)` | Only `rank == 0` sends |
| Server code | Unchanged | **Unchanged** -- server is GPU-agnostic |

---

## 4. Progression Paths: From Collab API to Production

This section describes the progression paths from a Collab API prototype toward production deployment. The path you choose depends on whether your algorithm matches an existing standard recipe or is a novel approach that requires custom server-side logic.

### 4.1 When to Progress Beyond Pure Collab API

The Collab API is designed to help researchers **develop new federated learning algorithms**. For single-GPU experiments with `@collab.publish` on the client, it is the fastest iteration loop. However, as your algorithm matures, you may want to:

- **Support multi-GPU / multi-node** -- requires Client API on the client side (DDP with `torchrun`).
- **Decouple server and client development** -- Client API lets teams work independently.
- **Use a standard algorithm** -- if your algorithm turns out to match FedAvg, Cyclic, etc., a built-in recipe is simpler and battle-tested.

| Scenario | Recommended Path |
|----------|-----------------|
| Novel algorithm, single GPU | Stay with Collab API on both sides -- fast iteration via `SimEnv` (Collab backend), pure Python function calls, almost free of FLARE concepts |
| Novel algorithm, multi-GPU or production | Develop new recipe: Collab API server + Client API client, wired via Job API -- see Section 4.4 (open: packaging tool needed) |
| Algorithm matches FedAvg/Cyclic/etc. | Switch to standard recipe (`FedAvgRecipe`, `CyclicRecipe`) -- see Section 4.2 |

### 4.2 Transition to Standard Recipe (When Your Algorithm Matches a Built-in)

This path applies **only when your algorithm matches an existing standard recipe** (e.g. your `@collab.main` is doing scatter-and-gather with weighted averaging -- that is exactly what `FedAvgRecipe` does). If your algorithm is novel, see **Section 4.4** instead.

**Step 1: Extract client training into a standalone script using Client API.**

Before (Collab API):

```python
class Trainer:
    @collab.publish
    def train(self, weights=None):
        model = SimpleModel()
        if weights: model.load_state_dict(weights)
        # ... training (computes loss) ...
        return model.state_dict(), loss.item()
```

After (Client API -- `train.py`):

```python
import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel

flare.init()
while flare.is_running():
    input_model = flare.receive()
    if input_model is None:
        break
    model = SimpleModel()
    if input_model.params:
        model.load_state_dict(input_model.params)
    # ... training ...
    flare.send(FLModel(params=model.state_dict(), metrics={"loss": loss.item()}))
```

**Step 2: Replace `CollabRecipe` with a standard Recipe (only if your algorithm matches one).**

Before (CollabRecipe):

```python
from nvflare.collab.sys.recipe import CollabRecipe

server = FedAvg(num_rounds=5)
client = Trainer()
recipe = CollabRecipe(job_name="fedavg", server=server, client=client, min_clients=5)
run = recipe.execute(SimEnv(num_clients=5))
```

After (FedAvgRecipe):

```python
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe.sim_env import SimEnv

recipe = FedAvgRecipe(
    name="fedavg",
    min_clients=5,
    num_rounds=5,
    train_script="train.py",
    model=SimpleModel(),
)
recipe.execute(SimEnv(num_clients=5))
```

**Step 3: Remove the custom server aggregation (the standard Recipe handles it).**

When using a standard recipe, the `@collab.main` method with manual aggregation is replaced by the recipe's built-in workflow (e.g. scatter-and-gather with `InTimeAccumulateWeightedAggregator` for FedAvg). If your aggregation logic does something the standard recipe does not support, you should **keep `CollabRecipe`** with Client API on the client side (see Section 4.4).

**Step 4: Deploy to production.**

```python
from nvflare.recipe.poc_env import PocEnv

# Export job for production FLARE deployment
recipe.export("/path/to/job_output")
# Or execute with PocEnv for local multi-process testing
recipe.execute(PocEnv(num_clients=5))
```

### 4.3 Progression Summary

| Component | Collab API (both sides) | New Recipe (Collab Server + Client API) | Standard Recipe (FedAvg, Cyclic, etc.) |
|-----------|------------------------|----------------------------------------|---------------------------------------|
| **Server algorithm** | `@collab.main` (custom) | `@collab.main` (custom, wired into recipe) | Built-in workflow (e.g. scatter-and-gather) |
| **Client training** | `@collab.publish` methods | `flare.receive()` / `flare.send()` script | `flare.receive()` / `flare.send()` script |
| **Job config** | `CollabRecipe(server=..., client=...)` | New recipe via Job API (packaging TBD) | `FedAvgRecipe(train_script=..., model=...)` |
| **Multi-GPU** | Not supported (single process) | Supported | Built-in via Recipe's launcher config |
| **Production deploy** | `recipe.export(...)` / `recipe.execute(...)` | Same | Same |
| **When to use** | Rapid prototyping, algorithm experiments | Novel algorithm -> production recipe | Algorithm matches a built-in recipe |

### 4.4 Intermediate Step: Collab API Server + Client API Client

#### API Layering (Background)

NVFlare has three levels of server-side API:

```
High-level:    Collab API  (@collab.main, @collab.publish)   <-- promoted API for new development
Mid-level:     ModelController API                            <-- no longer promoted once Collab API ships
Low-level:     Controller API + Executor API                  <-- internal plumbing
```

The Collab API is itself **built on top of** the Controller and Executor APIs. It is not a separate system -- it is a higher-level abstraction over the same underlying infrastructure.

Today's standard recipes (`FedAvgRecipe`, `CyclicRecipe`, `SimpleSwarmLearningRecipe`, etc.) were built using the **ModelController API** (mid-level). The Job Recipe API wires the server-side controller together with a client-side training script (via Client API) to produce a complete job definition. Users of standard recipes do not write server-side logic -- it is pre-built and battle-tested.

Going forward, **new FL algorithms and recipes should be developed using the Collab API** (high-level). The Collab API replaces the ModelController as the recommended way for researchers to develop and iterate on new federated learning algorithms.

#### The Collab API Approach to Recipe Building

The Collab API is designed for **research and new recipe development**. The typical workflow is:

1. **Research phase (Collab API everywhere):** Use `@collab.main` on the server and `@collab.publish` on the client. Both server and client run in the same process. Fast iteration in `SimEnv` (which uses the Collab simulation backend -- pure function calls, no simulator overhead).

2. **Hybrid phase (Collab API server + Client API client):** Once the server-side algorithm is stable, switch the client to the **Client API** pattern (`train_script` with `flare.receive()` / `flare.send()`). This enables multi-GPU, multi-node, and decoupled development.

3. **New Recipe creation:** Package the Collab API server logic and Client API client into a **new named recipe** (e.g. `SplitLearningRecipe`) using the **Job API**. Note: this step uses the Job API to wire up the components -- **not** `CollabRecipe`. The end product is a proper recipe that other users can consume like `FedAvgRecipe`, without understanding Collab API internals.

```
Phase 1 (Research):    CollabRecipe + @collab.main + @collab.publish  (SimEnv with Collab backend, pure function calls)
Phase 2 (Hybrid):      @collab.main server + Client API client        (wiring mechanism TBD -- see open question below)
Phase 3 (New Recipe):  Job API packages server + client -> SplitLearningRecipe(...)  (end-user-facing)
```

**Open question -- Phase 2 wiring:** How the Collab API server and Client API client are wired together in Phase 2 is an open design question. Options include:
- Reuse `CollabRecipe` with `train_script` (current prototype approach)
- A per-algorithm recipe class (e.g. `SplitLearningRecipe` created in Phase 3 could also serve Phase 2)
- A new intermediate mechanism or base class

The right answer may differ per algorithm and is not yet decided.

**Example:** Developing a new `SplitLearningRecipe`:

```python
# Phase 1: Research -- develop the algorithm with CollabRecipe
class SplitLearningServer:
    def __init__(self, num_rounds=5):
        self.num_rounds = num_rounds

    @collab.main
    def run(self):
        server_weights = initialize_server_model()  # user-defined
        for round_num in range(self.num_rounds):
            activations = collab.clients.forward(server_weights)
            gradients = self._server_backward(activations)  # user-defined; omitted for brevity
            collab.clients.backward(gradients)

# trainer: a class or module with @collab.publish methods, defined elsewhere
recipe = CollabRecipe(job_name="split_learning", server=SplitLearningServer(num_rounds=5), client=trainer, min_clients=3)
run = recipe.execute(SimEnv(num_clients=3))  # Collab backend: pure function calls, fast iteration

# Phase 2: Hybrid -- switch client to Client API for multi-GPU
# How server + client are wired is an open question (CollabRecipe is one option):
recipe = CollabRecipe(
    job_name="split_learning",
    server=SplitLearningServer(num_rounds=5),
    train_script="split_client.py",   # Client API
    min_clients=3,
)  # or a per-algorithm recipe, or a new mechanism -- TBD

# Phase 3: Package into a new recipe via Job API (NOT CollabRecipe)
# The Job API wires the Collab server + Client API client into a recipe class.
# --> SplitLearningRecipe(name=..., num_rounds=..., train_script=..., model=...)
#     End users use this like FedAvgRecipe -- no Collab API knowledge needed
```

**Open requirement -- Recipe packaging tool:** Today, creating a new named recipe from Collab API server logic + Client API client requires using the Job API to build every piece from scratch (similar to how `CollabRecipe` itself was built). There is a need for a **packaging tool or base class** that makes it easy for researchers to turn their Collab API server code + Client API client into a reusable recipe without having to re-implement all the wiring manually. The design of this tool is an open question.

There is no need to "translate down" to the Controller API. The Collab API is built on top of the Controller and Executor APIs internally.

#### Why Not Use Collab API for Both Server and Client?

For **single-process / single-GPU** training, you absolutely can -- and should -- use Collab API on both sides. This is the fastest way to prototype (`SimEnv` with the Collab backend runs everything as pure function calls in one process).

However, for **multi-GPU (DDP) and multi-node GPU** training, this becomes challenging:

- **Multi-GPU requires subprocess launching.** DDP training uses `torchrun` (or equivalent), which spawns separate processes. The Collab API's in-process `@collab.publish` mechanism does not naturally span across `torchrun`-launched subprocesses.
- **Client API is subprocess-native.** The Client API (`flare.receive()` / `flare.send()`) is designed to work inside externally launched processes. `CollabRecipe` with `launch_external_process=True` handles this by bridging the Collab server to a Client API subprocess (see Section 3.1, CollabClientAPI bridge).

Additionally, **Client API enables independent development:**

- **Collab API** requires server and client to share a tight contract -- the `@collab.publish` method signatures must exactly match what `collab.clients.<method>()` calls. Server and client code are coupled.
- **Client API** decouples the two sides. The client only knows about `FLModel` (`flare.receive()` / `flare.send()`). The server and client can be developed, tested, and deployed by different teams with no shared code beyond the model format.

#### Example: Hybrid Pattern

```python
# Server: custom aggregation via Collab API
class MyAggregation:
    def __init__(self, num_rounds=5):
        self.num_rounds = num_rounds

    @collab.main
    def run(self):
        model = initial_weights()
        for round_num in range(self.num_rounds):
            results = collab.clients.train(model)
            model = my_custom_aggregate(results)  # user's novel algorithm

# Wire into a recipe with Client API client
recipe = CollabRecipe(
    job_name="custom_fedavg",
    server=MyAggregation(num_rounds=5),
    train_script="train.py",   # Client API pattern (flare.receive / flare.send)
    min_clients=5,
    launch_external_process=True,
    command="torchrun --nproc_per_node=4",
)

# Works in any environment
run = recipe.execute(SimEnv(num_clients=5))   # or PocEnv, ProdEnv
```

---

## 5. Execution Environments

### 5.1 Backward Compatibility and Explicit API Specification

**Requirements:**

1. **Backward compatible:** Existing recipes that do not use the Collab API must continue to work exactly as before with no changes.
2. **Collab Backend instantiation:** When the Collab API is involved, the execution environment must instantiate a **Collab Backend** to handle `@collab.main` / `@collab.publish` dispatch.
3. **Explicit, not automatic:** There is **no auto-matching or auto-detection magic**. The use of Collab API must be **explicitly specified** by the recipe (not inferred by scanning code for annotations). The recipe itself carries the information about which API it uses. The user writes the same `SimEnv(...)` / `PocEnv(...)` for all recipes -- the environment consults the recipe to determine which backend to instantiate.

The mechanism by which the recipe indicates its API type (e.g. a flag, a base class check, metadata, or another mechanism) is an **open question**. What is **not** acceptable: any form of implicit detection or code scanning.

### 5.2 Expected Environment Behavior

The user writes the **same environment** (`SimEnv`, `PocEnv`, `ProdEnv`) regardless of recipe type. Behind the scenes, the environment consults the recipe to choose the appropriate backend.

| Environment | Standard Recipe (e.g. `FedAvgRecipe`) | Collab Recipe (`CollabRecipe`) |
|------------|------|------|
| **`SimEnv`** | Uses the existing FLARE simulator (multi-threaded, full simulator infrastructure) | Uses the **Collab simulation backend**: pure function calls within the same process -- much faster, no simulator overhead |
| **`PocEnv`** | Uses existing FLARE multi-process infrastructure (CellNet) | Same `PocEnv` + **Collab FLARE Backend**: handles `@collab.main`/`@collab.publish` dispatch over CellNet |
| **`ProdEnv`** | Uses existing FLARE deployment | Same `ProdEnv` + **Collab FLARE Backend**: handles Collab API within deployed FLARE infrastructure |

### 5.3 Usage Examples

```python
from nvflare.recipe.sim_env import SimEnv

# --- Standard recipe: SimEnv uses the FLARE simulator behind the scenes ---

recipe = FedAvgRecipe(name="fedavg", min_clients=5, train_script="train.py", model=model)
run = recipe.execute(SimEnv(num_clients=5))    # standard FLARE simulator
run = recipe.execute(PocEnv(num_clients=5))    # multi-process FLARE


# --- Collab recipe: same SimEnv, but behind the scenes uses Collab simulation backend ---

recipe = CollabRecipe(job_name="split_learning", server=server, client=client, min_clients=5)
run = recipe.execute(SimEnv(num_clients=5))    # Collab backend: pure function calls, fastest iteration

# POC: same PocEnv, Collab FLARE Backend instantiated behind the scenes
env = PocEnv(num_clients=5)
run = recipe.execute(env)
env.stop(clean_poc=True)

# Production: export job
recipe.export("/path/to/exported_job")
```

### 5.4 SimEnv Internal Backends

`SimEnv` selects its backend based on the recipe it executes. The user does not need to choose.

| | Standard Backend (standard recipes) | Collab Backend (Collab recipes) |
|---|---|---|
| **Execution model** | Full FLARE simulator with job submission, task dispatch, component lifecycle | Pure function calls: `@collab.main` calls `@collab.publish` directly |
| **Speed** | Slower (simulator overhead) | Much faster (no simulator infrastructure) |
| **Fidelity** | High: exercises full FLARE job pipeline | Lower: no job submission, no CellNet, no component lifecycle |
| **Use case** | Testing standard recipes, integration testing | Rapid algorithm prototyping, unit testing custom logic |
| **Multi-GPU** | Supported via subprocess launcher | Supported via `launch_external_process=True` |

---

## 6. Relationship Between CollabRecipe and Standard Recipes

### 6.1 Class Hierarchy

All recipes share a common base class (`Recipe`) which provides shared infrastructure (export, execute, filters, decomposers, config). `CollabRecipe` and the standard recipes are **siblings**, not parent-child:

```
Recipe (base class -- nvflare.recipe.spec)
  |
  |-- CollabRecipe (nvflare.collab.sys.recipe)
  |     Server: user-written @collab.main method  [Collab API -- high-level, promoted]
  |     Client: @collab.publish methods or Client API train_script
  |
  |-- FedAvgRecipe (nvflare.app_opt.pt.recipes.fedavg)
  |     Server: built-in scatter-and-gather  [ModelController API -- mid-level, legacy]
  |     Client: Client API train_script
  |
  |-- CyclicRecipe (nvflare.app_opt.pt.recipes.cyclic)
  |     Server: built-in cyclic workflow  [ModelController API -- mid-level, legacy]
  |     Client: Client API train_script
  |
  |-- SimpleSwarmLearningRecipe (nvflare.app_opt.pt.recipes.swarm)
  |     Server: built-in swarm workflow  [ModelController API -- mid-level, legacy]
  |     Client: Client API train_script
  |
  |-- KMeansFedAvgRecipe, SVMFedAvgRecipe, ... (sklearn recipes)
  |-- FedStatsRecipe (federated statistics)
  |-- NumpyFedAvgRecipe, FedAvgLrRecipe, ... (numpy recipes)

API Layering (all built on the same low-level Controller + Executor APIs):
  High-level:   Collab API (@collab.main)           <-- promoted for new development
  Mid-level:    ModelController API                  <-- used by existing standard recipes
  Low-level:    Controller API + Executor API        <-- internal infrastructure
```

**Note:** `CollabRecipe` is not just for prototyping. It is the primary mechanism for researchers to **develop new federated learning algorithms** using the Collab API. When paired with the Client API on the client side (`train_script`), a `CollabRecipe` is a production-ready recipe that supports multi-GPU, multi-node, and full FLARE deployment. Existing standard recipes (`FedAvgRecipe`, etc.) were built using the ModelController API (a mid-level API). Going forward, the Collab API is the promoted high-level API for building new recipes -- it is built on top of the same Controller and Executor APIs internally, so there is no need to drop to lower-level APIs for production use.

### 6.2 What Each Recipe Controls

| Layer | CollabRecipe | FedAvgRecipe / CyclicRecipe / etc. |
|-------|-------------|-------------------------------------|
| **Server algorithm** | User-written `@collab.main` method -- full control over orchestration, aggregation, round logic | Built-in workflow (e.g. scatter-and-gather, cyclic round-robin) -- user only sets parameters (num_rounds, etc.) |
| **Client training** | `@collab.publish` methods (Collab API) **or** `train_script` (Client API) | `train_script` only (Client API) |
| **Aggregation** | User code inside `@collab.main` (manual `weighted_avg()`, etc.) | Built-in aggregator (e.g. `InTimeAccumulateWeightedAggregator`) |
| **Model handling** | User manages model weights directly (dicts, state_dicts) | Recipe manages model persistence (e.g. `PTFileModelPersistor`) |
| **Multi-GPU** | `launch_external_process=True` + `command="torchrun ..."` | Built-in launcher configuration |
| **Execution envs** | `SimEnv`, `PocEnv`, `export()` | `SimEnv`, `PocEnv`, `export()` |

### 6.3 When to Use Which

| Use Case | Recommended Recipe | Why |
|----------|--------------------|-----|
| **Developing a new FL algorithm** | `CollabRecipe` | Pure Python, fast iteration, almost free of FLARE concepts |
| **Standard FedAvg training** | `FedAvgRecipe` | Battle-tested workflow, no server code needed |
| **Cyclic training** | `CyclicRecipe` | Built-in round-robin orchestration |
| **Swarm learning** | `SimpleSwarmLearningRecipe` | Built-in peer-to-peer aggregation |
| **Custom aggregation (e.g. weighted by data quality)** | `CollabRecipe` | User controls aggregation logic in `@collab.main` |
| **Custom communication pattern (e.g. gossip, async)** | `CollabRecipe` | User controls when and how to call clients |
| **Production deployment of a standard algorithm** | `FedAvgRecipe` / `CyclicRecipe` | Proven, optimized, documented |
| **Novel algorithm -> new recipe** | Research with `CollabRecipe`, then wire Collab server + Client API client into a new recipe via Job API | Collab API for research -> new recipe for production; see Section 4.4 |
| **Algorithm matches a standard recipe** | Migrate to `FedAvgRecipe` / `CyclicRecipe` (see Section 4.2) | When your algorithm matches a built-in recipe |

### 6.4 Key Difference: Who Writes the Server Logic?

**CollabRecipe -- you write the server algorithm:**

```python
class MyAlgorithm:
    def __init__(self, num_rounds=5):
        self.num_rounds = num_rounds

    @collab.main
    def run(self):
        # YOU control the entire workflow
        weights = initialize_model()
        for r in range(self.num_rounds):
            results = collab.clients.train(weights)      # call clients
            weights = my_custom_aggregation(results)     # your aggregation
            if should_early_stop(weights):               # your stopping logic
                break
        return weights

recipe = CollabRecipe(job_name="custom", server=MyAlgorithm(num_rounds=5), client=Trainer(), min_clients=5)
# For production: use train_script instead of client= for Client API pattern
# recipe = CollabRecipe(job_name="custom", server=MyAlgorithm(num_rounds=5), train_script="train.py", min_clients=5)
```

**FedAvgRecipe -- the workflow is built-in, you only configure it:**

```python
recipe = FedAvgRecipe(
    name="fedavg",
    min_clients=5,
    num_rounds=10,
    train_script="train.py",
    model=SimpleModel(),
)
# No server code needed -- FedAvgRecipe provides scatter-and-gather + weighted averaging
```

### 6.5 Shared Capabilities (from Base Recipe)

All recipes (CollabRecipe and standard recipes) inherit these from the base `Recipe` class:

| Capability | Method | Description |
|-----------|--------|-------------|
| Execute locally | `recipe.execute(env)` | Run with `SimEnv` or `PocEnv` |
| Export for production | `recipe.export(path)` | Generate job folder for FLARE deployment |
| Server filters | `recipe.add_server_output_filter(...)` | Data transformation on server |
| Client filters | `recipe.add_client_input_filter(...)` | Data transformation on client |
| Config | `recipe.add_server_config(...)` | Top-level config key-value pairs |
| Decomposers | `recipe.add_decomposers(...)` | Custom serialization |
| Environment processing | `recipe.process_env(env)` | Environment-specific configuration |

### 6.6 Progression Paths

There are **two progression paths** from Collab API research:

```
CollabRecipe (research / experiment)
    |
    |  -- develop new FL algorithm with @collab.main + @collab.publish
    |  -- fast iteration, pure Python function calls, almost free of FLARE concepts
    |  -- SimEnv (Collab backend): same process, no simulator overhead
    |
    +---> Path A: Novel algorithm -> new recipe
    |         |
    |         |  1. Switch client to Client API (train_script)
    |         |  2. Wire Collab server + Client API client into a new recipe via Job API
    |         |     (open: need a packaging tool to make this step easy)
    |         v
    |     SplitLearningRecipe(...) / MyCustomRecipe(...)  (NEW RECIPE)
    |         |
    |         |  -- end users consume it like FedAvgRecipe (no Collab API knowledge needed)
    |         |  -- supports multi-GPU, multi-node, production deployment
    |         |  -- deploy via export() or K8s
    |
    +---> Path B: Algorithm matches an existing standard recipe
              |
              v
          FedAvgRecipe / CyclicRecipe / etc.  (EXISTING RECIPE)
              |
              |  -- use built-in workflow (originally built with ModelController)
              |  -- no custom server code needed
              |  -- deploy via export() or K8s
```

**Key insights:**

- **Path A is the primary goal** of the Collab API: enable researchers to develop novel FL algorithms and turn them into reusable, production-ready recipes. Using Collab API on the server and Client API on the client requires wiring them together with a recipe -- this **is** developing a new recipe. The packaging step currently requires using the Job API directly; a higher-level packaging tool is an **open design requirement** (see Section 4.4).
- **Path B** is for when the algorithm turns out to match an existing built-in workflow.

---

## 7. Development Requirements

The following table captures the development tasks required to deliver the Collab API as described in this specification.

| # | Requirement | Description | Dependencies | Priority |
|---|-------------|-------------|--------------|----------|
| 1 | **SimEnv Collab Backend (Simulation)** | Extend `SimEnv` with a Collab simulation backend that dispatches `@collab.main` and `@collab.publish` calls as pure function calls within the same process (no old simulator overhead). Handles `collab.clients` proxy, `collab.site_name` injection, and result aggregation. Must support all Collab API usage patterns: single file, multiple files, with classes, without classes (standalone functions). From the user's perspective there is only one `SimEnv` -- the Collab backend is an internal implementation detail selected based on the recipe. | -- | P0 |
| 2 | **Collab FLARE Backend (POC / Production)** | Implement the Collab FLARE Backend that enables `PocEnv`/`ProdEnv` to support Collab API. Dispatches `@collab.main` and `@collab.publish` calls over CellNet (multi-process / multi-node). Must support all Collab API usage patterns (single file, multiple files, with/without classes). Must integrate with existing FLARE communication infrastructure. There is no separate `CollabPocEnv`/`CollabProdEnv` -- the Collab FLARE Backend is what makes `PocEnv`/`ProdEnv` Collab-aware. | 8 | P0 |
| 3 | **CollabClientAPI bridge (Simulation)** | In `SimEnv` (Collab backend), support Collab API on the server side (`@collab.main`) with Client API on the client side (`flare.receive()`/`flare.send()`), connected via the `CollabClientAPI` bridge. Must support both in-process (single GPU) and subprocess (`launch_external_process=True` for DDP) modes. | 1 | P0 |
| 4 | **CollabClientAPI bridge (POC / Production)** | In `PocEnv`/`ProdEnv`, support Collab API on the server side with Client API on the client side, connected via the `CollabClientAPI` bridge over CellNet. Must support subprocess mode for multi-GPU / multi-node DDP. | 2, 3 | P0 |
| 5 | **Recipe packaging utility** | Create a utility, base class, or tooling that makes it easy to package Collab API server logic + Client API client into a new named recipe (e.g. `SplitLearningRecipe`) via the Job API. Today this requires building every piece from scratch. The tool should let researchers turn their `@collab.main` server + `train_script` client into a reusable recipe without re-implementing all the wiring manually. Design is an open question (see Section 4.4). | 3, 4 | P1 |
| 6 | **CollabRecipe** | Create `CollabRecipe` that works in both simulation (`SimEnv`) and POC/Production environments. Should support all usage patterns: Collab API on both sides, and Collab API server + Client API client (via `train_script`). This is the primary recipe researchers use during the research and hybrid phases. Initial simulation support requires #1 only; full POC/Production support requires #2, #3, #4. | 1 (initial), 2, 3, 4 (full) | P0 |
| 7 | **ExecEnv enhancement** | Enhance `ExecEnv` to support Collab API while remaining backward compatible. Existing recipes (no Collab API) must work exactly as before. When Collab API is involved, the environment must instantiate the Collab Backend. Requires an explicit mechanism for the recipe/job to indicate API type (not code scanning). Design is an open question (see Section 5.1). | 1, 2 | P0 |
| 8 | **Decorator and runtime infrastructure** | Implement `@collab.main`, `@collab.publish` decorators and the runtime infrastructure: `collab.clients` (ProxyList), `collab.site_name`, `contextvars`-based thread isolation, error handling for client failures. | -- | P0 |
| 9 | **Multi-GPU / DDP support** | Support `launch_external_process=True` with `torchrun` (or equivalent) for multi-GPU DDP training. The subprocess launcher spawns the training script; rank 0 communicates with the server via `CollabClientAPI`; other ranks sync via `dist.broadcast`. | 3, 4 | P1 |

### Suggested Implementation Order

```
Phase 1 (Foundation):   #8 (decorators/runtime) + #1 (SimEnv Collab Backend)
Phase 2 (Research):     #6 (CollabRecipe, initial sim-only) + #7 (ExecEnv enhancement)
Phase 3 (Bridge):       #3 (CollabClientAPI in Sim) + #9 (multi-GPU/DDP) + #6 (CollabRecipe + Client API bridge)
Phase 4 (Production):   #2 (Collab FLARE Backend for Poc/Prod) + #4 (CollabClientAPI in Poc/Prod) + #6 (CollabRecipe full)
Phase 5 (Packaging):    #5 (Recipe packaging utility)
```

### Open Design Questions

| # | Question | See Section |
|---|----------|-------------|
| OQ-1 | How does the recipe indicate its API type to the environment? (flag, base class check, metadata, or other mechanism -- no code scanning or auto-detection) | 5.1 |
| OQ-2 | What does the recipe packaging utility look like? (base class, builder, code generator, or other) | 4.4 |
| OQ-3 | How are Collab server + Client API client wired in Phase 2 (hybrid)? (`CollabRecipe`, per-algorithm recipe, or new mechanism) | 4.4 |
