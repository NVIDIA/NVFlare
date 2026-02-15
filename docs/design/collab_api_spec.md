# FLARE Collab API Specification

*Target audience: FL algorithm researchers and data scientists developing federated learning algorithms with NVIDIA FLARE.*

*For internal design and implementation details, see [collab_api_design_spec.md](collab_api_design_spec.md).*

## 1. Overview

The **Collab API** (code name Fox -- FLARE Object Exchange) is NVIDIA FLARE's collaborative API that simplifies federated learning development. It allows researchers to write natural Python code while the framework handles distribution, communication, and orchestration.

### Design Goals

1. **Simplicity:** Write federated learning code as if it were local Python.
2. **Flexibility:** Support in-process simulation, multi-process POC, and distributed production.
3. **Transparency:** Hide infrastructure complexity from data scientists.
4. **Compatibility:** Minimal code changes when moving from simulation to production.

### Two API Patterns

The Collab API supports two client-side patterns:

| Pattern | Client Code | Use Case |
|---------|------------|----------|
| **Collab API** | `@collab.publish` decorator; server pushes data to client methods | Algorithm development, rapid prototyping, single-GPU in-process experiments |
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
| `SimEnv` | Simulation environment -- fast iteration for Collab recipes (pure function calls, no simulator overhead) |
| `PocEnv` | Multi-process local environment for realistic testing |

---

## 2. Usage Patterns

### 2.1 In-Process -- Single File (Single GPU)

The simplest pattern: server and client logic in a single Python file. Everything runs in-process via `SimEnv` with fast, pure function call execution.

**Single file with classes:**

```python
# fedavg_train.py -- server + client + execution in one file
from nvflare.collab import collab
from nvflare.recipe.sim_env import SimEnv
from nvflare.collab.sys.recipe import CollabRecipe

# --- Client ---
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
import torch.nn as nn
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

`CollabRecipe` extends the base `Recipe` class, so it supports all Recipe API features:

```python
from nvflare.collab.sys.recipe import CollabRecipe
from nvflare.recipe.sim_env import SimEnv
from nvflare.recipe.poc_env import PocEnv

# Simulation (pure function calls, fast iteration)
recipe = CollabRecipe(job_name="fedavg", server=server, client=client, min_clients=5)
run = recipe.execute(SimEnv(num_clients=5))

# POC (multi-process, CellNet communication, more realistic)
run = recipe.execute(PocEnv(num_clients=5))

# Export job for production deployment
recipe.export("/path/to/job_output")
```

---

## 3. Collab Server + Client API (Hybrid Pattern)

As your algorithm matures, you may want to use the **Client API** (`flare.receive()` / `flare.send()`) on the client side while keeping the Collab API on the server. This enables multi-GPU support, multi-node deployment, and independent server/client development.

The framework provides a transparent bridge that connects the two. You don't need to understand how it works -- just configure `train_script` in `CollabRecipe`.

### 3.1 Single GPU

**Server (server.py):**

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

**Client (client.py) -- standard Client API:**

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
        # ... standard PyTorch training loop (computes loss) ...
        flare.send(FLModel(params=model.state_dict(), metrics={"loss": loss.item()}))
```

**Job (job.py):**

```python
from server import FedAvg
from nvflare.recipe.sim_env import SimEnv
from nvflare.collab.sys.recipe import CollabRecipe

recipe = CollabRecipe(
    job_name="fedavg_client_api",
    server=FedAvg(num_rounds=3),
    train_script="client.py",   # Client API pattern
    min_clients=2,
)
run = recipe.execute(SimEnv(num_clients=2))
```

**Key points:**
- `train_script="client.py"` tells `CollabRecipe` to use the Client API pattern.
- The client is **completely unaware** of the Collab API -- portable to any NVFlare deployment.

### 3.2 Multi-GPU (DDP)

Add `launch_external_process=True` with a `torchrun` command. Server code is unchanged.

**Client (client.py) -- DDP + Client API:**

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
        if rank == 0 and input_model.params:
            net.load_state_dict(input_model.params)
        for param in net.parameters():
            dist.broadcast(param.data, src=0)
        # ... training loop with ddp_model and DistributedSampler ...
        if rank == 0:
            flare.send(FLModel(params=net.state_dict(), metrics={"loss": loss.item()}))
    dist.destroy_process_group()
```

**Job (job.py):**

```python
recipe = CollabRecipe(
    job_name="fedavg_ddp",
    server=FedAvg(num_rounds=3),
    train_script="client.py",
    min_clients=2,
    launch_external_process=True,
    command="torchrun --nproc_per_node=4",
    subprocess_timeout=600.0,
)
run = recipe.execute(SimEnv(num_clients=2))
```

---

## 4. Progression Paths

### 4.1 When to Progress

| Scenario | Recommended Path |
|----------|-----------------|
| Novel algorithm, single GPU | Stay with Collab API on both sides -- fastest iteration |
| Novel algorithm, multi-GPU or production | Collab server + Client API client -> new recipe (Section 4.3) |
| Algorithm matches FedAvg/Cyclic/etc. | Switch to standard recipe (Section 4.2) |

### 4.2 Transition to Standard Recipe

When your algorithm matches a built-in recipe, migrate in three steps:

1. **Extract client** into a standalone `train.py` using Client API (`flare.receive()` / `flare.send()`).
2. **Replace CollabRecipe** with the matching recipe (e.g. `FedAvgRecipe`).
3. **Deploy** via `recipe.export()` or `recipe.execute(PocEnv(...))`.

### 4.3 Creating a New Recipe

This is the **primary goal** of the Collab API.

```
Phase 1 (Research):   CollabRecipe + @collab.main + @collab.publish  (SimEnv, pure function calls)
Phase 2 (Hybrid):     Collab server + Client API client              (multi-GPU, decoupled development)
Phase 3 (New Recipe): Package into SplitLearningRecipe(...)          (end users consume like FedAvgRecipe)
```

### 4.4 Summary

| Component | Collab API (both sides) | Collab Server + Client API | Standard Recipe |
|-----------|------------------------|----------------------------|-----------------|
| **Server** | `@collab.main` (custom) | `@collab.main` (custom) | Built-in workflow |
| **Client** | `@collab.publish` | `flare.receive()` / `flare.send()` | `flare.receive()` / `flare.send()` |
| **Multi-GPU** | No | Yes | Yes |
| **When** | Prototyping | Novel algorithm -> production | Algorithm matches a built-in |

---

## 5. Execution Environments

The user writes the same environment (`SimEnv`, `PocEnv`) regardless of recipe type. The environment selects the appropriate backend automatically.

| Environment | Standard Recipe | Collab Recipe |
|------------|-----------------|---------------|
| **`SimEnv`** | Full FLARE simulator | Collab backend: pure function calls, much faster |
| **`PocEnv`** | FLARE multi-process (CellNet) | Same + Collab FLARE Backend |
| **`ProdEnv`** | FLARE deployment | Same + Collab FLARE Backend |

---

## 6. CollabRecipe and Standard Recipes

### 6.1 When to Use Which

| Use Case | Recipe |
|----------|--------|
| Developing a new FL algorithm | `CollabRecipe` |
| Standard FedAvg | `FedAvgRecipe` |
| Cyclic training | `CyclicRecipe` |
| Swarm learning | `SimpleSwarmLearningRecipe` |
| Custom aggregation or communication | `CollabRecipe` |
| Novel algorithm -> production | `CollabRecipe` -> new recipe via Job API |

### 6.2 Key Difference

**CollabRecipe** -- you write the server algorithm:

```python
class MyAlgorithm:
    def __init__(self, num_rounds=5):
        self.num_rounds = num_rounds

    @collab.main
    def run(self):
        weights = initialize_model()
        for r in range(self.num_rounds):
            results = collab.clients.train(weights)
            weights = my_custom_aggregation(results)
        return weights

recipe = CollabRecipe(job_name="custom", server=MyAlgorithm(num_rounds=5), client=Trainer(), min_clients=5)
```

**FedAvgRecipe** -- the workflow is built-in:

```python
recipe = FedAvgRecipe(name="fedavg", min_clients=5, num_rounds=10, train_script="train.py", model=SimpleModel())
# No server code needed
```

### 6.3 Shared Capabilities (from Base Recipe)

| Capability | Method |
|-----------|--------|
| Execute locally | `recipe.execute(env)` |
| Export for production | `recipe.export(path)` |
| Server/Client filters | `recipe.add_server_output_filter(...)` / `recipe.add_client_input_filter(...)` |
| Config | `recipe.add_server_config(...)` |
| Decomposers | `recipe.add_decomposers(...)` |

### 6.4 Progression Paths

```
CollabRecipe (research)
    |
    +---> Path A: Novel algorithm -> new recipe (Section 4.3)
    |         Package into SplitLearningRecipe(...) via Job API
    |
    +---> Path B: Matches a built-in -> FedAvgRecipe / CyclicRecipe (Section 4.2)
```
