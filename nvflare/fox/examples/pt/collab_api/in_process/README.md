# From Simulation to Federated Learning with Fox

This guide shows a **progressive comparison** from local simulation to NVFlare Fox federated learning:

1. `simulate_fedavg_train.py` - Sequential simulation
2. `simulate_parallel_fedavg_train.py` - Parallel simulation (mimics Fox pattern)
3. `collab_fedavg_train.py` - Full Fox federated learning

The code is intentionally structured to **minimize differences** - making it easy to see exactly what changes at each step.

---

## Part 1: Sequential → Parallel Simulation

This section compares `simulate_fedavg_train.py` (sequential) with `simulate_parallel_fedavg_train.py` (parallel).

### Imports

| simulate_fedavg_train.py | simulate_parallel_fedavg_train.py |
|--------------------------|-----------------------------------|

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
```

```python
from concurrent.futures import ThreadPoolExecutor, as_completed  # +

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
```

**Change:** Add threading imports (hidden in `ClientGroup` abstraction).

---

### Framework Abstraction (New in Parallel Version)

The parallel version adds a `ClientGroup` class that hides parallelism:

```python
class ClientGroup:
    """A group of clients that can execute methods in parallel."""

    def __init__(self, client_ids, trainer, max_workers=None):
        self.client_ids = client_ids
        self.trainer = trainer
        self.max_workers = max_workers or len(client_ids)

    def __getattr__(self, method_name):
        """Allow calling any method on the trainer in parallel."""
        def parallel_call(*args, **kwargs):
            results = []
            with ThreadPoolExecutor(...) as executor:
                # Submit all client tasks in parallel
                future_to_client = {
                    executor.submit(method, client_id, *args, **kwargs): client_id
                    for client_id in self.client_ids
                }
                # Collect results as they complete
                for future in as_completed(future_to_client):
                    client_id = future_to_client[future]
                    result = future.result()
                    results.append((client_id, result))
            return results
        return parallel_call
```

**Purpose:** This abstraction allows `clients.train(weights)` to work like `fox.clients.train(weights)`.

---

### Training Function

| simulate_fedavg_train.py | simulate_parallel_fedavg_train.py |
|--------------------------|-----------------------------------|

```python
def train(weights=None):
    # Setup data
    inputs = torch.randn(100, 10)
    labels = torch.randn(100, 1)
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=10)

    model = SimpleModel()
    if weights is not None:
        model.load_state_dict(weights)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(5):
        for batch in dataloader:
            # ... training loop ...



    return model.state_dict(), loss.item()
```

```python
class Trainer:                                    # + Wrap in class
    def train(self, client_id, weights=None):     # + Add client_id
        # Setup data
        inputs = torch.randn(100, 10)
        labels = torch.randn(100, 1)
        dataset = TensorDataset(inputs, labels)
        dataloader = DataLoader(dataset, batch_size=10)

        model = SimpleModel()
        if weights is not None:
            model.load_state_dict(weights)

        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        for epoch in range(5):
            for batch in dataloader:
                # ... training loop ...

        print(f"  [{client_id}] Loss: {loss.item():.4f}")  # + Logging

        return model.state_dict(), loss.item()
```

**Changes:**
1. Wrap in `class Trainer`
2. Add `client_id` parameter for identification
3. Add logging with client_id

---

### FedAvg Workflow

| simulate_fedavg_train.py | simulate_parallel_fedavg_train.py |
|--------------------------|-----------------------------------|

```python
def fed_avg(clients, num_rounds=5):
    print(f"Starting FedAvg for {num_rounds} rounds")
    global_weights = None

    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1} ===")

        # Each client trains (simulated sequentially)
        client_results = []
        for client_id in clients:
            weights, loss = train(global_weights)
            client_results.append((client_id, (weights, loss)))
            print(f"  [{client_id}] Loss: {loss:.4f}")

        global_weights, global_loss = weighted_avg(client_results)
        print(f"  Global average loss: {global_loss:.4f}")

    print(f"\nFedAvg completed after {num_rounds} rounds")
    return global_weights
```

```python
def fed_avg(clients, num_rounds=5):
    print(f"Starting FedAvg for {num_rounds} rounds")
    global_weights = None

    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1} ===")

        # Each client trains (in parallel via clients group)
        client_results = clients.train(global_weights)
        # ^^^ Replaces the entire for-loop! ^^^




        global_weights, global_loss = weighted_avg(client_results)
        print(f"  Global average loss: {global_loss:.4f}")

    print(f"\nFedAvg completed after {num_rounds} rounds")
    return global_weights
```

**Key Change:** Replace the sequential for-loop with `clients.train(global_weights)`.

---

### Execution

| simulate_fedavg_train.py | simulate_parallel_fedavg_train.py |
|--------------------------|-----------------------------------|

```python
if __name__ == "__main__":
    clients = ["site-1", "site-2", "site-3", "site-4", "site-5"]
    result = fed_avg(clients, num_rounds=5)


    print("Job Status:", "Completed")
```

```python
if __name__ == "__main__":
    client_ids = ["site-1", "site-2", "site-3", "site-4", "site-5"]
    trainer = Trainer()                           # + Create trainer
    clients = ClientGroup(client_ids, trainer)    # + Wrap in ClientGroup

    result = fed_avg(clients, num_rounds=5)

    print("Job Status:", "Completed")
```

**Changes:**
1. Create `Trainer` instance
2. Wrap client_ids in `ClientGroup` for parallel execution

---

### Part 1 Summary: Sequential → Parallel

| Component | Sequential | Parallel Sim |
|-----------|------------|--------------|
| **Imports** | PyTorch only | + threading (hidden in ClientGroup) |
| **Parallelism** | None (for-loop) | `ClientGroup` abstraction |
| **Training** | Function | Class + `client_id` parameter |
| **Workflow** | For-loop over clients | `clients.train()` call |
| **Execution** | Direct function call | ClientGroup wraps trainer |

**Key Insight:** Replace the for-loop with `clients.train()` - parallelism is abstracted away!

---

## Part 2: Parallel Simulation → Fox API

This section compares `simulate_parallel_fedavg_train.py` with `collab_fedavg_train.py`.

### Imports

| simulate_parallel_fedavg_train.py | collab_fedavg_train.py |
|-----------------------------------|------------------------|

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
```

```python
                                                  # - Remove threading imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from nvflare.fox import fox                       # + Add Fox imports
from nvflare.fox.sim import SimEnv                # +
from nvflare.fox.sys.recipe import FoxRecipe      # +
```

**Changes:**
1. Remove threading imports (Fox handles parallelism)
2. Add 3 Fox imports

---

### Framework Abstraction

| simulate_parallel_fedavg_train.py | collab_fedavg_train.py |
|-----------------------------------|------------------------|

```python
class ClientGroup:
    """A group of clients that can execute
    methods in parallel."""

    def __init__(self, client_ids, trainer, ...):
        self.client_ids = client_ids
        self.trainer = trainer
        ...

    def __getattr__(self, method_name):
        def parallel_call(*args, **kwargs):
            # ThreadPoolExecutor logic...
            ...
        return parallel_call
```

```python
# No ClientGroup needed!
# Fox provides fox.clients automatically.




# fox.clients is a built-in ProxyList that
# dispatches calls to all clients in parallel.


```

**Change:** Remove `ClientGroup` - Fox provides `fox.clients` built-in.

---

### Training Class

| simulate_parallel_fedavg_train.py | collab_fedavg_train.py |
|-----------------------------------|------------------------|

```python
class Trainer:
    def train(self, client_id, weights=None):
        # Setup data
        inputs = torch.randn(100, 10)
        ...

        for epoch in range(5):
            for batch in dataloader:
                # ... training loop ...

        print(f"  [{client_id}] Loss: ...")

        return model.state_dict(), loss.item()
```

```python
class Trainer:
    @fox.collab                                   # + Add decorator
    def train(self, weights=None):                # - Remove client_id
        # Setup data
        inputs = torch.randn(100, 10)
        ...

        for epoch in range(5):
            for batch in dataloader:
                # ... training loop ...

        print(f"  [{fox.site_name}] Loss: ...")   # Use fox.site_name

        return model.state_dict(), loss.item()
```

**Changes:**
1. Add `@fox.collab` decorator
2. Remove `client_id` parameter (Fox injects context)
3. Use `fox.site_name` instead of `client_id`

---

### FedAvg Workflow

| simulate_parallel_fedavg_train.py | collab_fedavg_train.py |
|-----------------------------------|------------------------|

```python
def fed_avg(clients, num_rounds=5):
    print(f"Starting FedAvg for {num_rounds} rounds")
    global_weights = None

    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1} ===")

        client_results = clients.train(global_weights)

        global_weights, global_loss = weighted_avg(client_results)
        print(f"  Global average loss: {global_loss:.4f}")

    print(f"\nFedAvg completed after {num_rounds} rounds")
    return global_weights
```

```python
class FedAvg:                                     # + Wrap in class
    def __init__(self, num_rounds=5):             # + Constructor
        self.num_rounds = num_rounds

    @fox.algo                                     # + Add decorator
    def fed_avg(self):                            # - Remove clients param
        print(f"Starting FedAvg for {self.num_rounds} rounds")
        global_weights = None

        for round_num in range(self.num_rounds):
            print(f"\n=== Round {round_num + 1} ===")

            client_results = fox.clients.train(global_weights)  # fox.clients

            global_weights, global_loss = weighted_avg(client_results)
            print(f"  Global average loss: {global_loss:.4f}")

        print(f"\nFedAvg completed after {self.num_rounds} rounds")
        return global_weights
```

**Changes:**
1. Wrap in `class FedAvg` with constructor
2. Add `@fox.algo` decorator
3. Remove `clients` parameter
4. Use `fox.clients` instead of passed-in `clients`

---

### Execution

| simulate_parallel_fedavg_train.py | collab_fedavg_train.py |
|-----------------------------------|------------------------|

```python
if __name__ == "__main__":
    client_ids = ["site-1", ..., "site-5"]
    trainer = Trainer()
    clients = ClientGroup(client_ids, trainer)

    result = fed_avg(clients, num_rounds=5)

    print("Job Status:", "Completed")
    print("Results:", "In memory (state_dict)")
```

```python
if __name__ == "__main__":
    server = FedAvg(num_rounds=5)                 # + Server object
    client = Trainer()                            # Client object

    recipe = FoxRecipe(                           # + Recipe configuration
        job_name="fedavg_train",
        server=server,
        client=client,
        min_clients=5,
    )
    env = SimEnv(num_clients=5)                   # + Environment

    run = recipe.execute(env)                     # + Execute via recipe

    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())
```

**Changes:**
1. Create server and client objects separately
2. Configure via `FoxRecipe`
3. Create `SimEnv` for execution environment
4. Execute via `recipe.execute(env)`

---

### Part 2 Summary: Parallel Simulation → Fox API

| Component | Parallel Sim | Fox API |
|-----------|--------------|---------|
| **Imports** | + threading | + Fox, - threading |
| **Parallelism** | `ClientGroup` (custom) | `fox.clients` (built-in) |
| **Training** | Class + `client_id` param | Class + `@fox.collab` decorator |
| **Client ID** | Parameter | `fox.site_name` (injected) |
| **Workflow** | Function | Class + `@fox.algo` decorator |
| **Execution** | Direct call | Recipe + Environment pattern |

**Key Insight:** Replace `ClientGroup` with `fox.clients`, add decorators - the training logic stays the same!

---

## Running the Examples

```bash
# Sequential simulation
python simulate_fedavg_train.py

# Parallel simulation (mimics Fox pattern)
python simulate_parallel_fedavg_train.py

# Fox federated simulation  
python collab_fedavg_train.py
```

All three produce similar output, but:
- `simulate_fedavg_train.py` runs clients **sequentially**
- `simulate_parallel_fedavg_train.py` runs clients **in parallel threads**
- `collab_fedavg_train.py` runs clients **in parallel** and can be deployed to **real distributed environments**
