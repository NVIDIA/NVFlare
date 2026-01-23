# From Simulation to Federated Learning with Collab

This guide shows a **progressive comparison** from local simulation to NVFlare Collab federated learning:

1. `simulate_fedavg_train.py` - Sequential simulation
2. `simulate_parallel_fedavg_train.py` - Parallel simulation (mimics Collab pattern)
3. `collab_fedavg_train.py` - Collab with classes
4. `collab_fedavg_no_class.py` - Collab with standalone functions (no classes!)
5. `collab_fedavg_no_class_job.py` - Collab with split server/client modules

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
    """A group of clients that can execute functions in parallel."""

    def __init__(self, client_ids, max_workers=None):
        self.client_ids = client_ids
        self.max_workers = max_workers or len(client_ids)
        self._functions = {}

    def register(self, func):
        """Register a function for parallel calls."""
        self._functions[func.__name__] = func
        return func

    def __getattr__(self, func_name):
        """Allow calling registered function in parallel."""
        func = self._functions.get(func_name)
        def parallel_call(*args, **kwargs):
            # ThreadPoolExecutor logic...
            ...
        return parallel_call
```

**Purpose:** This abstraction allows `clients.train(weights)` to work like `collab.clients.train(weights)`.

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
@clients.register                                 # + Register with ClientGroup
def train(client_id, weights=None):               # + Add client_id
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
1. Add `@clients.register` decorator
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
NUM_ROUNDS = 5                                    # Module-level config

def fed_avg():                                    # - Remove clients param
    print(f"Starting FedAvg for {NUM_ROUNDS} rounds")
    global_weights = None

    for round_num in range(NUM_ROUNDS):
        print(f"\n=== Round {round_num + 1} ===")

        # Each client trains (in parallel via clients group)
        client_results = clients.train(global_weights)
        # ^^^ Replaces the entire for-loop! ^^^

        global_weights, global_loss = weighted_avg(client_results)
        print(f"  Global average loss: {global_loss:.4f}")

    print(f"\nFedAvg completed after {NUM_ROUNDS} rounds")
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
# ClientGroup and clients defined at module level

if __name__ == "__main__":
    result = fed_avg()                            # Just call it!

    print("Job Status:", "Completed")
```

**Changes:**
1. `ClientGroup` defined at module level
2. `@clients.register` decorates the train function
3. Direct `fed_avg()` call - no parameters needed

---

### Part 1 Summary: Sequential → Parallel

| Component | Sequential | Parallel Sim |
|-----------|------------|--------------|
| **Imports** | PyTorch only | + threading (hidden in ClientGroup) |
| **Parallelism** | None (for-loop) | `ClientGroup` + `@clients.register` |
| **Training** | Function | Function + `client_id` parameter |
| **Workflow** | For-loop over clients | `clients.train()` call |
| **Execution** | `fed_avg(clients, ...)` | `fed_avg()` |

**Key Insight:** Replace the for-loop with `clients.train()` - parallelism is abstracted away!

---

## Part 2: Parallel Simulation → Collab API (with Classes)

This section compares `simulate_parallel_fedavg_train.py` with `collab_fedavg_train.py`.
This shows the class-based Collab API approach.

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

from nvflare.collab import collab                       # + Add Collab imports
from nvflare.publish.sim import SimEnv                # +
from nvflare.publish.sys.recipe import FoxRecipe      # +
```

**Changes:**
1. Remove threading imports (Collab handles parallelism)
2. Add 3 Collab imports

---

### Framework Abstraction

| simulate_parallel_fedavg_train.py | collab_fedavg_train.py |
|-----------------------------------|------------------------|

```python
class ClientGroup:
    """A group of clients that can execute
    functions in parallel."""

    def __init__(self, client_ids, ...):
        ...

    def register(self, func):
        ...

    def __getattr__(self, func_name):
        def parallel_call(*args, **kwargs):
            # ThreadPoolExecutor logic...
            ...
        return parallel_call
```

```python
# No ClientGroup needed!
# Collab provides collab.clients automatically.




# collab.clients is a built-in ProxyList that
# dispatches calls to all clients in parallel.

```

**Change:** Remove `ClientGroup` - Collab provides `collab.clients` built-in.

---

### Training

| simulate_parallel_fedavg_train.py | collab_fedavg_train.py |
|-----------------------------------|------------------------|

```python
@clients.register                                 # Register with ClientGroup
def train(client_id, weights=None):               # Standalone function
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
class Trainer:                                    # + Wrap in class
    @collab.publish                                   # + Collab decorator
    def train(self, weights=None):                # - Remove client_id
        # Setup data
        inputs = torch.randn(100, 10)
        ...

        for epoch in range(5):
            for batch in dataloader:
                # ... training loop ...

        print(f"  [{collab.site_name}] Loss: ...")   # + collab.site_name

        return model.state_dict(), loss.item()
```

**Changes:**
1. Wrap in `class Trainer`
2. Replace `@clients.register` with `@collab.publish`
3. Remove `client_id` parameter
4. Use `collab.site_name` instead of `client_id`

---

### FedAvg Workflow

| simulate_parallel_fedavg_train.py | collab_fedavg_train.py |
|-----------------------------------|------------------------|

```python
NUM_ROUNDS = 5

def fed_avg():                                    # Standalone function
    print(f"Starting FedAvg for {NUM_ROUNDS} rounds")
    global_weights = None

    for round_num in range(NUM_ROUNDS):
        print(f"\n=== Round {round_num + 1} ===")

        client_results = clients.train(global_weights)

        global_weights, global_loss = weighted_avg(client_results)
        print(f"  Global average loss: {global_loss:.4f}")

    print(f"\nFedAvg completed after {NUM_ROUNDS} rounds")
    return global_weights
```

```python
class FedAvg:                                     # + Wrap in class
    def __init__(self, num_rounds=5):             # + Constructor
        self.num_rounds = num_rounds

    @collab.main                                     # + Add decorator
    def fed_avg(self):
        print(f"Starting FedAvg for {self.num_rounds} rounds")
        global_weights = None

        for round_num in range(self.num_rounds):
            print(f"\n=== Round {round_num + 1} ===")

            client_results = collab.clients.train(global_weights)  # collab.clients

            global_weights, global_loss = weighted_avg(client_results)
            print(f"  Global average loss: {global_loss:.4f}")

        print(f"\nFedAvg completed after {self.num_rounds} rounds")
        return global_weights
```

**Changes:**
1. Wrap in `class FedAvg` with constructor
2. Add `@collab.main` decorator  
3. Replace `clients.train()` with `collab.clients.train()`

---

### Execution

| simulate_parallel_fedavg_train.py | collab_fedavg_train.py |
|-----------------------------------|------------------------|

```python
# ClientGroup defined at module level

if __name__ == "__main__":
    result = fed_avg()


    print("Job Status:", "Completed")
    print("Results:", "In memory (state_dict)")
```

```python
if __name__ == "__main__":
    server = FedAvg(num_rounds=5)                 # + Server class
    client = Trainer()                            # + Client class

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
1. Create `FedAvg` and `Trainer` class instances
2. Configure via `FoxRecipe`
3. Create `SimEnv` for execution environment
4. Execute via `recipe.execute(env)`

---

### Part 2 Summary: Parallel Simulation → Collab API (Classes)

| Component | Parallel Sim | Collab API (Classes) |
|-----------|--------------|-------------------|
| **Imports** | + threading | + Collab, - threading |
| **Parallelism** | `ClientGroup` (custom) | `collab.clients` (built-in) |
| **Training** | `@clients.register` function | `class Trainer` + `@collab.publish` |
| **Client ID** | `client_id` parameter | `collab.site_name` (injected) |
| **Workflow** | `fed_avg()` function | `class FedAvg` + `@collab.main` |
| **Execution** | Direct `fed_avg()` call | Recipe + Environment pattern |

**Key Insight:** Collab's class-based API wraps training and workflow in classes with decorators.

---

## Part 3: Parallel Simulation → Collab API (No Classes!)

This section compares `simulate_parallel_fedavg_train.py` with `collab_fedavg_no_class.py`.
Both use **standalone functions** - making the comparison clean and direct!

### Imports

| simulate_parallel_fedavg_train.py | collab_fedavg_no_class.py |
|-----------------------------------|---------------------------|

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

from nvflare.collab import collab                       # + Add Collab imports
from nvflare.publish.sim import SimEnv                # +
from nvflare.publish.sys.recipe import FoxRecipe      # +
```

**Changes:**
1. Remove threading imports (Collab handles parallelism)
2. Add 3 Collab imports

---

### Framework Abstraction

| simulate_parallel_fedavg_train.py | collab_fedavg_no_class.py |
|-----------------------------------|---------------------------|

```python
class ClientGroup:
    """A group of clients that can execute
    functions in parallel."""

    def __init__(self, client_ids, ...):
        self.client_ids = client_ids
        ...

    def register(self, func):
        """Register function for parallel calls."""
        ...

    def __getattr__(self, func_name):
        def parallel_call(*args, **kwargs):
            # ThreadPoolExecutor logic...
            ...
        return parallel_call
```

```python
# No ClientGroup needed!
# No custom abstraction at all!

# Collab provides collab.clients automatically.
# FoxRecipe auto-detects the caller's module.




```

**Change:** Remove `ClientGroup` entirely - Collab provides `collab.clients` built-in.

---

### Training Function

| simulate_parallel_fedavg_train.py | collab_fedavg_no_class.py |
|-----------------------------------|---------------------------|

```python
@clients.register                                 # Register with ClientGroup
def train(client_id, weights=None):               # Need client_id param
    # Setup data
    inputs = torch.randn(100, 10)
    ...

    for epoch in range(5):
        for batch in dataloader:
            # ... training loop ...

    print(f"  [{client_id}] Loss: ...")           # Use client_id

    return model.state_dict(), loss.item()
```

```python
@collab.publish                                       # + Collab decorator
def train(weights=None):                          # - No client_id param
    # Setup data
    inputs = torch.randn(100, 10)
    ...

    for epoch in range(5):
        for batch in dataloader:
            # ... training loop ...

    print(f"  [{collab.site_name}] Loss: ...")       # + collab.site_name

    return model.state_dict(), loss.item()
```

**Changes:**
1. Replace `@clients.register` with `@collab.publish`
2. Remove `client_id` parameter
3. Use `collab.site_name` instead of `client_id`

---

### FedAvg Workflow

| simulate_parallel_fedavg_train.py | collab_fedavg_no_class.py |
|-----------------------------------|---------------------------|

```python
NUM_ROUNDS = 5

def fed_avg():                                    # Standalone function
    print(f"Starting FedAvg for {NUM_ROUNDS} rounds")
    global_weights = None

    for round_num in range(NUM_ROUNDS):
        print(f"\n=== Round {round_num + 1} ===")

        client_results = clients.train(global_weights)

        global_weights, global_loss = weighted_avg(client_results)
        print(f"  Global average loss: {global_loss:.4f}")

    print(f"\nFedAvg completed after {NUM_ROUNDS} rounds")
    return global_weights
```

```python
NUM_ROUNDS = 5

@collab.main                                         # + Add decorator
def fed_avg():                                    # Same standalone function!
    print(f"Starting FedAvg for {NUM_ROUNDS} rounds")
    global_weights = None

    for round_num in range(NUM_ROUNDS):
        print(f"\n=== Round {round_num + 1} ===")

        client_results = collab.clients.train(global_weights)  # collab.clients

        global_weights, global_loss = weighted_avg(client_results)
        print(f"  Global average loss: {global_loss:.4f}")

    print(f"\nFedAvg completed after {NUM_ROUNDS} rounds")
    return global_weights
```

**Changes:**
1. Add `@collab.main` decorator
2. Replace `clients.train()` with `collab.clients.train()`

That's it! The workflow logic is **identical**.

---

### Execution

| simulate_parallel_fedavg_train.py | collab_fedavg_no_class.py |
|-----------------------------------|---------------------------|

```python
if __name__ == "__main__":
    result = fed_avg()


    print("Job Status:", "Completed")
    print("Results:", "In memory (state_dict)")
```

```python
if __name__ == "__main__":
    recipe = FoxRecipe(job_name="fedavg", min_clients=5)
    env = SimEnv(num_clients=5)
    run = recipe.execute(env)

    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())
```

**Changes:**
1. Replace direct `fed_avg()` call with `FoxRecipe` + `SimEnv`
2. `FoxRecipe` auto-detects the module containing `@collab.main` and `@collab.publish`

---

### Part 3 Summary: Parallel Sim → Collab (No Classes)

| Component | Parallel Sim | Collab (No Classes) |
|-----------|--------------|------------------|
| **Imports** | + threading | + Collab, - threading |
| **Parallelism** | `ClientGroup` (custom) | `collab.clients` (built-in) |
| **Training** | `@clients.register` + `client_id` | `@collab.publish` + `collab.site_name` |
| **Workflow** | `clients.train()` | `collab.clients.train()` |
| **Execution** | Direct `fed_avg()` call | `FoxRecipe().execute()` |

**Key Insight:** The core training and workflow logic are **nearly identical**!
- Replace `@clients.register` → `@collab.publish`
- Replace `client_id` → `collab.site_name`
- Replace `clients.train()` → `collab.clients.train()`
- Add `@collab.main` to the workflow function

The simulation code mirrors Collab so closely that migration is trivial!

---

### Bonus: Split Server and Client Modules

For larger projects, you can split server and client logic into separate files:

```
collab_fedavg_no_class_client.py   # @collab.publish train()
collab_fedavg_no_class_server.py   # @collab.main fed_avg()
collab_fedavg_no_class_job.py      # Recipe ties them together
```

```python
# collab_fedavg_no_class_job.py
from ... import collab_fedavg_no_class_client as client_module
from ... import collab_fedavg_no_class_server as server_module

recipe = FoxRecipe(
    job_name="fedavg_split",
    server=server_module,   # Auto-wrapped with ModuleWrapper
    client=client_module,   # Auto-wrapped with ModuleWrapper
    min_clients=5,
)
```

---

## Running the Examples

```bash
# Sequential simulation
python simulate_fedavg_train.py

# Parallel simulation (mimics Collab pattern)
python simulate_parallel_fedavg_train.py

# Collab with classes
python collab_fedavg_train.py

# Collab with standalone functions (no classes!)
python collab_fedavg_no_class.py

# Collab with split server/client modules
python collab_fedavg_no_class_job.py
```

All produce similar output, but:
- `simulate_fedavg_train.py` runs clients **sequentially**
- `simulate_parallel_fedavg_train.py` runs clients **in parallel threads**
- `collab_fedavg_train.py` runs clients **in parallel** with Collab classes
- `collab_fedavg_no_class.py` runs clients **in parallel** with Collab standalone functions
- `collab_fedavg_no_class_job.py` demonstrates **split modules** for real deployments
