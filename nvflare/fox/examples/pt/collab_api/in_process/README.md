# From Simulation to Federated Learning with Fox

This guide shows a **side-by-side comparison** between a local simulation (`simulate_fedavg_train.py`) and NVFlare Fox federated learning (`collab_fedavg_train.py`).

The code is intentionally structured to **minimize differences** - making it easy to see exactly what changes when converting to federated learning.

---

## Side-by-Side Comparison

### Imports

| simulate_fedavg_train.py | collab_fedavg_train.py |
|--------------------------|------------------------|
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset




```
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from nvflare.fox import fox                    # +
from nvflare.fox.sim import SimEnv             # +
from nvflare.fox.sys.recipe import FoxRecipe   # +
```

**Change:** Add 3 Fox imports.

---

### Model Definition (Identical)

Both files use the exact same model:

```python
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)
```

**Change:** None.

---

### Training Function

| simulate_fedavg_train.py | collab_fedavg_train.py |
|--------------------------|------------------------|

```python
# 1.2 Define Training Function
def train(weights=None):
    # Setup data
    inputs = torch.randn(100, 10)
    labels = torch.randn(100, 1)
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=10)

    # Load global model weights if provided
    model = SimpleModel()
    if weights is not None:
        model.load_state_dict(weights)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(5):
        for batch in dataloader:
            batch_inputs, batch_labels = batch
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()



    # Return updated weights and loss
    return model.state_dict(), loss.item()
```

```python
# 1.2 Define Training Function
class Trainer:                                           # + Wrap in class
    @fox.collab                                          # + Add decorator
    def train(self, weights=None):                       # + Add self
        # Setup data
        inputs = torch.randn(100, 10)
        labels = torch.randn(100, 1)
        dataset = TensorDataset(inputs, labels)
        dataloader = DataLoader(dataset, batch_size=10)

        # Load global model weights if provided
        model = SimpleModel()
        if weights is not None:
            model.load_state_dict(weights)

        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Training loop
        for epoch in range(5):
            for batch in dataloader:
                batch_inputs, batch_labels = batch
                optimizer.zero_grad()
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

        print(f"  [{fox.site_name}] Loss: {loss.item():.4f}")  # + Optional logging

        # Return updated weights and loss
        return model.state_dict(), loss.item()
```

**Changes:**
1. Wrap function in a `class Trainer`
2. Add `@fox.collab` decorator
3. Add `self` parameter
4. (Optional) Use `fox.site_name` for logging

---

### Weighted Average Function (Identical)

Both versions use the exact same function:

```python
def weighted_avg(client_results):
    # client_results: [(client_id, (weights, loss)), ...]
    # Filter out any exceptions
    valid_results = {}
    for client_id, result in client_results:
        if isinstance(result, Exception):
            print(f"  Warning: {client_id} failed")
            continue
        valid_results[client_id] = result

    all_weights = [result[0] for result in valid_results.values()]
    all_losses = [result[1] for result in valid_results.values()]

    # Simple averaging of model parameters
    avg_weights = {}
    for key in all_weights[0].keys():
        avg_weights[key] = torch.stack(
            [w[key].float() for w in all_weights]
        ).mean(dim=0)

    avg_loss = sum(all_losses) / len(all_losses)
    return avg_weights, avg_loss
```

**Change:** None.

---

### FedAvg Workflow

| simulate_fedavg_train.py | collab_fedavg_train.py |
|--------------------------|------------------------|

```python
def fed_avg(clients, num_rounds=5):
    print(f"Starting FedAvg for {num_rounds} rounds")
    global_weights = None

    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1} ===")

        # Each client trains (simulated sequentially)
        client_results = {}
        for client_id in clients:
            weights, loss = train(global_weights)
            client_results[client_id] = (weights, loss)
            print(f"  [{client_id}] Round {round_num}, ...")

        # Aggregate results
        global_weights, global_loss = weighted_avg(client_results)

        print(f"  Global average loss: {global_loss:.4f}")

    print(f"\nFedAvg completed after {num_rounds} rounds")
    return global_weights
```

```python
class FedAvg:                                            # + Wrap in class
    def __init__(self, num_rounds=5):                    # + Constructor
        self.num_rounds = num_rounds                     # +

    @fox.algo                                            # + Add decorator
    def fed_avg(self):                                   # + No clients param
        print(f"Starting FedAvg for {self.num_rounds} rounds")
        global_weights = None

        for round_num in range(self.num_rounds):
            print(f"\n=== Round {round_num + 1} ===")

            # Each client trains (in parallel via fox.clients)
            client_results = fox.clients.train(global_weights)
            # ^^^ Replaces the entire for-loop above! ^^^



            # Aggregate results
            global_weights, global_loss = weighted_avg(client_results)

            print(f"  Global average loss: {global_loss:.4f}")

        print(f"\nFedAvg completed after {self.num_rounds} rounds")
        return global_weights
```

**Changes:**
1. Wrap function in `class FedAvg` with `__init__`
2. Add `@fox.algo` decorator  
3. Remove `clients` parameter (Fox manages clients)
4. **Replace for-loop with `fox.clients.train()`** - the key change!

---

### Execution

| simulate_fedavg_train.py | collab_fedavg_train.py |
|--------------------------|------------------------|

```python
if __name__ == "__main__":
    clients = ["site-1", "site-2", ...]
    result = fed_avg(clients, num_rounds=5)

    print()
    print("Job Status:", "Completed")
    print("Results:", "In memory (state_dict)")
```

```python
if __name__ == "__main__":
    server = FedAvg(num_rounds=5)
    client = Trainer()

    recipe = FoxRecipe(
        job_name="fedavg_train",
        server=server,
        client=client,
        min_clients=5,
    )
    env = SimEnv(num_clients=5)

    run = recipe.execute(env)

    print()
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())
```

**Changes:**
1. Create server and client instances
2. Configure via `FoxRecipe`
3. Create execution environment (`SimEnv`)
4. Execute via `recipe.execute(env)`

---

## Summary of Changes

| Component | Change |
|-----------|--------|
| **Imports** | +3 Fox imports |
| **Model** | None |
| **Training** | Wrap in class + `@fox.collab` + `self` |
| **Aggregation** | None |
| **FedAvg** | Wrap in class + `@fox.algo` + `fox.clients.xxx()` |
| **Execution** | Recipe + Environment pattern |

### The Key Insight

The **only algorithmic change** is replacing:
```python
for client_id in clients:
    result = train(global_weights)
```

with:
```python
results = fox.clients.train(global_weights)
```

Everything else is just wrapping functions in classes with decorators!

---

## Running the Examples

```bash
# Local simulation
python simulate_fedavg_train.py

# Fox federated simulation  
python collab_fedavg_train.py
```

Both produce similar output - but `collab_fedavg_train.py` runs clients in **true parallel** and can be deployed to real distributed environments.
