"""Parallel Federated Averaging Simulation.

This script demonstrates federated averaging with parallel client training
using standalone functions (no classes) - directly comparable to collab_fedavg_no_class.py.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# Framework Abstraction (hides parallel execution details)
# =============================================================================


class ClientGroup:
    """A group of clients that can execute functions in parallel.

    This abstraction hides ThreadPoolExecutor details, providing a simple
    interface similar to fox.clients in the Fox API.
    """

    def __init__(self, client_ids, max_workers=None):
        """Initialize the client group.

        Args:
            client_ids: List of client identifiers.
            max_workers: Maximum parallel workers (defaults to len(client_ids)).
        """
        self.client_ids = client_ids
        self.max_workers = max_workers or len(client_ids)
        self._functions = {}

    def register(self, func):
        """Register a function that can be called on all clients."""
        self._functions[func.__name__] = func
        return func

    def __getattr__(self, func_name):
        """Allow calling any registered function in parallel across all clients.

        Example: clients.train(weights) calls train(client_id, weights)
                 for each client in parallel.
        """
        if func_name.startswith("_"):
            raise AttributeError(func_name)

        func = self._functions.get(func_name)
        if func is None:
            raise AttributeError(f"Function '{func_name}' not registered")

        def parallel_call(*args, **kwargs):
            results = []
            with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="client") as executor:
                # Submit all client tasks
                future_to_client = {
                    executor.submit(func, client_id, *args, **kwargs): client_id for client_id in self.client_ids
                }

                # Collect results as they complete
                for future in as_completed(future_to_client):
                    client_id = future_to_client[future]
                    try:
                        result = future.result()
                        results.append((client_id, result))
                    except Exception as ex:
                        results.append((client_id, ex))

            return results

        return parallel_call


# =============================================================================
# User Code (focuses on training logic - standalone functions, no classes!)
# =============================================================================


# 1. Define Model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# 2. Setup client group
client_ids = ["site-1", "site-2", "site-3", "site-4", "site-5"]
clients = ClientGroup(client_ids)


# 3. Define training function (standalone, no class!)
@clients.register
def train(client_id, weights=None):
    """Train a local model - standalone function, not a method.

    Args:
        client_id: Identifier for the client.
        weights: Optional global model weights to initialize from.

    Returns:
        Tuple of (updated_weights, final_loss)
    """
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
    loss = None

    # Training loop
    for epoch in range(5):
        for batch in dataloader:
            batch_inputs, batch_labels = batch
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    print(f"  [{client_id}] Loss: {loss.item():.4f}")

    # Return updated weights and loss
    return model.state_dict(), loss.item()


# 4. Define weighted averaging function
def weighted_avg(client_results):
    """Aggregate client results using simple averaging."""
    valid_results = {}
    for client_id, result in client_results:
        if isinstance(result, Exception):
            print(f"  Warning: {client_id} failed: {result}")
            continue
        valid_results[client_id] = result

    all_weights = [result[0] for result in valid_results.values()]
    all_losses = [result[1] for result in valid_results.values()]

    avg_weights = {}
    for key in all_weights[0].keys():
        avg_weights[key] = torch.stack([w[key].float() for w in all_weights]).mean(dim=0)

    avg_loss = sum(all_losses) / len(all_losses)
    return avg_weights, avg_loss


# 5. Define federated averaging workflow (standalone function!)
NUM_ROUNDS = 5


def fed_avg():
    """Run federated averaging - standalone function, not a method."""
    print(f"Starting FedAvg for {NUM_ROUNDS} rounds")
    global_weights = None

    for round_num in range(NUM_ROUNDS):
        print(f"\n=== Round {round_num + 1} ===")

        # Each client trains (in parallel via clients group)
        # Similar to: fox.clients.train(global_weights)
        client_results = clients.train(global_weights)

        # Aggregate results using weighted average
        global_weights, global_loss = weighted_avg(client_results)

        # Print global loss
        print(f"  Global average loss: {global_loss:.4f}")

    print(f"\nFedAvg completed after {NUM_ROUNDS} rounds")
    return global_weights


# 6. Execute the training
if __name__ == "__main__":
    result = fed_avg()

    print()
    print("Job Status:", "Completed")
    print("Results:", "In memory (state_dict)")
