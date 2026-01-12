"""Parallel Federated Averaging Simulation.

This script demonstrates federated averaging with parallel client training.
The parallelism is abstracted away in the ClientGroup class, allowing users
to focus on training logic - similar to the Fox API pattern.
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
    """A group of clients that can execute methods in parallel.

    This abstraction hides ThreadPoolExecutor details, providing a simple
    interface similar to fox.clients in the Fox API.
    """

    def __init__(self, client_ids, trainer, max_workers=None):
        """Initialize the client group.

        Args:
            client_ids: List of client identifiers.
            trainer: The trainer object with training methods.
            max_workers: Maximum parallel workers (defaults to len(client_ids)).
        """
        self.client_ids = client_ids
        self.trainer = trainer
        self.max_workers = max_workers or len(client_ids)

    def __getattr__(self, method_name):
        """Allow calling any method on the trainer in parallel across all clients.

        Example: clients.train(weights) calls trainer.train(client_id, weights)
                 for each client in parallel.
        """

        def parallel_call(*args, **kwargs):
            results = []
            with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="client") as executor:
                # Get the method from trainer
                method = getattr(self.trainer, method_name)

                # Submit all client tasks
                future_to_client = {
                    executor.submit(method, client_id, *args, **kwargs): client_id for client_id in self.client_ids
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
# User Code (focuses on training logic)
# =============================================================================

# 1. Define Model, Data, and Training Function


# 1.1 Define Model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# 1.2 Define Training Class
class Trainer:
    def train(self, client_id, weights=None):
        """Train a local model for a single client.

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


# 1.3 Define weighted averaging function
def weighted_avg(client_results):
    """Aggregate client results using simple averaging.

    Args:
        client_results: List of (client_id, (weights, loss)) tuples.

    Returns:
        Tuple of (averaged_weights, averaged_loss)
    """
    # Filter out any exceptions
    valid_results = {}
    for client_id, result in client_results:
        if isinstance(result, Exception):
            print(f"  Warning: {client_id} failed: {result}")
            continue
        valid_results[client_id] = result

    all_weights = [result[0] for result in valid_results.values()]
    all_losses = [result[1] for result in valid_results.values()]

    # Simple averaging of model parameters
    avg_weights = {}
    for key in all_weights[0].keys():
        avg_weights[key] = torch.stack([w[key].float() for w in all_weights]).mean(dim=0)

    avg_loss = sum(all_losses) / len(all_losses)
    return avg_weights, avg_loss


# 1.4 Define federated averaging workflow
def fed_avg(clients, num_rounds=5):
    """Run federated averaging with parallel client training.

    Args:
        clients: ClientGroup object for parallel execution.
        num_rounds: Number of federated rounds.

    Returns:
        Final global model weights.
    """
    print(f"Starting FedAvg for {num_rounds} rounds")
    global_weights = None

    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1} ===")

        # Each client trains (in parallel via clients group)
        # Similar to: fox.clients.train(global_weights)
        client_results = clients.train(global_weights)

        # Aggregate results using weighted average
        global_weights, global_loss = weighted_avg(client_results)

        # Print global loss
        print(f"  Global average loss: {global_loss:.4f}")

    print(f"\nFedAvg completed after {num_rounds} rounds")
    return global_weights


# 2. Execute the training
if __name__ == "__main__":
    # Setup - similar to FoxRecipe setup
    client_ids = ["site-1", "site-2", "site-3", "site-4", "site-5"]
    trainer = Trainer()
    clients = ClientGroup(client_ids, trainer)

    # Run federated averaging
    result = fed_avg(clients, num_rounds=5)

    print()
    print("Job Status:", "Completed")
    print("Results:", "In memory (state_dict)")
