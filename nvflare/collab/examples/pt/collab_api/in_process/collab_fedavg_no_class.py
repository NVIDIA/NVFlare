"""Federated Averaging with Collab - NO CLASSES NEEDED!

This demonstrates using Collab with standalone functions only.
Just define @fox.algo and @fox.collab functions, then call CollabRecipe()!

CollabRecipe automatically uses the current module when server/client are not specified.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from nvflare.collab import fox
from nvflare.collab.sim import SimEnv
from nvflare.collab.sys.recipe import CollabRecipe

# =============================================================================
# Define Model (same as simulate_parallel_fedavg_train.py)
# =============================================================================


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# =============================================================================
# Client-side: Training Function (standalone, no class!)
# =============================================================================


@fox.collab
def train(weights=None):
    """Train a local model - standalone function, not a method."""
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

    print(f"  [{fox.site_name}] Loss: {loss.item():.4f}")

    # Return updated weights and loss
    return model.state_dict(), loss.item()


# =============================================================================
# Aggregation Function (same as simulate_parallel_fedavg_train.py)
# =============================================================================


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


# =============================================================================
# Server-side: FedAvg Algorithm (standalone function, no class!)
# =============================================================================

NUM_ROUNDS = 5  # Configuration as module variable


@fox.algo
def fed_avg():
    """Federated averaging - standalone function, not a method."""
    print(f"Starting FedAvg for {NUM_ROUNDS} rounds")
    global_weights = None

    for round_num in range(NUM_ROUNDS):
        print(f"\n=== Round {round_num + 1} ===")

        # Each client trains (in parallel via fox.clients)
        # Same pattern as simulate_parallel_fedavg_train.py!
        client_results = fox.clients.train(global_weights)

        # Aggregate results using weighted average
        global_weights, global_loss = weighted_avg(client_results)

        # Print global loss
        print(f"  Global average loss: {global_loss:.4f}")

    print(f"\nFedAvg completed after {NUM_ROUNDS} rounds")
    return global_weights


# =============================================================================
# Execute - CollabRecipe auto-detects this module!
# =============================================================================

if __name__ == "__main__":
    # That's it! CollabRecipe uses the caller's module automatically
    recipe = CollabRecipe(job_name="fedavg_no_class", min_clients=5)
    env = SimEnv(num_clients=5)

    run = recipe.execute(env)

    print()
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())
