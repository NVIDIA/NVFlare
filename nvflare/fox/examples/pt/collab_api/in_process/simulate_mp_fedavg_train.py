"""Multi-Process Federated Averaging Simulation using torch.multiprocessing.

This script demonstrates federated averaging with parallel client training
using torch.multiprocessing for true multi-process execution.

This is closer to real distributed training than the thread-based version.
"""

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# Model Definition
# =============================================================================


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# =============================================================================
# Client Training Function (runs in separate process)
# =============================================================================


def train(client_id: str, weights_dict: dict, result_queue: mp.Queue):
    """Train a local model in a separate process.

    Args:
        client_id: Identifier for the client.
        weights_dict: Global model weights to initialize from (dict or None).
        result_queue: Queue to put results back to main process.
    """
    try:
        # Setup data (each client has different random data)
        torch.manual_seed(hash(client_id) % 2**32)
        inputs = torch.randn(100, 10)
        labels = torch.randn(100, 1)
        dataset = TensorDataset(inputs, labels)
        dataloader = DataLoader(dataset, batch_size=10)

        # Create and initialize model
        model = SimpleModel()
        if weights_dict is not None:
            model.load_state_dict(weights_dict)

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

        final_loss = loss.item()
        print(f"  [{client_id}] Loss: {final_loss:.4f}")

        # Put result in queue
        result_queue.put((client_id, model.state_dict(), final_loss))

    except Exception as ex:
        result_queue.put((client_id, None, str(ex)))


# =============================================================================
# Aggregation Function
# =============================================================================


def weighted_avg(results):
    """Aggregate client results using simple averaging.

    Args:
        results: List of (client_id, weights_dict, loss) tuples.

    Returns:
        Tuple of (averaged_weights, average_loss)
    """
    valid_results = []
    for client_id, weights, loss in results:
        if weights is None:
            print(f"  Warning: {client_id} failed: {loss}")
            continue
        valid_results.append((weights, loss))

    if not valid_results:
        raise RuntimeError("All clients failed!")

    all_weights = [r[0] for r in valid_results]
    all_losses = [r[1] for r in valid_results]

    # Average weights
    avg_weights = {}
    for key in all_weights[0].keys():
        avg_weights[key] = torch.stack([w[key].float() for w in all_weights]).mean(dim=0)

    avg_loss = sum(all_losses) / len(all_losses)
    return avg_weights, avg_loss


# =============================================================================
# Federated Averaging Workflow
# =============================================================================


def fed_avg(client_ids: list, num_rounds: int = 5):
    """Run federated averaging using multiprocessing.

    Args:
        client_ids: List of client identifiers.
        num_rounds: Number of training rounds.

    Returns:
        Final global weights.
    """
    print(f"Starting FedAvg for {num_rounds} rounds with {len(client_ids)} clients")
    print("Using torch.multiprocessing with 'spawn' start method")

    global_weights = None

    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1} ===")

        # Create a queue for collecting results
        result_queue = mp.Queue()

        # Convert weights to regular dict for pickling (if needed)
        weights_to_send = None
        if global_weights is not None:
            weights_to_send = {k: v.clone() for k, v in global_weights.items()}

        # Spawn training processes for each client
        processes = []
        for client_id in client_ids:
            p = mp.Process(
                target=train,
                args=(client_id, weights_to_send, result_queue),
            )
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Collect results from queue
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        # Aggregate results
        global_weights, global_loss = weighted_avg(results)

        print(f"  Global average loss: {global_loss:.4f}")

    print(f"\nFedAvg completed after {num_rounds} rounds")
    return global_weights


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Required for multiprocessing on some platforms (e.g., macOS, Windows)
    mp.set_start_method("spawn", force=True)

    # Define clients
    client_ids = ["site-1", "site-2", "site-3", "site-4", "site-5"]

    # Run federated averaging
    result = fed_avg(client_ids, num_rounds=5)

    print()
    print("Job Status:", "Completed")
    print("Results:", "In memory (state_dict)")
