"""Server-side aggregation logic for Federated Averaging.

This module contains only the @fox.main decorated functions
that run on the server.
"""

import torch

from nvflare.collab import fox

NUM_ROUNDS = 5  # Configuration as module variable


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


@fox.main
def fed_avg():
    """Federated averaging - runs on server."""
    print(f"Starting FedAvg for {NUM_ROUNDS} rounds")
    global_weights = None

    for round_num in range(NUM_ROUNDS):
        print(f"\n=== Round {round_num + 1} ===")

        # Call client's train() method on all clients
        client_results = fox.clients.train(global_weights)

        # Aggregate results using weighted average
        global_weights, global_loss = weighted_avg(client_results)

        # Print global loss
        print(f"  Global average loss: {global_loss:.4f}")

    print(f"\nFedAvg completed after {NUM_ROUNDS} rounds")
    return global_weights
