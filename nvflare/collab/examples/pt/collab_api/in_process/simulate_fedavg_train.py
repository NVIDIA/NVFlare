import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Define Model, Data, and Training Function


# 1.1 Define Model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


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


# 1.3 Define weighted averaging function
def weighted_avg(client_results):
    # client_results: [(client_id, (weights, loss)), ...]
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
    print(f"Starting FedAvg for {num_rounds} rounds")
    global_weights = None

    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1} ===")

        # Each client trains (simulated sequentially)
        client_results = []
        for client_id in clients:
            weights, loss = train(global_weights)
            client_results.append((client_id, (weights, loss)))
            print(f"  [{client_id}] Round {round_num}, Loss: {loss:.4f}")

        # Aggregate results using weighted average
        global_weights, global_loss = weighted_avg(client_results)

        # Print global loss
        print(f"  Global average loss: {global_loss:.4f}")

    print(f"\nFedAvg completed after {num_rounds} rounds")
    return global_weights


# 2. Execute the training
if __name__ == "__main__":
    clients = ["site-1", "site-2", "site-3", "site-4", "site-5"]
    result = fed_avg(clients, num_rounds=5)

    print()
    print("Job Status:", "Completed")
    print("Results:", "In memory (state_dict)")
