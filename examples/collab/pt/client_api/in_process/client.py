# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Client-side training using Client API pattern for in-process execution.

This module demonstrates the standard Client API pattern:

    import nvflare.client as flare

    flare.init()
    while flare.is_running():
        model = flare.receive()
        # ... training ...
        flare.send(result)

The training function is registered with CollabClientAPI and called when
the server invokes execute(). The framework sets CLIENT_API_TYPE=COLLAB_IN_PROCESS_API
and the API instance before calling the training function.

This is the SAME code pattern used for standard NVFlare Client API - the only
difference is the underlying implementation (CollabClientAPI vs InProcessClientAPI).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import Client API as 'flare' - SAME as standard NVFlare Client API!
import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel


class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Cache for client data - ensures consistent data across rounds
_client_data_cache = {}


def _get_client_data(site_name: str, num_samples: int = 100):
    """Get or create consistent training data for a client.

    Each client has its own fixed dataset that persists across rounds.
    This simulates real FL where each client has its own local data.

    Args:
        site_name: Client identifier for caching.
        num_samples: Number of samples.

    Returns:
        Tuple of (inputs, labels).
    """
    if site_name not in _client_data_cache:
        # Use site_name hash as seed for reproducibility
        seed = hash(site_name) % (2**32)
        generator = torch.Generator().manual_seed(seed)

        # Create a simple regression problem: y = sum(x) + noise
        inputs = torch.randn(num_samples, 10, generator=generator)
        # Labels based on a consistent function of inputs
        labels = inputs.sum(dim=1, keepdim=True) * 0.1 + torch.randn(num_samples, 1, generator=generator) * 0.1

        _client_data_cache[site_name] = (inputs, labels)

    return _client_data_cache[site_name]


def train_local(weights, site_name: str, num_epochs: int = 5):
    """Perform local training.

    Args:
        weights: Initial model weights (None for first round).
        site_name: Client identifier for getting consistent data.
        num_epochs: Number of local training epochs.

    Returns:
        Tuple of (new_weights, loss).
    """
    # Get consistent data for this client
    inputs, labels = _get_client_data(site_name)
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Initialize model
    model = SimpleModel()
    if weights is not None:
        model.load_state_dict(weights)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    final_loss = 0.0
    for epoch in range(num_epochs):
        for batch_inputs, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

    return model.state_dict(), final_loss


def training_loop():
    """Client training loop using standard Client API pattern.

    Uses module-level flare functions:
    - flare.init() - Initialize
    - flare.is_running() - Check if should continue
    - flare.receive() - Get model from server
    - flare.send() - Return results to server
    - flare.get_site_name() - Get this client's name
    """
    flare.init()

    site_name = flare.get_site_name() or "unknown"
    print(f"  [{site_name}] Training loop started")

    while flare.is_running():
        # Receive model from server
        input_model = flare.receive()

        if input_model is None:
            # Server signaled stop
            break

        current_round = input_model.current_round
        total_rounds = input_model.total_rounds
        weights = input_model.params

        print(f"  [{site_name}] Round {current_round}/{total_rounds} - Training...")

        # Perform local training with consistent data for this client
        new_weights, loss = train_local(weights, site_name, num_epochs=5)

        print(f"  [{site_name}] Round {current_round}/{total_rounds} - Loss: {loss:.4f}")

        # Send results back to server
        output_model = FLModel(
            params=new_weights,
            metrics={"loss": loss},
            current_round=current_round,
        )
        flare.send(output_model)

    print(f"  [{site_name}] Training loop completed")
