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

"""Simulated Federated Learning with DDP (No Fox/FLARE API).

This script simulates federated learning where:
- Multiple FL CLIENTS participate (e.g., site-1, site-2, site-3)
- Each CLIENT can run DDP training with multiple GPUs
- A central SERVER aggregates client models using FedAvg

This demonstrates the FL+DDP pattern WITHOUT using the Fox Collab API.
Compare with job.py/client.py/server.py which uses the Fox API.

Architecture:
    Server (main process)
      ├── Client site-1 (subprocess, can use DDP internally)
      ├── Client site-2 (subprocess, can use DDP internally)
      └── Client site-3 (subprocess, can use DDP internally)

Usage:
    # Simple simulation (no DDP within clients)
    python sim_distributed_fedavg_train.py

    # With DDP within each client (2 GPUs per client)
    python sim_distributed_fedavg_train.py --gpus 2

Comparison:
    - distributed_train.py: Pure DDP training on single node (no FL)
    - sim_distributed_fedavg_train.py: FL simulation with optional DDP (this file)
    - job.py + client.py + server.py: FL using Fox Collab API
"""

import argparse
import multiprocessing as mp
import os
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# Model Definition
# =============================================================================


class SimpleModel(nn.Module):
    """Simple linear model for demonstration."""

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# =============================================================================
# Client-side Training
# =============================================================================


def client_train_single(global_weights, client_id, num_epochs=5):
    """Single-process local training on a client.

    Args:
        global_weights: Global model weights from server (None for first round)
        client_id: Client identifier (e.g., "site-1")
        num_epochs: Number of local training epochs

    Returns:
        Tuple of (updated_weights, final_loss)
    """
    # Create local dataset (each client has different data in real FL)
    # Use client_id to seed different data for each client
    torch.manual_seed(hash(client_id) % 2**32)
    inputs = torch.randn(100, 10)
    labels = torch.randn(100, 1)
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=10)

    # Initialize model with global weights
    model = SimpleModel()
    if global_weights is not None:
        model.load_state_dict(global_weights)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Local training loop
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


def client_train_ddp(global_weights, client_id, num_gpus, num_epochs=5):
    """DDP training on a client using torchrun.

    This spawns a torchrun subprocess that runs distributed_train.py
    with the given global weights.

    Args:
        global_weights: Global model weights from server
        client_id: Client identifier
        num_gpus: Number of GPUs to use for DDP
        num_epochs: Number of local training epochs

    Returns:
        Tuple of (updated_weights, final_loss)
    """
    # Save global weights to temp file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        weights_file = f.name
        if global_weights is not None:
            torch.save(global_weights, weights_file)
        else:
            torch.save({}, weights_file)

    # Output file for results
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        output_file = f.name

    try:
        # Run DDP training via torchrun
        script_path = Path(__file__).parent / "distributed_train.py"

        env = os.environ.copy()
        env["FL_CLIENT_ID"] = client_id
        env["FL_WEIGHTS_FILE"] = weights_file
        env["FL_OUTPUT_FILE"] = output_file
        env["FL_NUM_EPOCHS"] = str(num_epochs)

        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            str(script_path),
        ]

        print(f"  [{client_id}] Starting DDP training with {num_gpus} processes...")
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            print(f"  [{client_id}] DDP training failed:")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return None, float("inf")

        # Load results
        output = torch.load(output_file, weights_only=False)
        return output["weights"], output["loss"]

    finally:
        # Cleanup temp files
        Path(weights_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)


# =============================================================================
# Server-side Aggregation
# =============================================================================


def fedavg_aggregate(client_results):
    """FedAvg aggregation of client models.

    Args:
        client_results: List of (client_id, weights, loss) tuples

    Returns:
        Tuple of (aggregated_weights, average_loss)
    """
    valid_results = [(cid, w, l) for cid, w, l in client_results if w is not None]

    if not valid_results:
        raise RuntimeError("All clients failed!")

    # Average weights
    num_clients = len(valid_results)
    avg_weights = {}
    first_weights = valid_results[0][1]

    for key in first_weights.keys():
        stacked = torch.stack([w[key].float() for _, w, _ in valid_results])
        avg_weights[key] = stacked.mean(dim=0)

    # Average loss
    avg_loss = sum(l for _, _, l in valid_results) / num_clients

    return avg_weights, avg_loss


# =============================================================================
# Client Process (for parallel execution)
# =============================================================================


def run_client(client_id, global_weights, num_gpus, num_epochs, result_queue):
    """Run a single client's training (called in subprocess)."""
    try:
        if num_gpus > 0:
            weights, loss = client_train_ddp(global_weights, client_id, num_gpus, num_epochs)
        else:
            weights, loss = client_train_single(global_weights, client_id, num_epochs)

        result_queue.put((client_id, weights, loss))
    except Exception as e:
        result_queue.put((client_id, None, str(e)))


# =============================================================================
# Main FL Simulation
# =============================================================================


def run_federated_learning(num_clients=3, num_rounds=5, num_gpus=0, local_epochs=5):
    """Run simulated federated learning.

    Args:
        num_clients: Number of FL clients
        num_rounds: Number of FL rounds
        num_gpus: GPUs per client (0 = single-process, >0 = DDP)
        local_epochs: Local training epochs per round
    """
    print("=" * 60)
    print("Simulated Federated Learning (No Fox API)")
    print("=" * 60)
    print(f"  Clients: {num_clients}")
    print(f"  Rounds:  {num_rounds}")
    print(f"  GPUs per client: {num_gpus if num_gpus > 0 else 'None (CPU)'}")
    print("=" * 60)

    client_ids = [f"site-{i+1}" for i in range(num_clients)]
    global_weights = None

    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")

        # Run all clients (can be parallel or sequential)
        if num_gpus > 0:
            # Sequential for DDP (each client uses all GPUs)
            client_results = []
            for client_id in client_ids:
                weights, loss = client_train_ddp(global_weights, client_id, num_gpus, local_epochs)
                print(f"  [{client_id}] Local loss: {loss:.4f}")
                client_results.append((client_id, weights, loss))
        else:
            # Parallel for single-process training
            result_queue = mp.Queue()
            processes = []

            for client_id in client_ids:
                p = mp.Process(
                    target=run_client,
                    args=(client_id, global_weights, num_gpus, local_epochs, result_queue),
                )
                p.start()
                processes.append(p)

            # Collect results
            client_results = []
            for _ in client_ids:
                result = result_queue.get()
                client_id, weights, loss = result
                if weights is not None:
                    print(f"  [{client_id}] Local loss: {loss:.4f}")
                else:
                    print(f"  [{client_id}] Failed: {loss}")
                client_results.append(result)

            for p in processes:
                p.join()

        # Server aggregation
        global_weights, global_loss = fedavg_aggregate(client_results)
        print(f"  [Server] Global avg loss: {global_loss:.4f}")

    print("\n" + "=" * 60)
    print(f"Simulated FL completed after {num_rounds} rounds")
    print("=" * 60)

    return global_weights


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Simulated FL with optional DDP (no Fox API)")
    parser.add_argument("--num-clients", type=int, default=3, help="Number of FL clients")
    parser.add_argument("--num-rounds", type=int, default=3, help="Number of FL rounds")
    parser.add_argument("--gpus", type=int, default=0, help="GPUs per client (0=CPU)")
    parser.add_argument("--epochs", type=int, default=5, help="Local epochs per round")
    args = parser.parse_args()

    run_federated_learning(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        num_gpus=args.gpus,
        local_epochs=args.epochs,
    )
