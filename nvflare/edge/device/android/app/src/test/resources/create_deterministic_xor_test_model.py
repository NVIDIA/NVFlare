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
"""
Script to create a DETERMINISTIC XOR PyTorch model for Android/iOS testing.
This generates a simple XOR neural network model with fixed weights and ensures
deterministic training results for reliable testing.
"""

import base64
import json

import numpy as np
import torch
import torch.nn as nn
from executorch.exir import to_edge
from torch.export import export
from torch.export.experimental import _export_forward_backward

# Set seeds for deterministic behavior
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


class DeterministicXorNet(nn.Module):
    """Deterministic XOR neural network with fixed initial weights.
    This ensures reproducible training results for testing.
    """

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 4)
        self.sigmoid_1 = nn.Sigmoid()
        self.linear2 = nn.Linear(4, 2)

        # Set fixed weights for deterministic behavior
        with torch.no_grad():
            # Fixed weights for linear1
            self.linear1.weight.data = torch.tensor(
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=torch.float32
            )
            self.linear1.bias.data = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)

            # Fixed weights for linear2
            self.linear2.weight.data = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=torch.float32)
            self.linear2.bias.data = torch.tensor([0.1, 0.2], dtype=torch.float32)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid_1(x)
        x = self.linear2(x)
        return x


class DeterministicDeviceModel(nn.Module):
    """Wrapper model that includes loss calculation for training.
    Uses deterministic loss function.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, label):
        pred = self.net(input)
        return self.loss(pred, label), pred.detach().argmax(dim=1)


def export_model(net, input_tensor_example, label_tensor_example):
    """Export model to ExecuTorch format."""
    # Captures the forward graph
    ep = export(net, (input_tensor_example, label_tensor_example), strict=True)
    # Captures the backward graph
    ep = _export_forward_backward(ep)
    # Lower the graph to edge dialect
    ep = to_edge(ep)
    # Lower the graph to executorch
    ep = ep.to_executorch()
    return ep


def create_deterministic_xor_model():
    """Create a deterministic XOR model with fixed weights."""
    print("Creating DETERMINISTIC XOR neural network model...")
    print("Using fixed weights for reproducible results...")

    # Create the base XOR network with fixed weights
    xor_net = DeterministicXorNet()

    # Create the training wrapper
    training_net = DeterministicDeviceModel(xor_net)
    training_net.train()

    # Create FIXED example inputs for export (deterministic)
    # XOR input: 2 features (x1, x2)
    # XOR output: 2 classes (0, 1)
    example_input = torch.tensor(
        [
            [0.0, 0.0],  # XOR(0,0) = 0
            [0.0, 1.0],  # XOR(0,1) = 1
            [1.0, 0.0],  # XOR(1,0) = 1
            [1.0, 1.0],  # XOR(1,1) = 0
        ],
        dtype=torch.float32,
    )

    example_label = torch.tensor([0, 1, 1, 0], dtype=torch.long)

    print(f"Fixed example input shape: {example_input.shape}")
    print(f"Fixed example label shape: {example_label.shape}")
    print(f"Example input: {example_input}")
    print(f"Example labels: {example_label}")

    # Export the model to ExecuTorch format
    print("Exporting deterministic model to ExecuTorch format...")
    exported_program = export_model(training_net, example_input, example_label)

    # Save the model
    model_path = "deterministic_xor_test_model.pte"
    with open(model_path, "wb") as f:
        f.write(exported_program.buffer)

    print(f"Deterministic XOR model saved to {model_path}")
    return model_path


def convert_model_to_base64(model_path):
    """Convert model file to base64 string."""
    with open(model_path, "rb") as f:
        model_bytes = f.read()

    model_base64 = base64.b64encode(model_bytes).decode("utf-8")
    print(f"Deterministic XOR model converted to base64 (length: {len(model_base64)})")
    return model_base64


def create_deterministic_test_config(model_base64):
    """Create a test configuration with deterministic XOR model."""
    config_path = "proto_test_config_deterministic_xor.json"

    # Create a deterministic config structure for XOR testing
    config = {
        "headers": {
            "device_info": {
                "device_id": "1",
                "app_name": "test",
                "app_version": "1.0",
                "platform": "android",
                "platform_version": "1.2.2",
            },
            "user_info": {"user_id": "xyz"},
        },
        "steps": [
            {
                "job": {"job_name": "federated_learning"},
                "response": {
                    "status": "OK",
                    "job_id": "12345",
                    "job_name": "federated_learning",
                    "job_data": {
                        "config": {
                            "components": [
                                {
                                    "type": "Trainer.DLTrainer",
                                    "name": "trainer",
                                    "args": {
                                        "epoch": 5,
                                        "lr": 0.1,  # Fixed learning rate for determinism
                                        "method": "xor",
                                    },
                                }
                            ],
                            "executors": {"*": "@trainer"},
                        }
                    },
                },
            },
            {
                "task": {"job_id": "12345"},
                "response": {
                    "status": "OK",
                    "job_id": "12345",
                    "task_id": "task12345",
                    "task_name": "train",
                    "task_data": {
                        "kind": "model",
                        "data": {"model_buffer": model_base64},
                        "meta": {
                            "learning_rate": 0.1,  # Fixed learning rate
                            "batch_size": 4,  # Fixed batch size (all 4 XOR samples)
                            "epochs": 5,  # Fixed epochs
                            "method": "xor",
                        },
                    },
                    "cookie": {"model_version": 1, "selection_id": 101},
                },
            },
            {
                "report": {
                    "job_id": "12345",
                    "task_id": "task12345",
                    "task_name": "train",
                    "status": "OK",
                    "result": {
                        "kind": "number",
                        "data": {
                            "value": 0.693147,  # Expected loss value (ln(2)) for deterministic XOR
                            "count": 20,  # Expected count: 4 samples * 5 epochs
                        },
                    },
                    "cookie": {"model_version": 1, "selection_id": 101},
                },
                "response": {"status": "OK", "task_id": "task12345", "task_name": "train"},
            },
            {
                "task": {"job_id": "12345", "cookie": {"model_version": 1, "selection_id": 101}},
                "response": {"status": "DONE"},
            },
        ],
    }

    # Save config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created {config_path} with deterministic XOR model data")
    print("Expected deterministic results:")
    print("  - Loss value: ~0.693147 (ln(2))")
    print("  - Count: 20 (4 samples × 5 epochs)")


def main():
    """Main function to create deterministic XOR test model and config."""
    print("=" * 60)
    print("Creating DETERMINISTIC XOR Test PyTorch Model")
    print("=" * 60)
    print("This model uses fixed weights and deterministic training")
    print("for reproducible testing results.")
    print()

    model_path = create_deterministic_xor_model()

    print("\nConverting deterministic XOR model to base64...")
    model_base64 = convert_model_to_base64(model_path)

    print("\nCreating deterministic test configuration...")
    create_deterministic_test_config(model_base64)

    print("\n" + "=" * 60)
    print("DETERMINISTIC XOR Test Setup Complete!")
    print("=" * 60)
    print("Files created:")
    print(f"  - {model_path} (Deterministic XOR PyTorch model)")
    print("  - proto_test_config_deterministic_xor.json (Test configuration)")
    print("\nUsage:")
    print(
        "  python -m nvflare.edge.web.routing_proxy 4321 lcp_map.json rootCA.pem proto_test_config_deterministic_xor.json"
    )
    print("\nDeterministic Features:")
    print("  - Fixed initial weights")
    print("  - Fixed learning rate (0.1)")
    print("  - Fixed batch size (4)")
    print("  - Fixed epochs (5)")
    print("  - Fixed XOR data order")
    print("  - Expected loss: ~0.693147")
    print("  - Expected count: 20")
    print("\nThis ensures reproducible results for testing!")


if __name__ == "__main__":
    main()
