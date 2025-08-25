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
Script to create a test XOR PyTorch model for Android/iOS testing.
This generates a simple XOR neural network model and converts it to base64 format.
Based on the reference ExecuTorch training example.
"""

import base64
import json

import torch
import torch.nn as nn
from executorch.exir import to_edge
from torch.export import export
from torch.export.experimental import _export_forward_backward


class XorNet(nn.Module):
    """Simple XOR neural network for testing.
    Based on the reference implementation in nvflare/edge/models/model.py
    """

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 4)
        self.sigmoid_1 = nn.Sigmoid()
        self.linear2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid_1(x)
        x = self.linear2(x)
        return x


class DeviceModel(nn.Module):
    """Wrapper model that includes loss calculation for training.
    Based on the reference implementation in nvflare/edge/models/model.py
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, label):
        pred = self.net(input)
        return self.loss(pred, label), pred.detach().argmax(dim=1)


def export_model(net, input_tensor_example, label_tensor_example):
    """Export model to ExecuTorch format.
    Based on the reference implementation in nvflare/edge/models/model.py
    """
    # Captures the forward graph
    ep = export(net, (input_tensor_example, label_tensor_example), strict=True)
    # Captures the backward graph
    ep = _export_forward_backward(ep)
    # Lower the graph to edge dialect
    ep = to_edge(ep)
    # Lower the graph to executorch
    ep = ep.to_executorch()
    return ep


def create_xor_model():
    """Create a simple XOR model and save it."""
    print("Creating XOR neural network model...")

    # Create the base XOR network
    xor_net = XorNet()

    # Create the training wrapper
    training_net = DeviceModel(xor_net)
    training_net.train()

    # Create example inputs for export
    # XOR input: 2 features (x1, x2)
    # XOR output: 2 classes (0, 1)
    example_input = torch.randn(4, 2)  # 4 samples, 2 features
    example_label = torch.randint(0, 2, (4,))  # 4 labels, 0 or 1

    print(f"Example input shape: {example_input.shape}")
    print(f"Example label shape: {example_label.shape}")

    # Export the model to ExecuTorch format
    print("Exporting model to ExecuTorch format...")
    exported_program = export_model(training_net, example_input, example_label)

    # Save the model
    model_path = "xor_test_model.pte"
    with open(model_path, "wb") as f:
        f.write(exported_program.buffer)

    print(f"XOR model saved to {model_path}")
    return model_path


def convert_model_to_base64(model_path):
    """Convert model file to base64 string."""
    with open(model_path, "rb") as f:
        model_bytes = f.read()

    model_base64 = base64.b64encode(model_bytes).decode("utf-8")
    print(f"XOR model converted to base64 (length: {len(model_base64)})")
    return model_base64


def update_test_config(model_base64):
    """Update the test configuration with the XOR model."""
    config_path = "proto_test_config_xor.json"

    # Create a basic config structure for XOR testing
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
                                    "args": {"epoch": 5, "lr": 0.0001, "method": "xor"},
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
                        "meta": {"learning_rate": 0.0001, "batch_size": 4, "epochs": 5, "method": "xor"},
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
                    "result": {"kind": "number", "data": {"value": 0.123, "count": 100}},
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

    print(f"Created {config_path} with XOR model data")
    print("Note: XOR training returns simple numeric results, not tensor differences")


def main():
    """Main function to create XOR test model and config."""
    print("Creating XOR test PyTorch model...")
    model_path = create_xor_model()

    print("Converting XOR model to base64...")
    model_base64 = convert_model_to_base64(model_path)

    print("Creating XOR test configuration...")
    update_test_config(model_base64)

    print("\n" + "=" * 50)
    print("XOR Test Setup Complete!")
    print("=" * 50)
    print("Files created:")
    print(f"  - {model_path} (XOR PyTorch model)")
    print("  - proto_test_config_xor.json (Test configuration)")
    print("\nUsage:")
    print("  python -m nvflare.edge.web.routing_proxy 4321 lcp_map.json rootCA.pem proto_test_config_xor.json")
    print("\nThe XOR model:")
    print("  - Input: 2 features (x1, x2)")
    print("  - Output: 2 classes (0, 1)")
    print("  - Architecture: Linear(2,4) -> Sigmoid -> Linear(4,2)")
    print("  - Perfect for testing federated learning with simple data")


if __name__ == "__main__":
    main()
