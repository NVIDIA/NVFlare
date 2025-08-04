#!/usr/bin/env python3
"""
Script to create a DETERMINISTIC CNN PyTorch model for Android/iOS testing.
This generates a simplified CNN with fixed weights for reproducible testing.
"""

import torch
import torch.nn as nn
import base64
import json
import os
import numpy as np
from torch.export import export
from torch.export.experimental import _export_forward_backward
from executorch.exir import to_edge

# Set seeds for deterministic behavior
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

class DeterministicSimpleCNN(nn.Module):
    """Deterministic simplified CNN with fixed weights for testing.
    Uses minimal architecture to ensure reproducible results.
    """
    def __init__(self):
        super().__init__()
        # Simplified CNN: 1 conv layer + 1 linear layer
        self.conv1 = nn.Conv2d(3, 2, kernel_size=2, stride=1, padding=0)  # 3x32x32 -> 2x31x31
        self.pool = nn.MaxPool2d(2, 2)  # 2x31x31 -> 2x15x15
        self.fc1 = nn.Linear(2 * 15 * 15, 2)  # 450 -> 2 classes
        
        # Set fixed weights for deterministic behavior
        with torch.no_grad():
            # Fixed conv weights
            self.conv1.weight.data = torch.tensor([
                [[[0.1, 0.2], [0.3, 0.4]]],  # First filter
                [[[0.5, 0.6], [0.7, 0.8]]]   # Second filter
            ], dtype=torch.float32).repeat(3, 1, 1, 1)  # Repeat for 3 input channels
            self.conv1.bias.data = torch.tensor([0.1, 0.2], dtype=torch.float32)
            
            # Fixed linear weights
            self.fc1.weight.data = torch.tensor([
                [0.1] * 450,  # First output neuron
                [0.2] * 450   # Second output neuron
            ], dtype=torch.float32)
            self.fc1.bias.data = torch.tensor([0.1, 0.2], dtype=torch.float32)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class DeterministicDeviceModel(nn.Module):
    """Wrapper model that includes loss calculation for training."""
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

def create_deterministic_cnn_model():
    """Create a deterministic CNN model with fixed weights."""
    print("Creating DETERMINISTIC simplified CNN model...")
    print("Using fixed weights for reproducible results...")
    
    # Create the simplified CNN with fixed weights
    cnn_net = DeterministicSimpleCNN()
    
    # Create the training wrapper
    training_net = DeterministicDeviceModel(cnn_net)
    training_net.train()
    
    # Create FIXED example inputs for export (deterministic)
    # Small input: 2 samples, 3 channels, 8x8 images (simplified for testing)
    example_input = torch.tensor([
        # Sample 1: Simple pattern
        [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
          [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
          [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
          [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
          [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
          [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
          [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
          [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]]] * 3,  # Repeat for 3 channels
        
        # Sample 2: Different pattern
        [[[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
          [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
          [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
          [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1],
          [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2],
          [0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3],
          [0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4],
          [0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5]]] * 3  # Repeat for 3 channels
    ], dtype=torch.float32)
    
    example_label = torch.tensor([0, 1], dtype=torch.long)
    
    print(f"Fixed example input shape: {example_input.shape}")
    print(f"Fixed example label shape: {example_label.shape}")
    
    # Export the model to ExecuTorch format
    print("Exporting deterministic CNN model to ExecuTorch format...")
    exported_program = export_model(training_net, example_input, example_label)
    
    # Save the model
    model_path = "deterministic_cnn_test_model.pte"
    with open(model_path, "wb") as f:
        f.write(exported_program.buffer)
    
    print(f"Deterministic CNN model saved to {model_path}")
    return model_path

def convert_model_to_base64(model_path):
    """Convert model file to base64 string."""
    with open(model_path, 'rb') as f:
        model_bytes = f.read()
    
    model_base64 = base64.b64encode(model_bytes).decode('utf-8')
    print(f"Deterministic CNN model converted to base64 (length: {len(model_base64)})")
    return model_base64

def create_deterministic_cnn_test_config(model_base64):
    """Create a test configuration with deterministic CNN model."""
    config_path = "proto_test_config_deterministic_cnn.json"
    
    # Create a deterministic config structure for CNN testing
    config = {
        "headers": {
            "device_info": {
                "device_id": "1",
                "app_name": "test",
                "app_version": "1.0",
                "platform": "android",
                "platform_version": "1.2.2"
            },
            "user_info": {
                "user_id": "xyz"
            }
        },
        "steps": [
            {
                "job": {
                    "job_name": "federated_learning"
                },
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
                                        "epoch": 3,
                                        "lr": 0.01,  # Fixed learning rate for determinism
                                        "method": "cnn"
                                    }
                                }
                            ],
                            "executors": {
                                "*": "@trainer"
                            }
                        }
                    }
                }
            },
            {
                "task": {
                    "job_id": "12345"
                },
                "response": {
                    "status": "OK",
                    "job_id": "12345",
                    "task_id": "task12345",
                    "task_name": "train",
                    "task_data": {
                        "kind": "model",
                        "data": {
                            "model_buffer": model_base64
                        },
                        "meta": {
                            "learning_rate": 0.01,  # Fixed learning rate
                            "batch_size": 2,        # Fixed batch size (2 samples)
                            "epochs": 3,           # Fixed epochs
                            "method": "cnn"
                        }
                    },
                    "cookie": {
                        "model_version": 1,
                        "selection_id": 101
                    }
                }
            },
            {
                "report": {
                    "job_id": "12345",
                    "task_id": "task12345",
                    "task_name": "train",
                    "status": "OK",
                    "result": {
                        "kind": "model",
                        "data": {
                            "conv1.weight": {
                                "sizes": [2, 3, 2, 2],
                                "strides": [12, 4, 2, 1],
                                "data": [0.001, -0.002, 0.003, 0.004, 0.005, -0.006, 0.007, 0.008, 0.009, -0.010, 0.011, 0.012, 0.013, -0.014, 0.015, 0.016, 0.017, -0.018, 0.019, 0.020, 0.021, -0.022, 0.023, 0.024]
                            },
                            "conv1.bias": {
                                "sizes": [2],
                                "strides": [1],
                                "data": [0.001, -0.002]
                            },
                            "fc1.weight": {
                                "sizes": [2, 450],
                                "strides": [450, 1],
                                "data": [0.001] * 900  # Simplified for testing
                            },
                            "fc1.bias": {
                                "sizes": [2],
                                "strides": [1],
                                "data": [0.001, -0.002]
                            }
                        }
                    },
                    "cookie": {
                        "model_version": 1,
                        "selection_id": 101
                    }
                },
                "response": {
                    "status": "OK",
                    "task_id": "task12345",
                    "task_name": "train"
                }
            },
            {
                "task": {
                    "job_id": "12345",
                    "cookie": {
                        "model_version": 1,
                        "selection_id": 101
                    }
                },
                "response": {
                    "status": "DONE"
                }
            }
        ]
    }
    
    # Save config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created {config_path} with deterministic CNN model data")
    print("Expected deterministic results:")
    print("  - Tensor differences for conv1.weight, conv1.bias, fc1.weight, fc1.bias")
    print("  - Small, predictable parameter updates")

def main():
    """Main function to create deterministic CNN test model and config."""
    print("="*60)
    print("Creating DETERMINISTIC CNN Test PyTorch Model")
    print("="*60)
    print("This model uses fixed weights and simplified architecture")
    print("for reproducible CNN testing results.")
    print()
    
    model_path = create_deterministic_cnn_model()
    
    print("\nConverting deterministic CNN model to base64...")
    model_base64 = convert_model_to_base64(model_path)
    
    print("\nCreating deterministic CNN test configuration...")
    create_deterministic_cnn_test_config(model_base64)
    
    print("\n" + "="*60)
    print("DETERMINISTIC CNN Test Setup Complete!")
    print("="*60)
    print("Files created:")
    print(f"  - {model_path} (Deterministic CNN PyTorch model)")
    print(f"  - proto_test_config_deterministic_cnn.json (Test configuration)")
    print("\nUsage:")
    print("  python -m nvflare.edge.web.routing_proxy 4321 lcp_map.json rootCA.pem proto_test_config_deterministic_cnn.json")
    print("\nDeterministic Features:")
    print("  - Fixed initial weights")
    print("  - Fixed learning rate (0.01)")
    print("  - Fixed batch size (2)")
    print("  - Fixed epochs (3)")
    print("  - Simplified architecture (1 conv + 1 linear)")
    print("  - Small input size (8x8)")
    print("  - Predictable tensor differences")
    print("\nThis ensures reproducible CNN testing!")

if __name__ == "__main__":
    main() 