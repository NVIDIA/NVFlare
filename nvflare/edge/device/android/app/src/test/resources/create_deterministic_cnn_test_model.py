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

class DeterministicCIFAR10CNN(nn.Module):
    """Deterministic CIFAR-10 CNN matching ExecutorTorch architecture with fixed weights for testing.
    Uses the same architecture as the ExecutorTorch CIFAR example.
    """
    def __init__(self):
        super().__init__()
        # CIFAR-10 CNN architecture matching ExecutorTorch implementation
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 32x32x3 -> 32x32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # 32x32x32 -> 16x16x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 16x16x32 -> 16x16x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # 16x16x64 -> 8x8x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 8x8x64 -> 8x8x128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # 8x8x128 -> 4x4x128
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),                 # 2048 -> 512
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10),                          # 512 -> 10 classes
        )
        
        # Set fixed weights for deterministic behavior
        with torch.no_grad():
            # Fixed conv1 weights (3 -> 32)
            self.features[0].weight.data.normal_(0.0, 0.1)
            self.features[0].bias.data.fill_(0.01)
            
            # Fixed conv2 weights (32 -> 64)
            self.features[3].weight.data.normal_(0.0, 0.1)
            self.features[3].bias.data.fill_(0.01)
            
            # Fixed conv3 weights (64 -> 128)
            self.features[6].weight.data.normal_(0.0, 0.1)
            self.features[6].bias.data.fill_(0.01)
            
            # Fixed linear1 weights (2048 -> 512)
            self.classifier[0].weight.data.normal_(0.0, 0.01)
            self.classifier[0].bias.data.fill_(0.01)
            
            # Fixed linear2 weights (512 -> 10)
            self.classifier[3].weight.data.normal_(0.0, 0.01)
            self.classifier[3].bias.data.fill_(0.01)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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
    """Create a deterministic CIFAR-10 CNN model with fixed weights."""
    print("Creating DETERMINISTIC CIFAR-10 CNN model...")
    print("Using fixed weights for reproducible results...")
    
    # Create the CIFAR-10 CNN with fixed weights
    cnn_net = DeterministicCIFAR10CNN()
    
    # Create the training wrapper
    training_net = DeterministicDeviceModel(cnn_net)
    training_net.train()
    
    # Set deterministic seed BEFORE generating any random data
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create FIXED example inputs for export (deterministic)
    # CIFAR-10 input: 2 samples, 3 channels, 32x32 images
    # Generate deterministic "random" input that will be exactly the same every time
    example_input = torch.randn(2, 3, 32, 32, dtype=torch.float32)
    
    # Normalize to [0, 1] range like CIFAR-10
    example_input = torch.clamp(example_input * 0.5 + 0.5, 0, 1)
    
    example_label = torch.tensor([0, 1], dtype=torch.long)
    
    print(f"Fixed example input shape: {example_input.shape}")
    print(f"Fixed example label shape: {example_label.shape}")
    
    # Export the model to ExecuTorch format
    print("Exporting deterministic CIFAR-10 CNN model to ExecuTorch format...")
    exported_program = export_model(training_net, example_input, example_label)
    
    # Save the model
    model_path = "deterministic_cifar10_cnn_test_model.pte"
    with open(model_path, "wb") as f:
        f.write(exported_program.buffer)
    
    print(f"Deterministic CIFAR-10 CNN model saved to {model_path}")
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
                            "features.0.weight": {
                                "sizes": [32, 3, 3, 3],
                                "strides": [27, 9, 3, 1],
                                "data": [0.001] * (32 * 3 * 3 * 3)  # Conv1 weights
                            },
                            "features.0.bias": {
                                "sizes": [32],
                                "strides": [1],
                                "data": [0.001] * 32
                            },
                            "features.3.weight": {
                                "sizes": [64, 32, 3, 3],
                                "strides": [288, 9, 3, 1],
                                "data": [0.001] * (64 * 32 * 3 * 3)  # Conv2 weights
                            },
                            "features.3.bias": {
                                "sizes": [64],
                                "strides": [1],
                                "data": [0.001] * 64
                            },
                            "features.6.weight": {
                                "sizes": [128, 64, 3, 3],
                                "strides": [576, 9, 3, 1],
                                "data": [0.001] * (128 * 64 * 3 * 3)  # Conv3 weights
                            },
                            "features.6.bias": {
                                "sizes": [128],
                                "strides": [1],
                                "data": [0.001] * 128
                            },
                            "classifier.0.weight": {
                                "sizes": [512, 2048],
                                "strides": [2048, 1],
                                "data": [0.001] * (512 * 2048)  # FC1 weights
                            },
                            "classifier.0.bias": {
                                "sizes": [512],
                                "strides": [1],
                                "data": [0.001] * 512
                            },
                            "classifier.3.weight": {
                                "sizes": [10, 512],
                                "strides": [512, 1],
                                "data": [0.001] * (10 * 512)  # FC2 weights
                            },
                            "classifier.3.bias": {
                                "sizes": [10],
                                "strides": [1],
                                "data": [0.001] * 10
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
    """Main function to create deterministic CIFAR-10 CNN test model and config."""
    print("="*60)
    print("Creating DETERMINISTIC CIFAR-10 CNN Test PyTorch Model")
    print("="*60)
    print("This model uses fixed weights and full CIFAR-10 CNN architecture")
    print("for reproducible CIFAR-10 testing results.")
    print()
    
    model_path = create_deterministic_cnn_model()
    
    print("\nConverting deterministic CIFAR-10 CNN model to base64...")
    model_base64 = convert_model_to_base64(model_path)
    
    print("\nCreating deterministic CIFAR-10 CNN test configuration...")
    create_deterministic_cnn_test_config(model_base64)
    
    print("\n" + "="*60)
    print("DETERMINISTIC CIFAR-10 CNN Test Setup Complete!")
    print("="*60)
    print("Files created:")
    print(f"  - {model_path} (Deterministic CIFAR-10 CNN PyTorch model)")
    print(f"  - proto_test_config_deterministic_cnn.json (Test configuration)")
    print("\nUsage:")
    print("  python -m nvflare.edge.web.routing_proxy 4321 lcp_map.json rootCA.pem proto_test_config_deterministic_cnn.json")
    print("\nDeterministic Features:")
    print("  - Fixed initial weights")
    print("  - Fixed learning rate (0.01)")
    print("  - Fixed batch size (2)")
    print("  - Fixed epochs (3)")
    print("  - Full CIFAR-10 CNN architecture (3 conv + 2 linear layers)")
    print("  - Standard CIFAR-10 input size (32x32x3)")
    print("  - Compatible with ExecutorTorch CIFAR example")
    print("  - Predictable tensor differences")
    print("\nThis ensures reproducible CIFAR-10 CNN testing!")

if __name__ == "__main__":
    main() 