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
Script to create a test PyTorch model for Android/iOS testing.
This generates a simple CNN model and converts it to base64 format.
"""

import torch
import torch.nn as nn
import base64
import json
import os

class SimpleCNN(nn.Module):
    """Simple CNN model for testing."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_test_model():
    """Create a simple test model and save it."""
    model = SimpleCNN()
    model.eval()
    
    # Create a dummy input to trace the model
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save the model
    model_path = "test_model.pt"
    traced_model.save(model_path)
    
    print(f"Model saved to {model_path}")
    return model_path

def convert_model_to_base64(model_path):
    """Convert model file to base64 string."""
    with open(model_path, 'rb') as f:
        model_bytes = f.read()
    
    model_base64 = base64.b64encode(model_bytes).decode('utf-8')
    print(f"Model converted to base64 (length: {len(model_base64)})")
    return model_base64

def update_test_config(model_base64):
    """Update the test configuration with the real model."""
    config_path = "proto_test_config_with_model.json"
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Warning: {config_path} not found. Creating a basic config...")
        # Create a basic config structure
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
                                            "epoch": 5,
                                            "lr": 0.0001
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
                                "learning_rate": 0.0001,
                                "batch_size": 32,
                                "epochs": 5,
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
                                    "sizes": [32, 3, 3, 3],
                                    "strides": [27, 9, 3, 1],
                                    "data": [0.001, -0.002, 0.003, 0.004, 0.005]
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
    else:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading {config_path}: {e}")
            print("Creating a new config file...")
            return update_test_config(model_base64)  # Recursive call to create new config
    
    # Update the task data with the real model
    for step in config['steps']:
        if 'response' in step and 'task_data' in step['response']:
            if step['response']['task_data']['kind'] == 'model':
                step['response']['task_data']['data']['model_buffer'] = model_base64
                break
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Updated {config_path} with real model data")

def main():
    """Main function to create test model and update config."""
    print("Creating test PyTorch model...")
    model_path = create_test_model()
    
    print("Converting model to base64...")
    model_base64 = convert_model_to_base64(model_path)
    
    print("Updating test configuration...")
    update_test_config(model_base64)
    
    print("Done! You can now use proto_test_config_with_model.json for testing.")
    print("The model is a simple CNN compatible with CIFAR-10 dataset.")

if __name__ == "__main__":
    main() 