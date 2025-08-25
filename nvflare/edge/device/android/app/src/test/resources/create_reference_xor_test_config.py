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
Script to create a test configuration using the reference XOR model files.
This uses the known-working XOR models from the reference implementation
to test the Android app without model export issues.
"""

import base64
import json
import os
import shutil


def copy_reference_xor_models():
    """Copy the reference XOR model files to the current directory."""
    reference_dir = "/Users/kevlu/workspace/repos/NVFlare/nvflare/edge/device/reference_android_executorch_training/executorch_android/src/androidTest/resources"

    # Source files
    xor_pte_src = os.path.join(reference_dir, "xor.pte")
    xor_ptd_src = os.path.join(reference_dir, "xor.ptd")
    xor_full_src = os.path.join(reference_dir, "xor_full.pte")

    # Destination files
    xor_pte_dst = "reference_xor.pte"
    xor_ptd_dst = "reference_xor.ptd"
    xor_full_dst = "reference_xor_full.pte"

    print("Copying reference XOR model files...")

    # Copy the files
    shutil.copy2(xor_pte_src, xor_pte_dst)
    shutil.copy2(xor_ptd_src, xor_ptd_dst)
    shutil.copy2(xor_full_src, xor_full_dst)

    print(f"Copied {xor_pte_src} -> {xor_pte_dst}")
    print(f"Copied {xor_ptd_src} -> {xor_ptd_dst}")
    print(f"Copied {xor_full_src} -> {xor_full_dst}")

    return xor_pte_dst, xor_ptd_dst, xor_full_dst


def convert_model_to_base64(model_path):
    """Convert model file to base64 string."""
    with open(model_path, "rb") as f:
        model_bytes = f.read()

    model_base64 = base64.b64encode(model_bytes).decode("utf-8")
    print(f"Model {model_path} converted to base64 (length: {len(model_base64)})")
    return model_base64


def create_reference_test_config():
    """Create a test configuration using the reference XOR model."""
    print("Creating test configuration with reference XOR model...")

    # Copy reference models
    xor_pte, xor_ptd, xor_full = copy_reference_xor_models()

    # Convert the full model to base64 (single file approach)
    model_base64 = convert_model_to_base64(xor_full)

    # Create test configuration
    config_path = "proto_test_config_reference_xor.json"

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
                                    "args": {"epoch": 5, "lr": 0.1, "method": "xor"},
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
                        "meta": {"learning_rate": 0.1, "batch_size": 4, "epochs": 5, "method": "xor"},
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
                    "result": {"kind": "number", "data": {"value": 0.693147, "count": 20}},
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

    print(f"Created {config_path} with reference XOR model data")
    print(f"Model file: {xor_full}")
    print(f"Model size: {os.path.getsize(xor_full)} bytes")
    print(f"Base64 length: {len(model_base64)}")


def main():
    """Main function to create reference XOR test configuration."""
    print("=" * 60)
    print("Creating Reference XOR Test Configuration")
    print("=" * 60)
    print("This uses the known-working XOR models from the reference")
    print("implementation to test the Android app.")
    print()

    create_reference_test_config()

    print("\n" + "=" * 60)
    print("Reference XOR Test Setup Complete!")
    print("=" * 60)
    print("Files created:")
    print("  - reference_xor.pte (Reference XOR model)")
    print("  - reference_xor.ptd (Reference XOR tensor data)")
    print("  - reference_xor_full.pte (Reference XOR combined model)")
    print("  - proto_test_config_reference_xor.json (Test configuration)")
    print("\nUsage:")
    print(
        "  python -m nvflare.edge.web.routing_proxy 4321 lcp_map.json rootCA.pem proto_test_config_reference_xor.json"
    )
    print("\nThis uses the exact same XOR models that work in the reference")
    print("implementation, so the Android app should work correctly!")


if __name__ == "__main__":
    main()
