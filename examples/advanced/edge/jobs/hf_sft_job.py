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

import argparse
import os

import torch
from processors.hf_sft_task_processor import HFSFTTaskProcessor
from transformers import AutoModelForCausalLM

from nvflare.edge.tools.edge_fed_buff_recipe import (
    DeviceManagerConfig,
    EdgeFedBuffRecipe,
    ModelManagerConfig,
    SimulationConfig,
)
from nvflare.recipe.prod_env import ProdEnv


def load_hf_model(model_name_or_path: str):
    """Load the actual HuggingFace model for EdgeFedBuffRecipe.

    Args:
        model_name_or_path (str): HuggingFace model name or path

    Returns:
        torch.nn.Module: The loaded HuggingFace model
    """
    print(f"Loading HuggingFace model: {model_name_or_path}")

    # Load model with appropriate settings for federated learning
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="cpu",  # Load on CPU first for recipe creation
            use_cache=False,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,  # Allow custom model code if needed
        )
        model.config.pretraining_tp = 1

        # Move to CPU and set to eval mode for recipe initialization
        model = model.cpu()
        model.eval()

        print(f"Successfully loaded model with {sum(p.numel() for p in model.parameters())} parameters")

    except Exception as e:
        print(f"Error loading model {model_name_or_path}: {e}")
        raise
    finally:
        torch.set_default_dtype(default_dtype)

    return model


def create_hf_sft_recipe(
    model_name_or_path,
    data_path_train,
    data_path_valid,
    output_path,
    devices_per_leaf,
    num_leaf_nodes,
    global_rounds,
    subset_size=None,
    local_epochs=1,
    local_batch_size=4,
    local_lr=5e-4,
    lr_scheduler="constant",
    no_delay=False,
):
    """
    Create an HuggingFace SFT edge recipe for federated learning.
    Uses synchronous federated learning mode only.

    Args:
        model_name_or_path (str): HuggingFace model name or path
        data_path_train (str): Path to training data
        data_path_valid (str): Path to validation data
        output_path (str): Output directory for model checkpoints
        devices_per_leaf (int): Number of devices at each leaf node
        num_leaf_nodes (int): Number of leaf nodes in the hierarchy
        global_rounds (int): Number of global federated rounds
        subset_size (int): Size of data subset for each device (None for full dataset)
        local_epochs (int): Number of local training epochs per round
        batch_size (int): Training batch size
        gradient_accumulation_steps (int): Gradient accumulation steps
        learning_rate (float): Learning rate for training
        lr_scheduler (str): Learning rate scheduler type
        no_delay (bool): If True, set communication delay and device speed to 0.0
    """
    total_devices = devices_per_leaf * num_leaf_nodes

    # Set communication delay and device speed based on no_delay flag
    if no_delay:
        communication_delay = {"mean": 0.0, "std": 0.0}
        device_speed = {"mean": [0.0], "std": [0.0]}
        suffix = "_no_delay"
    else:
        # Adjust delays for longer HF training times
        communication_delay = {"mean": 10.0, "std": 2.0}
        device_speed = {"mean": [300.0, 600.0, 1200.0], "std": [30.0, 60.0, 120.0]}
        suffix = ""

    # Create the HF SFT task processor
    task_processor = HFSFTTaskProcessor(
        model_name_or_path=model_name_or_path,
        data_path_train=data_path_train,
        data_path_valid=data_path_valid,
        output_path=output_path,
        communication_delay=communication_delay,
        device_speed=device_speed,
        subset_size=subset_size,
        local_epochs=local_epochs,
        local_batch_size=local_batch_size,
        local_lr=local_lr,
        lr_scheduler=lr_scheduler,
    )

    # Configure model manager for synchronous FL
    model_manager_config = ModelManagerConfig(
        global_lr=1.0,  # Use simple averaging for SFT
        # Need all devices to train for one global model version
        num_updates_for_model=total_devices,
        max_model_version=global_rounds,
        update_timeout=1800,  # Longer timeout for HF training (30 minutes)
    )

    # Configure device manager for synchronous FL
    device_manager_config = DeviceManagerConfig(
        # Each leaf node has devices_per_leaf devices
        device_selection_size=total_devices,
        # Wait for all devices to finish training before starting
        # dispatching the next global model version (synchronous)
        min_hole_to_fill=total_devices,
        # Always reuse the same devices for federated learning
        device_reuse=True,
    )

    # Load the actual HuggingFace model for recipe initialization
    hf_model = load_hf_model(model_name_or_path)

    # Create the recipe
    recipe = EdgeFedBuffRecipe(
        job_name=f"hf_sft_job_sync{suffix}",
        model=hf_model,
        model_manager_config=model_manager_config,
        device_manager_config=device_manager_config,
        evaluator_config=None,  # No built-in evaluator for HF models
        simulation_config=SimulationConfig(
            task_processor=task_processor,
            job_timeout=7200.0,  # 2 hour timeout for HF training (increased)
            num_workers=2,  # Reduced workers to avoid resource conflicts
            # Simulation config is for each leaf node
            num_devices=devices_per_leaf,
        ),
        custom_source_root=None,
    )

    return recipe


def main():
    parser = argparse.ArgumentParser(description="Create HuggingFace SFT edge recipe for federated learning")
    parser.add_argument(
        "--model_name_or_path", type=str, default="facebook/opt-125m", help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--data_path_train", type=str, default="./dataset/dolly/training.jsonl", help="Path to training data"
    )
    parser.add_argument(
        "--data_path_valid", type=str, default="./dataset/dolly/validation.jsonl", help="Path to validation data"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./workspace_federated/llama-3.2-1b-dolly-sft",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--subset_size", type=int, default=None, help="Size of data subset for each device (None for full dataset)"
    )
    parser.add_argument("--devices_per_leaf", type=int, default=1, help="Number of devices on each leaf node")
    parser.add_argument("--num_leaf_nodes", type=int, default=4, help="Number of leaf nodes in the hierarchy")
    parser.add_argument("--global_rounds", type=int, default=3, help="Number of global federated rounds")
    parser.add_argument("--local_epochs", type=int, default=1, help="Number of local training epochs per round")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=10, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for training")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="Learning rate scheduler type")
    parser.add_argument("--workspace_dir", type=str, default="/tmp/nvflare/workspaces", help="Workspace directory")
    parser.add_argument(
        "--no_delay",
        action="store_true",
        help="If set, disable communication delay and device speed variations (set to 0.0)",
    )
    parser.add_argument(
        "--export_job", action="store_true", help="If set, export the recipe to the admin's transfer directory"
    )
    parser.add_argument("--project_name", type=str, default="edge_example", help="Project name")

    args = parser.parse_args()

    prod_dir = os.path.join(args.workspace_dir, args.project_name, "prod_00")
    admin_startup_kit_dir = os.path.join(prod_dir, "admin@nvidia.com")

    try:
        print("Creating HuggingFace SFT federated learning recipe...")

        # If subset_size is not specified, calculate a reasonable default
        # to ensure each device gets a portion of the dataset
        if args.subset_size is None:
            total_devices = args.devices_per_leaf * args.num_leaf_nodes
            print(
                f"No subset size specified. Consider setting --subset_size to distribute data across {total_devices} devices"
            )
            print("Example: For a 15000-sample dataset with 4 devices, use --subset_size 3750")

        recipe = create_hf_sft_recipe(
            model_name_or_path=args.model_name_or_path,
            data_path_train=args.data_path_train,
            data_path_valid=args.data_path_valid,
            output_path=args.output_path,
            devices_per_leaf=args.devices_per_leaf,
            num_leaf_nodes=args.num_leaf_nodes,
            global_rounds=args.global_rounds,
            subset_size=args.subset_size,
            local_epochs=args.local_epochs,
            local_batch_size=args.batch_size,
            local_lr=args.learning_rate,
            lr_scheduler=args.lr_scheduler,
            no_delay=args.no_delay,
        )

    except Exception as e:
        print(f"Error creating recipe: {e}")
        return 1

    if args.export_job:
        output_dir = os.path.join(admin_startup_kit_dir, "transfer")
        print(f"Exporting recipe to {output_dir}")
        recipe.export(job_dir=output_dir)
    else:
        env = ProdEnv(startup_kit_location=admin_startup_kit_dir, username="admin@nvidia.com")
        run = recipe.execute(env)
        print()
        print("Result can be found in:", run.get_result())
        print("Job Status is:", run.get_status())
        print()


if __name__ == "__main__":
    exit(main())
