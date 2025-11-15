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

from client import Cifar10PTTaskProcessor
from model import Cifar10ConvNet

from nvflare.edge.tools.edge_fed_buff_recipe import (
    DeviceManagerConfig,
    EdgeFedBuffRecipe,
    EvaluatorConfig,
    ModelManagerConfig,
    SimulationConfig,
)
from nvflare.recipe.prod_env import ProdEnv


def create_edge_recipe(devices_per_leaf, num_leaf_nodes, global_rounds, no_delay=False):
    """
    Create an edge recipe for asynchronous federated learning with CIFAR10.

    Args:
        devices_per_leaf (int): Number of devices at each leaf node
        num_leaf_nodes (int): Number of leaf nodes in the hierarchy
        global_rounds (int): Number of global federated rounds (will be multiplied by total devices for async)
        no_delay (bool): If True, set communication delay and device speed to 0.0
    """
    dataset_root = "/tmp/nvflare/datasets/cifar10"
    # CIFAR10 dataset size is 50000, will use to calculate the data size for each device, with a minimum subset size of 100
    training_size = 50000
    min_subset_size = 100
    total_devices = devices_per_leaf * num_leaf_nodes
    subset_size = max(training_size // total_devices, min_subset_size)

    # Set communication delay and device speed based on no_delay flag
    if no_delay:
        communication_delay = {"mean": 0.0, "std": 0.0}
        device_speed = {"mean": [0.0], "std": [0.0]}
        suffix = "_no_delay"
    else:
        communication_delay = {"mean": 5.0, "std": 1.0}
        device_speed = {"mean": [100.0, 200.0, 400.0], "std": [1.0, 2.0, 4.0]}
        suffix = ""

    task_processor = Cifar10PTTaskProcessor(
        data_root=dataset_root,
        subset_size=subset_size,
        communication_delay=communication_delay,
        device_speed=device_speed,
    )

    # Configure model manager for asynchronous FL
    model_manager_config = ModelManagerConfig(
        global_lr=0.05,
        num_updates_for_model=1,
        # to be comparable to sync mode w.r.t. the data amount visited by global model,
        # sync - each global model covers total_devices data
        # async - each global model covers 1 device's data
        max_model_version=global_rounds * total_devices,
        # increase the update timeout to allow for the slowest device to finish
        update_timeout=500,
    )
    device_manager_config = DeviceManagerConfig(
        # each leaf node has devices_per_leaf devices
        device_selection_size=total_devices,
        # immediately dispatch the current global model
        # once receiving an update from device
        min_hole_to_fill=1,
        # always reuse the same devices for federated learning
        device_reuse=True,
    )
    eval_frequency = total_devices

    recipe = EdgeFedBuffRecipe(
        job_name=f"pt_job_async{suffix}",
        model=Cifar10ConvNet(),
        model_manager_config=model_manager_config,
        device_manager_config=device_manager_config,
        evaluator_config=EvaluatorConfig(
            torchvision_dataset={"name": "CIFAR10", "path": dataset_root}, eval_frequency=eval_frequency
        ),
        simulation_config=SimulationConfig(
            task_processor=task_processor,
            job_timeout=20.0,
            num_workers=4,
            # simulation config is for each leaf node
            num_devices=devices_per_leaf,
        ),
        custom_source_root=None,
    )

    return recipe


def main():
    parser = argparse.ArgumentParser(description="Create edge recipe for asynchronous federated learning")
    parser.add_argument("--devices_per_leaf", type=int, default=4, help="Number of devices on each leaf node")
    parser.add_argument("--num_leaf_nodes", type=int, default=4, help="Number of leaf nodes in the hierarchy")
    parser.add_argument(
        "--global_rounds",
        type=int,
        default=10,
        help="Number of global federated rounds (total global model versions will be this number multiplied by total devices)",
    )
    parser.add_argument("--workspace_dir", type=str, default="/tmp/nvflare/workspaces", help="Workspace directory")
    parser.add_argument(
        "--no_delay",
        action="store_true",
        help="If set, disable communication delay and device speed variations (set to 0.0)",
    )
    parser.add_argument(
        "--export_job",
        action="store_true",
        help="If set, export the recipe to the admin's transfer directory",
    )
    parser.add_argument("--project_name", type=str, default="edge_example", help="Project name")

    args = parser.parse_args()

    prod_dir = os.path.join(args.workspace_dir, args.project_name, "prod_00")
    admin_startup_kit_dir = os.path.join(prod_dir, "admin@nvidia.com")

    try:
        print("Creating asynchronous federated learning recipe...")

        recipe = create_edge_recipe(
            devices_per_leaf=args.devices_per_leaf,
            num_leaf_nodes=args.num_leaf_nodes,
            global_rounds=args.global_rounds,
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
        print("Result can be found in :", run.get_result())
        print("Job Status is:", run.get_status())
        print()


if __name__ == "__main__":
    exit(main())
