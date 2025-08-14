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

from processors.cifar10_pt_task_processor import Cifar10PTTaskProcessor
from processors.models.cifar10_model import Cifar10ConvNet

from nvflare.edge.tools.edge_recipe import (
    DeviceManagerConfig,
    EdgeRecipe,
    EvaluatorConfig,
    ModelManagerConfig,
    SimulationConfig,
)


def create_edge_recipe(fl_mode, devices_per_leaf, num_leaf_nodes, global_rounds, no_delay=False):
    """
    Create an edge recipe based on the specified federated learning mode and parameters.
    Supports both sync and async modes - both modes use the basic setting:
    - Sync assumes all devices participate in each global model version
    - Async assumes generating a new global model version and immediately dispatch it once receiving an update from device

    Args:
        fl_mode (str): Either 'sync' or 'async'
        devices_per_leaf (int): Number of devices at each leaf node
        num_leaf_nodes (int): Number of leaf nodes in the hierarchy
        global_rounds (int): Number of global model versions, i.e., number of federated rounds for sync mode
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

    # Configure model manager based on FL mode
    if fl_mode == "sync":
        model_manager_config = ModelManagerConfig(
            global_lr=1.0,
            # need all devices to train for one global model version
            num_updates_for_model=total_devices,
            max_model_version=global_rounds,
            # basic synchronous mode, no need to discard old model updates
            max_model_history=1,
        )
        device_manager_config = DeviceManagerConfig(
            # each leaf node has devices_per_leaf devices
            device_selection_size=total_devices,
            # wait for all devices to finish training before starting
            # dispatching the next global model version
            min_hole_to_fill=total_devices,
            # always reuse the same devices for federated learning
            device_reuse=True,
        )
        eval_frequency = 1
    else:  # async mode
        model_manager_config = ModelManagerConfig(
            global_lr=0.05,
            num_updates_for_model=1,
            # to be comparable to sync mode w.r.t. the data amount visited by global model,
            # sync - each global model covers total_devices data
            # async - each global model covers 1 device's data
            max_model_version=global_rounds * total_devices,
            # basic async mode, set max model update version diff so that
            # the updater will not discard old model updates
            # since the fastest device is 4 times faster than the slowest device,
            # worst case is that there is only 1 slowest device and (total_devices - 1) fastest devices,
            # to ensure that the updater will not discard old model updates,
            # we need to allow (total_devices - 1) * 4 model updates
            # on server side:
            max_model_history=(total_devices - 1) * 4,
            # on client/updater side:
            max_num_active_model_versions=(total_devices - 1) * 4,
            # increase the update timeout to allow for the slowest device to finish
            update_timeout=500.0,
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

    recipe = EdgeRecipe(
        job_name=f"pt_job_{fl_mode}{suffix}",
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
    parser = argparse.ArgumentParser(description="Create edge recipe for federated learning")
    parser.add_argument(
        "--fl_mode", type=str, choices=["sync", "async"], required=True, help="Federated learning mode: sync or async"
    )
    parser.add_argument("--devices_per_leaf", type=int, default=4, help="Number of devices on each leaf node")
    parser.add_argument("--num_leaf_nodes", type=int, default=4, help="Number of leaf nodes in the hierarchy")
    parser.add_argument(
        "--global_rounds",
        type=int,
        default=10,
        help="Number of global federated rounds under sync mode, total globalmodel version under async mode will multiply this number by the total number of devices",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/nvflare/workspaces/edge_example/prod_00/admin@nvidia.com/transfer",
        help="Output directory for the recipe",
    )
    parser.add_argument(
        "--no_delay",
        action="store_true",
        help="If set, disable communication delay and device speed variations (set to 0.0)",
    )

    args = parser.parse_args()

    try:
        print(f"Creating {args.fl_mode} federated learning recipe...")

        recipe = create_edge_recipe(
            fl_mode=args.fl_mode,
            devices_per_leaf=args.devices_per_leaf,
            num_leaf_nodes=args.num_leaf_nodes,
            global_rounds=args.global_rounds,
            no_delay=args.no_delay,
        )

        print("Exporting recipe...")
        recipe.export(args.output_dir)
        print("DONE")

    except Exception as e:
        print(f"Error creating recipe: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
