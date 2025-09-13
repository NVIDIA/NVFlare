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

from nvflare.edge.tools.et_fed_buff_recipe import (
    DeviceManagerConfig,
    ETFedBuffRecipe,
    EvaluatorConfig,
    ModelManagerConfig,
    SimulationConfig,
)
from nvflare.recipe.prod_env import ProdEnv

parser = argparse.ArgumentParser()
parser.add_argument("--export_job", action="store_true")
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--workspace_dir", type=str, default="/tmp/nvflare/workspaces")
parser.add_argument("--project_name", type=str, default="edge_example")
parser.add_argument("--total_num_of_devices", type=int, default=4)
parser.add_argument("--num_of_simulated_devices_on_each_leaf", type=int, default=1)
args = parser.parse_args()

prod_dir = os.path.join(args.workspace_dir, args.project_name, "prod_00")
admin_startup_kit_dir = os.path.join(prod_dir, "admin@nvidia.com")
total_num_of_devices = args.total_num_of_devices
num_of_simulated_devices_on_each_leaf = args.num_of_simulated_devices_on_each_leaf

if args.dataset == "cifar10":
    from processors.cifar10_et_task_processor import Cifar10ETTaskProcessor
    from processors.models.cifar10_model import TrainingNet

    dataset_root = "/tmp/nvflare/cifar10"
    job_name = "cifar10_et"
    device_model = TrainingNet()
    batch_size = 4
    input_shape = (batch_size, 3, 32, 32)
    output_shape = (batch_size,)
    task_processor = Cifar10ETTaskProcessor(
        data_path=dataset_root,
        training_config={
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": 0,
        },
        subset_size=100,
    )
    evaluator_config = EvaluatorConfig(
        torchvision_dataset={"name": "CIFAR10", "path": dataset_root},
        eval_frequency=1,
    )
elif args.dataset == "xor":
    from processors.models.xor_model import TrainingNet
    from processors.xor_et_task_processor import XorETTaskProcessor

    job_name = "xor_et"
    device_model = TrainingNet()
    batch_size = 1
    input_shape = (batch_size, 2)
    output_shape = (batch_size,)
    task_processor = XorETTaskProcessor(
        training_config={
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": 0,
        },
    )
    evaluator_config = None


recipe = ETFedBuffRecipe(
    job_name=job_name,
    device_model=device_model,
    input_shape=input_shape,
    output_shape=output_shape,
    model_manager_config=ModelManagerConfig(
        # max_num_active_model_versions=1,
        max_model_version=3,
        update_timeout=1000,
        num_updates_for_model=total_num_of_devices,
        # max_model_history=1,
    ),
    device_manager_config=DeviceManagerConfig(
        device_selection_size=total_num_of_devices,
        min_hole_to_fill=total_num_of_devices,
    ),
    evaluator_config=evaluator_config,
    simulation_config=(
        SimulationConfig(
            task_processor=task_processor,
            num_devices=num_of_simulated_devices_on_each_leaf,
        )
        if num_of_simulated_devices_on_each_leaf > 0
        else None
    ),
    device_training_params={"epoch": 3, "lr": 0.0001, "batch_size": batch_size},
)
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
