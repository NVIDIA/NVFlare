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

from nvflare.edge.tools.et_recipe import (
    DeviceManagerConfig,
    ETRecipe,
    EvaluatorConfig,
    ModelManagerConfig,
    SimulationConfig,
)
from nvflare.recipe.simulation_env import SimEnv

parser = argparse.ArgumentParser()
parser.add_argument("--export_job", action="store_true")
parser.add_argument("--dataset", type=str, default="cifar10")
args = parser.parse_args()


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
    task_processor = XorETTaskProcessor()
    evaluator_config = None


recipe = ETRecipe(
    job_name=job_name,
    device_model=device_model,
    input_shape=input_shape,
    output_shape=output_shape,
    model_manager_config=ModelManagerConfig(
        # max_num_active_model_versions=1,
        max_model_version=3,
        update_timeout=1000.0,
        num_updates_for_model=5,
        # max_model_history=1,
    ),
    device_manager_config=DeviceManagerConfig(
        device_selection_size=5,
        min_hole_to_fill=5,
    ),
    evaluator_config=evaluator_config,
    simulation_config=SimulationConfig(
        task_processor=task_processor,
        num_devices=1,
    ),
    device_training_params={"epoch": 3, "lr": 0.0001},
)
if args.export_job:
    recipe.export(job_dir="/tmp/nvflare/workspaces/edge_example/prod_00/admin@nvidia.com/transfer")
else:
    env = SimEnv(num_clients=1)
    recipe.execute(env)
