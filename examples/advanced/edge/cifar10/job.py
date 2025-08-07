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

from et_task_processor import Cifar10ETTaskProcessor
from model import TrainingNet

from nvflare.edge.tools.et_recipe import (
    DeviceManagerConfig,
    ETRecipe,
    EvaluatorConfig,
    ModelManagerConfig,
    SimulationConfig,
)
from nvflare.recipe.simulation_env import SimulationExecEnv

dataset_root = "/tmp/nvflare/cifar10"
BATCH_SIZE = 4

parser = argparse.ArgumentParser()
parser.add_argument("--export_job", action="store_true")
args = parser.parse_args()

recipe = ETRecipe(
    job_name="cifar10_et",
    device_model=TrainingNet(),
    input_shape=(BATCH_SIZE, 3, 32, 32),
    output_shape=(BATCH_SIZE,),
    model_manager_config=ModelManagerConfig(
        # max_num_active_model_versions=1,
        max_model_version=1,
        update_timeout=1000.0,
        num_updates_for_model=5,
        # max_model_history=1,
    ),
    device_manager_config=DeviceManagerConfig(
        device_selection_size=5,
        min_hole_to_fill=5,
    ),
    evaluator_config=EvaluatorConfig(
        torchvision_dataset={"name": "CIFAR10", "path": dataset_root},
        eval_frequency=1,
    ),
    simulation_config=SimulationConfig(
        task_processor=Cifar10ETTaskProcessor(
            data_path=dataset_root,
            training_config={
                "batch_size": BATCH_SIZE,
                "shuffle": True,
                "num_workers": 0,
            },
            subset_size=100,
        ),
        num_devices=5,
    ),
    device_training_params={"epoch": 3, "lr": 0.0001},
)
if args.export_job:
    recipe.export(job_dir="./job")
else:
    env = SimulationExecEnv(num_clients=1)
    recipe.execute(env)
