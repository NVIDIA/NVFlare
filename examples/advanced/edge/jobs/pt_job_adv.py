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

from processors.cifar10_pt_task_processor import Cifar10PTTaskProcessor
from processors.models.cifar10_model import Cifar10ConvNet

from nvflare.edge.tools.edge_fed_buff_recipe import (
    DeviceManagerConfig,
    EdgeFedBuffRecipe,
    EvaluatorConfig,
    ModelManagerConfig,
    SimulationConfig,
)
from nvflare.recipe.prod_env import ProdEnv


def main():
    # FL global and local parameters
    devices_per_leaf = 10000
    device_selection_size = 200
    num_leaf_nodes = 4
    startup_kit_location = "/tmp/nvflare/workspaces/edge_example/prod_00/admin@nvidia.com"
    username = "admin@nvidia.com"
    output_dir = "/tmp/nvflare/workspaces/edge_example/prod_00/admin@nvidia.com/transfer"
    dataset_root = "/tmp/nvflare/datasets/cifar10"
    subset_size = 100
    communication_delay = {"mean": 0.0, "std": 0.0}
    device_speed = {"mean": [0.0], "std": [0.0]}
    global_lr = 0.1
    num_updates_for_model = 20
    max_model_version = 200
    max_model_history = None
    min_hole_to_fill = 10

    eval_frequency = 1
    local_batch_size = 10
    local_epochs = 4
    local_lr = 0.1
    local_momentum = 0.0

    print("Creating federated learning recipe...")
    # Task processor for device training simulation
    task_processor = Cifar10PTTaskProcessor(
        data_root=dataset_root,
        subset_size=subset_size,
        communication_delay=communication_delay,
        device_speed=device_speed,
        local_batch_size=local_batch_size,
        local_epochs=local_epochs,
        local_lr=local_lr,
        local_momentum=local_momentum,
    )

    # Model manager and device manager configurations
    model_manager_config = ModelManagerConfig(
        global_lr=global_lr,
        num_updates_for_model=num_updates_for_model,
        max_model_version=max_model_version,
        max_model_history=max_model_history,
        max_num_active_model_versions=max_model_history,
        update_timeout=500,
    )
    device_manager_config = DeviceManagerConfig(
        device_selection_size=device_selection_size,
        # wait for all clients report to server before starting
        initial_min_client_num=num_leaf_nodes,
        min_hole_to_fill=min_hole_to_fill,
        device_reuse=False,
    )
    eval_frequency = eval_frequency

    # Generate recipe
    recipe = EdgeFedBuffRecipe(
        job_name="pt_job_adv",
        model=Cifar10ConvNet(),
        model_manager_config=model_manager_config,
        device_manager_config=device_manager_config,
        evaluator_config=EvaluatorConfig(
            torchvision_dataset={"name": "CIFAR10", "path": dataset_root}, eval_frequency=eval_frequency
        ),
        simulation_config=SimulationConfig(
            task_processor=task_processor,
            job_timeout=20.0,
            num_workers=device_selection_size // num_leaf_nodes + 1,
            num_devices=devices_per_leaf,
        ),
        custom_source_root=None,
    )

    print(f"Exporting recipe to {output_dir}")
    recipe.export(output_dir)
    print("DONE")

    env = ProdEnv(startup_kit_location=startup_kit_location, username=username)
    run = recipe.execute(env)
    print()
    print("Result can be found in :", run.get_result())
    print("Job Status is:", run.get_status())
    print()


if __name__ == "__main__":
    exit(main())
