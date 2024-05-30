# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os

from src.tf_net import TFNet
from src.cifar10_data_split import cifar10_split

from nvflare import FedAvg, FedJob, ScriptExecutor


if __name__ == "__main__":
    n_clients = 2
    num_rounds = 2
    alpha = 0.1
    train_split_root = f"/tmp/cifar10_splits/clients{n_clients}_alpha{alpha}"  # avoid overwriting results
    train_script = "src/cifar10_tf_fl_alpha_split.py"

    # Prepare data splits
    train_idx_paths = cifar10_split(num_sites=n_clients, alpha=alpha, split_dir=train_split_root)

    # Define job
    job = FedJob(name=f"cifar10_tf_fedavg_alpha{alpha}")

    # Define the controller workflow and send to server
    controller = FedAvg(
        min_clients=n_clients,
        num_rounds=num_rounds,
    )
    job.to(controller, "server")

    # Define the initial global model and send to server
    job.to(TFNet(input_shape=(None, 32, 32, 3)), "server")

    # Add clients
    for i, train_idx_path in enumerate(train_idx_paths):
        executor = ScriptExecutor(
            task_script_path=train_script, task_script_args=f"--batch_size 128 --train_idx_path {train_idx_path}"
        )
        job.to(executor, f"site-{i}", gpu=0)

    # job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir")
