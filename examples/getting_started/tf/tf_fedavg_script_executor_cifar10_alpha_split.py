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

import argparse

from src.tf_net import ModerateTFNet
from src.cifar10_data_split import cifar10_split

from nvflare import FedAvg, FedJob, ScriptExecutor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_clients",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
    )       
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )    
    args = parser.parse_args()
    
    train_script = "src/cifar10_tf_fl_alpha_split.py"
    train_split_root = f"/tmp/cifar10_splits/clients{args.n_clients}_alpha{args.alpha}"  # avoid overwriting results

    # Prepare data splits
    train_idx_paths = cifar10_split(num_sites=args.n_clients, alpha=args.alpha, split_dir=train_split_root)

    # Define job
    job = FedJob(name=f"cifar10_tf_fedavg_alpha{args.alpha}")

    # Define the controller workflow and send to server
    controller = FedAvg(
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
    )
    job.to(controller, "server")

    # Define the initial global model and send to server
    job.to(ModerateTFNet(input_shape=(None, 32, 32, 3)), "server")

    # Add clients
    for i, train_idx_path in enumerate(train_idx_paths):
        executor = ScriptExecutor(
            task_script_path=train_script, task_script_args=f"--batch_size {args.batch_size} --epochs {args.epochs} --train_idx_path {train_idx_path}"
        )
        job.to(executor, f"site-{i+1}", gpu=args.gpu)

    # job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run(f"/tmp/nvflare/jobs/{job.job_name}_3")
