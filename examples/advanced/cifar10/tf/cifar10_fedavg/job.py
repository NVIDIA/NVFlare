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
import multiprocessing

import tensorflow as tf
from model import ModerateTFNet
from data.cifar10_data_utils import cifar10_split

from nvflare import FedJob
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.tf.job_config.model import TFModel
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=8)
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--workspace", type=str, default="/tmp")
    parser.add_argument("--gpu", type=str, default="0")

    args = parser.parse_args()
    multiprocessing.set_start_method("spawn")

    train_script = "cifar10_fedavg/client.py"
    train_split_root = f"{args.workspace}/cifar10_splits/clients{args.n_clients}_alpha{args.alpha}"

    # Prepare data splits
    if args.alpha > 0.0:
        print(f"preparing CIFAR10 and doing alpha split with alpha = {args.alpha}")
        train_idx_paths = cifar10_split(num_sites=args.n_clients, alpha=args.alpha, split_dir=train_split_root)
        print(train_idx_paths)
    else:
        train_idx_paths = [None for __ in range(args.n_clients)]

    # Define job
    job = FedJob(name=f"cifar10_tf_fedavg_alpha{args.alpha}")

    # Define the FedAvg controller workflow and send to server
    controller = FedAvg(
        num_clients=args.n_clients,
        num_rounds=args.num_rounds,
    )
    job.to(controller, "server")

    # Define the initial global model and send to server
    job.to(TFModel(ModerateTFNet(input_shape=(None, 32, 32, 3))), "server")
    job.to(IntimeModelSelector(key_metric="accuracy"), "server")

    # Add clients
    task_script_args = f"--batch_size {args.batch_size} --epochs {args.epochs} --train_idx_root {train_split_root}"
    
    for i in range(args.n_clients):
        executor = ScriptRunner(
            script=train_script, 
            script_args=task_script_args, 
            framework=FrameworkType.TENSORFLOW
        )
        job.to(executor, f"site-{i + 1}")

    # Run the job using simulator
    job.simulator_run(f"{args.workspace}/nvflare/jobs/{job.name}", gpu=args.gpu)
