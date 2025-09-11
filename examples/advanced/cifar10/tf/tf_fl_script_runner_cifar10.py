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
from src.cifar10_data_split import cifar10_split
from src.tf_net import ModerateTFNet

from nvflare import FedJob
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_opt.tf.job_config.model import TFModel
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


CENTRALIZED_ALGO = "centralized"
FEDAVG_ALGO = "fedavg"
FEDOPT_ALGO = "fedopt"
SCAFFOLD_ALGO = "scaffold"
FEDPROX_ALGO = "fedprox"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--fedprox_mu",
        type=float,
        default=0.0,
    )
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
        "--workspace",
        type=str,
        default="/tmp",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
    )

    args = parser.parse_args()
    multiprocessing.set_start_method("spawn")

    supported_algos = (CENTRALIZED_ALGO, FEDAVG_ALGO, FEDOPT_ALGO, SCAFFOLD_ALGO, FEDPROX_ALGO)

    if args.algo not in supported_algos:
        raise ValueError(f"--algo should be one of: {supported_algos}, got: {args.algo}")

    train_script = "src/cifar10_tf_fl_alpha_split.py"
    train_split_root = (
        f"{args.workspace}/cifar10_splits/clients{args.n_clients}_alpha{args.alpha}"  # avoid overwriting results
    )

    # Prepare data splits
    if args.alpha > 0.0:

        # Do alpha splitting if alpha value > 0.0
        print(f"preparing CIFAR10 and doing alpha split with alpha = {args.alpha}")
        train_idx_paths = cifar10_split(num_sites=args.n_clients, alpha=args.alpha, split_dir=train_split_root)

        print(train_idx_paths)
    else:
        train_idx_paths = [None for __ in range(args.n_clients)]

    # Define job
    job = FedJob(name=f"cifar10_tf_{args.algo}_alpha{args.alpha}")

    # Define the controller workflow and send to server
    controller = None
    task_script_args = f"--batch_size {args.batch_size} --epochs {args.epochs}"

    if args.algo == FEDAVG_ALGO or args.algo == CENTRALIZED_ALGO:
        from nvflare.app_common.workflows.fedavg import FedAvg

        controller = FedAvg(
            num_clients=args.n_clients,
            num_rounds=args.num_rounds,
        )

    elif args.algo == FEDOPT_ALGO:
        from nvflare.app_opt.tf.fedopt_ctl import FedOpt

        controller = FedOpt(
            num_clients=args.n_clients,
            num_rounds=args.num_rounds,
        )
    elif args.algo == FEDPROX_ALGO:
        from nvflare.app_common.workflows.fedavg import FedAvg

        controller = FedAvg(
            num_clients=args.n_clients,
            num_rounds=args.num_rounds,
        )
        task_script_args += f" --fedprox_mu {args.fedprox_mu}"

    elif args.algo == SCAFFOLD_ALGO:
        train_script = "src/cifar10_tf_fl_alpha_split_scaffold.py"
        from nvflare.app_common.workflows.scaffold import Scaffold

        controller = Scaffold(
            num_clients=args.n_clients,
            num_rounds=args.num_rounds,
        )

    job.to(controller, "server")

    # Define the initial global model and send to server
    job.to(TFModel(ModerateTFNet(input_shape=(None, 32, 32, 3))), "server")

    job.to(IntimeModelSelector(key_metric="accuracy"), "server")

    # Add clients
    for i, train_idx_path in enumerate(train_idx_paths):
        curr_task_script_args = task_script_args + f" --train_idx_path {train_idx_path}"
        executor = ScriptRunner(
            script=train_script, script_args=curr_task_script_args, framework=FrameworkType.TENSORFLOW
        )
        job.to(executor, f"site-{i + 1}")

    # Can export current job to folder.
    # job.export_job(f"{args.workspace}/nvflare/jobs/job_config")

    # Here we launch the job using simulator.
    job.simulator_run(f"{args.workspace}/nvflare/jobs/{job.name}", gpu=args.gpu)
