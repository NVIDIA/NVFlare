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

from torch_geometric.nn import GraphSAGE

from nvflare import FedJob
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.job_config.script_runner import ScriptRunner


def main():
    args = define_parser()

    site_names = args.sites
    n_clients = len(site_names)

    work_dir = args.work_dir
    task_script_path = args.task_script_path
    task_script_args = args.task_script_args

    job = FedJob(name="gnn_train_encode_job")

    # Define the controller workflow and send to server
    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=args.num_rounds,
    )
    job.to(controller, "server")

    # Define the model
    model = GraphSAGE(
        in_channels=10,
        hidden_channels=64,
        num_layers=2,
        out_channels=64,
    )
    job.to(PTModel(model), "server")

    # Add clients
    for site_name in site_names:
        executor = ScriptRunner(script=task_script_path, script_args=task_script_args)
        job.to(executor, site_name)

    if work_dir:
        print(f"{work_dir=}")
        job.export_job(work_dir)

    if not args.config_only:
        job.simulator_run(work_dir)


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--sites",
        nargs="*",  # 0 or more values expected => creates a list
        type=str,
        default=[],  # default if nothing is provided
        help="Space separated site names",
    )
    parser.add_argument(
        "-n",
        "--num_rounds",
        type=int,
        default=100,
        help="number of FL rounds",
    )
    parser.add_argument(
        "-w",
        "--work_dir",
        type=str,
        nargs="?",
        default="/tmp/nvflare/jobs/xgb/workdir",
        help="work directory, default to '/tmp/nvflare/jobs/xgb/workdir'",
    )

    parser.add_argument(
        "-p",
        "--task_script_path",
        type=str,
        nargs="?",
        help="task script",
    )

    parser.add_argument(
        "-a",
        "--task_script_args",
        type=str,
        nargs="?",
        default="",
        help="",
    )

    parser.add_argument("-co", "--config_only", action="store_true", help="config only mode, will not run simulator")

    return parser.parse_args()


if __name__ == "__main__":
    main()
