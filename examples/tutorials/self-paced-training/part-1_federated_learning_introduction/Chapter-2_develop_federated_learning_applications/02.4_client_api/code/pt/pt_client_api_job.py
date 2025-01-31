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

from src.lit_net import LitNet
from src.net import Net

from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--script", type=str, default="src/cifar10_fl.py")
    parser.add_argument("--key_metric", type=str, default="accuracy")
    parser.add_argument("--launch_process", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--launch_command", type=str, default="python3 -u")
    parser.add_argument("--ports", type=str, default="7777,8888")
    parser.add_argument("--export_config", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()


def main():
    # define local parameters
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    script = args.script
    key_metric = args.key_metric
    launch_process = args.launch_process
    launch_command = args.launch_command
    ports = args.ports.split(",")
    export_config = args.export_config

    job = FedAvgJob(
        name="pt_client_api",
        n_clients=n_clients,
        num_rounds=num_rounds,
        key_metric=key_metric,
        initial_model=LitNet() if "lightning" in script else Net(),
    )

    for i in range(n_clients):
        executor = ScriptRunner(
            script=script,
            launch_external_process=launch_process,
            command=launch_command.replace("{PORT}", ports[i]),
            framework=FrameworkType.PYTORCH,
        )
        job.to(executor, f"site-{i + 1}")

    if export_config:
        job.export_job("/tmp/nvflare/jobs/job_config")
    else:
        job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")


if __name__ == "__main__":
    main()
