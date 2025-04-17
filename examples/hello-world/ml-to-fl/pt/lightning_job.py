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

from src.lit_net import LitNet

from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher
from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import BaseScriptRunner, FrameworkType


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--script", type=str, default="src/cifar10_lightning_fl.py")
    parser.add_argument("--key_metric", type=str, default="accuracy")
    parser.add_argument("--launch_external_process", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--export_config", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()


def main():
    # define local parameters
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    script = args.script
    key_metric = args.key_metric
    launch_external_process = args.launch_external_process
    export_config = args.export_config

    job = FedAvgJob(
        name="pt_lightning_client_api",
        n_clients=n_clients,
        num_rounds=num_rounds,
        key_metric=key_metric,
        initial_model=LitNet(),
    )

    for i in range(n_clients):
        executor = BaseScriptRunner(
            script=script,
            launch_external_process=launch_external_process,
            framework=FrameworkType.PYTORCH,
            # Adds a shutdown grace period to make sure after
            # flare.send the lightning script can finish predict and test and exit gracefully
            launcher=SubprocessLauncher(script=f"python3 -u custom/{script}", shutdown_timeout=100.0),
        )
        job.to(executor, f"site-{i + 1}")

    if export_config:
        job.export_job("/tmp/nvflare/jobs/job_config")
    else:
        job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")


if __name__ == "__main__":
    main()
