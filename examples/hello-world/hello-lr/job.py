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

from nvflare.app_common.workflows.lr.fedavg import FedAvgLR
from nvflare.client.config import ExchangeFormat
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=5)

    return parser.parse_args()


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds

    print("n_clients=", n_clients)

    # Create FedJob.
    job = FedJob(name="fed_avg_lr")

    # Send custom controller to server
    controller = FedAvgLR(num_clients=n_clients, num_rounds=num_rounds, damping_factor=0.8)
    job.to(controller, "server")

    # Add clients
    for i in range(n_clients):
        runner = ScriptRunner(
            script="client.py",
            script_args="--data_root /tmp/flare/dataset/heart_disease_data",
            # launch_external_process=True,
            framework=FrameworkType.RAW,
            server_expected_format=ExchangeFormat.RAW,
        )
        job.to(runner, f"site-{i + 1}")

    job.export_job("/tmp/nvflare/jobs/job_config")
    print("running simulator")
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")


if __name__ == "__main__":
    main()
