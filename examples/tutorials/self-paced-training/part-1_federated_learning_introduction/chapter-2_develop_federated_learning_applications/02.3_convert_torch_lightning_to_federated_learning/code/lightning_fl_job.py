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

from src.lit_net import LitNet

from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner

if __name__ == "__main__":
    n_clients = 5
    num_rounds = 2

    job = BaseFedJob(
        name="cifar10_lightning_fedavg",
        initial_model=LitNet(),
    )

    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
    )
    job.to(controller, "server")

    # Add clients
    for i in range(n_clients):
        runner = ScriptRunner(
            script="src/cifar10_lightning_fl.py", script_args=""  # f"--batch_size 32 --data_path /tmp/data/site-{i}"
        )
        job.to(runner, f"site-{i + 1}")

    job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
