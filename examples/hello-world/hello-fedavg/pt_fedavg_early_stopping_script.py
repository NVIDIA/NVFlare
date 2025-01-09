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

from src.net import Net

from nvflare import FedJob
from nvflare.app_opt.pt.fedavg_early_stopping import PTFedAvgEarlyStopping
from nvflare.job_config.script_runner import ScriptRunner

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 5
    train_script = "src/cifar10_fl.py"

    job = FedJob(name="cifar10_fedavg_early_stopping")

    # Define the controller workflow and send to server
    controller = PTFedAvgEarlyStopping(
        num_clients=n_clients,
        num_rounds=num_rounds,
        stop_cond="accuracy >= 40",
        initial_model=Net(),
    )
    job.to(controller, "server")

    # Add clients
    for i in range(n_clients):
        executor = ScriptRunner(script=train_script, script_args="")
        job.to(executor, f"site-{i}")

    # job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
