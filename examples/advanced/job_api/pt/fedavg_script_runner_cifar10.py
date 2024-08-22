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

from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.model import PTModel

# from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import ScriptRunner

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 2
    train_script = "src/cifar10_fl.py"

    job = FedJob(name="cifar10_fedavg")

    # Define the controller workflow and send to server
    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
    )
    job.to(controller, "server")
    # job.to_server(controller)

    # Define the initial global model and send to server
    job.to(PTModel(Net()), "server")

    job.to(IntimeModelSelector(key_metric="accuracy"), "server")

    # Note: We can optionally replace the above code with the FedAvgJob, which is a pattern to simplify FedAvg job creations
    # job = FedAvgJob(name="cifar10_fedavg", num_rounds=num_rounds, n_clients=n_clients, initial_model=Net())

    # Add clients
    for i in range(n_clients):
        executor = ScriptRunner(
            script=train_script, script_args=""  # f"--batch_size 32 --data_path /tmp/data/site-{i}"
        )
        job.to(executor, target=f"site-{i}")
    # job.to_clients(executor)

    # job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
