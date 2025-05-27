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

from src.densenet import DenseNet121

from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 5
    train_script = "src/monai_mednist_train.py"

    job = FedAvgJob(
        name="mednist_fedavg",
        n_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=DenseNet121(spatial_dims=2, in_channels=1, out_channels=6),
    )

    # Add clients
    executor = ScriptRunner(script=train_script, script_args="")
    job.to_clients(executor)

    job.export_job("/tmp/nvflare/jobs/")
    job.simulator_run("/tmp/nvflare/workspaces/mednist_fedavg", n_clients=n_clients, gpu="0")
