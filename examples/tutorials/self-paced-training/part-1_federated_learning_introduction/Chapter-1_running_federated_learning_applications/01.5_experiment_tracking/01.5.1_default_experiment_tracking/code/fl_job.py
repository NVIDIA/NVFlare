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

from src.network import SimpleNetwork

from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner

if __name__ == "__main__":
    n_clients = 5
    num_rounds = 20

    train_script = "src/client.py"

    job = FedAvgJob(name="fedavg", n_clients=n_clients, num_rounds=num_rounds, initial_model=SimpleNetwork())

    # Add clients
    for i in range(n_clients):
        executor = ScriptRunner(
            script=train_script, script_args=""  # f"--batch_size 32 --data_path /tmp/data/site-{i}"
        )
        job.to(executor, f"site-{i + 1}")

    job.simulator_run(workspace="/tmp/nvflare/jobs/workdir", log_config="./log_config.json")
