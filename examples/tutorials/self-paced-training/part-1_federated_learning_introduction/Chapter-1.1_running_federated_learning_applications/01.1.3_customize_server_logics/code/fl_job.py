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


from nvflare.job_config.api import FedJob
from src.network import SimpleNetwork
from src.fedavg_v2 import FedAvgV2

from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner

if __name__ == "__main__":
    num_clients = 5
    num_rounds = 2

    train_script = "src/client.py"

    job = FedJob(name="fedavg_v2", min_clients=num_clients)

    controller = FedAvgV2(
            stop_cond = None,
            save_filename = "global_model.pt",
            initial_model=SimpleNetwork(), 
            num_clients=num_clients,
            num_rounds=num_rounds)
        
    job.to_server(controller)

    # Add clients
    for i in range(num_clients):
        executor = ScriptRunner(
            script=train_script, script_args=""
        )
        job.to(executor, f"site-{i + 1}")

    job.simulator_run("/tmp/nvflare/jobs/workdir")
