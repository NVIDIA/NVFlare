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


import os
import shutil

from src.fedavg import FedAvg
from src.network import SimpleNetwork

from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import ScriptRunner

if __name__ == "__main__":
    num_clients = 2
    num_rounds = 2
    train_script = "src/client.py"
    job_config_dir = "/tmp/nvflare/jobs/workdir"

    for i in range(5):
        job_name = f"job_{i + 1}"
        job = FedJob(name=job_name, min_clients=num_clients)

        controller = FedAvg(
            stop_cond="accuracy > 25",
            save_filename="global_model.pt",
            initial_model=SimpleNetwork(),
            num_clients=num_clients,
            num_rounds=num_rounds,
        )

        job.to_server(controller)

        # Add clients
        for site_name in ["site_a", "site_b"]:
            executor = ScriptRunner(script=train_script)
            job.to(executor, site_name)

        print("job-config is at ", os.path.join(job_config_dir, job_name))
        job.export_job(job_config_dir)
        source_meta_file = os.path.join(f"job{i + 1}", "meta.json")
        dest_meta_file = os.path.join(job_config_dir, job_name, "meta.json")
        shutil.copy2(source_meta_file, dest_meta_file)
