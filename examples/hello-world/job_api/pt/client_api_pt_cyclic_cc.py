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

from nvflare import FedJob, ScriptExecutor
from nvflare.app_common.ccwf import CyclicClientController, CyclicServerController
from nvflare.app_common.ccwf.comps.simple_model_shareable_generator import SimpleModelShareableGenerator
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 3
    train_script = "src/cifar10_fl.py"

    job = FedJob(name="cifar10_cyclic")

    controller = CyclicServerController(num_rounds=num_rounds, max_status_report_interval=300)
    job.to(controller, "server")

    for i in range(n_clients):
        executor = ScriptExecutor(
            task_script_path=train_script, task_script_args=""  # f"--batch_size 32 --data_path /tmp/data/site-{i}"
        )
        job.to(executor, f"site-{i}", gpu=0)

        # Add client-side controller for cyclic workflow
        executor = CyclicClientController()
        job.to(executor, f"site-{i}", tasks=["cyclic_*"])

        # In swarm learning, each client uses a model persistor and shareable_generator
        job.to(PTFileModelPersistor(model=Net()), f"site-{i}", id="persistor")
        job.to(SimpleModelShareableGenerator(), f"site-{i}", id="shareable_generator")

    # job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir")
