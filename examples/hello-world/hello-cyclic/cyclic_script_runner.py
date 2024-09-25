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

from src.tf_net import Net

from nvflare import FedJob
from nvflare.app_common.workflows.cyclic import Cyclic
from nvflare.app_opt.tf.job_config.model import TFModel
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 3
    train_script = "src/hello-cyclic_fl.py"

    job = FedJob(name="hello-tf_cyclic")

    # Define the controller workflow and send to server
    controller = Cyclic(
        num_clients=n_clients,
        num_rounds=num_rounds,
    )
    job.to(controller, "server")

    # Define the initial global model and send to server
    job.to(TFModel(Net()), "server")

    # Add clients
    for i in range(n_clients):
        executor = ScriptRunner(
            script=train_script,
            script_args="",  # f"--batch_size 32 --data_path /tmp/data/site-{i}"
            framework=FrameworkType.TENSORFLOW,
        )
        job.to(executor, f"site-{i + 1}")

    # job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
