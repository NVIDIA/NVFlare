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

from nvflare import FedJob
from nvflare.app_common.np.np_model_persistor import NPModelPersistor
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 3
    train_script = "src/hello-numpy_fl.py"

    job = FedJob(name="hello-fedavg-numpy")

    persistor_id = job.to_server(NPModelPersistor(), "persistor")

    # Define the controller workflow and send to server
    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
        persistor_id=persistor_id,
    )
    job.to(controller, "server")

    job.to(IntimeModelSelector(key_metric="accuracy"), "server")

    # Add clients
    for i in range(n_clients):
        executor = ScriptRunner(script=train_script, script_args="", framework=FrameworkType.NUMPY)
        job.to(executor, f"site-{i + 1}")

    # job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
