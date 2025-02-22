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

from src.newton_raphson_persistor import NewtonRaphsonModelPersistor
from src.newton_raphson_workflow import FedAvgNewtonRaphson

from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.client.config import ExchangeFormat
from nvflare.job_config.script_runner import ScriptRunner

if __name__ == "__main__":
    n_clients = 4
    num_rounds = 5

    job = BaseFedJob(
        name="logistic_regression_fedavg",
        model_persistor=NewtonRaphsonModelPersistor(n_features=13),
    )

    controller = FedAvgNewtonRaphson(
        num_clients=n_clients,
        num_rounds=num_rounds,
        damping_factor=0.8,
        persistor_id="newton_raphson_persistor",
    )
    job.to(controller, "server")

    # Add clients
    for i in range(n_clients):
        runner = ScriptRunner(
            script="src/newton_raphson_train.py",
            script_args="--data_root /tmp/flare/dataset/heart_disease_data",
            launch_external_process=True,
            params_exchange_format=ExchangeFormat.RAW,
        )
        job.to(runner, f"site-{i + 1}")

    job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0", log_config="./log_config.json")
