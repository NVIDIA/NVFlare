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
from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.ccwf import (
    CrossSiteEvalClientController,
    CrossSiteEvalServerController,
    SwarmClientController,
    SwarmServerController,
)
from nvflare.app_common.ccwf.comps.simple_model_shareable_generator import SimpleModelShareableGenerator
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 3
    train_script = "src/train_eval_submit.py"

    job = FedJob(name="cifar10_swarm")

    controller = SwarmServerController(
        num_rounds=num_rounds,
    )
    job.to(controller, "server")
    controller = CrossSiteEvalServerController(eval_task_timeout=300)
    job.to(controller, "server")

    # Define the initial server model
    job.to(Net(), "server")

    for i in range(n_clients):
        executor = ScriptExecutor(task_script_path=train_script)
        job.to(executor, f"site-{i}", gpu=0, tasks=["train", "validate", "submit_model"])

        client_controller = SwarmClientController()
        job.to(client_controller, f"site-{i}", tasks=["swarm_*"])

        client_controller = CrossSiteEvalClientController()
        job.to(client_controller, f"site-{i}", tasks=["cse_*"])

        # In swarm learning, each client acts also as an aggregator
        aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS)
        job.to(aggregator, f"site-{i}")

        # In swarm learning, each client uses a model persistor and shareable_generator
        job.to(PTFileModelPersistor(model=Net()), f"site-{i}")
        job.to(SimpleModelShareableGenerator(), f"site-{i}")

    # job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir")
