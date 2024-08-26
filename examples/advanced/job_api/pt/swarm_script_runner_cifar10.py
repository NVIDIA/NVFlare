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

from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.ccwf.ccwf_job import CCWFJob, CrossSiteEvalConfig, SwarmClientConfig, SwarmServerConfig
from nvflare.app_common.ccwf.comps.simple_model_shareable_generator import SimpleModelShareableGenerator
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.job_config.script_runner import ScriptRunner

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 3
    train_script = "src/train_eval_submit.py"

    job = CCWFJob(name="cifar10_swarm")
    aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS)
    job.add_swarm(
        server_config=SwarmServerConfig(num_rounds=num_rounds),
        client_config=SwarmClientConfig(
            executor=ScriptRunner(script=train_script),
            aggregator=aggregator,
            persistor=PTFileModelPersistor(model=Net()),
            shareable_generator=SimpleModelShareableGenerator(),
        ),
        cse_config=CrossSiteEvalConfig(eval_task_timeout=300),
    )

    # job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir", n_clients=n_clients, gpu="0")
