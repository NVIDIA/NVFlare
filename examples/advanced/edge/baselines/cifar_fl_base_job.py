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
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.edge.models.model import Cifar10ConvNet
from nvflare.job_config.script_runner import ScriptRunner

if __name__ == "__main__":
    n_clients = 16
    num_rounds = 10
    train_script = "cifar_fl_base.py"

    job = FedJob(name="cifar10_fl_base")

    shareable_generator_id = job.to_server(FullModelShareableGenerator(), id="shareable_generator")
    persistor_id = job.to_server(PTFileModelPersistor(model=Cifar10ConvNet()), id="persistor")
    aggregator_id = job.to_server(InTimeAccumulateWeightedAggregator(expected_data_kind="WEIGHTS"), id="aggregator")
    controller = ScatterAndGather(
        min_clients=n_clients,
        num_rounds=num_rounds,
        aggregator_id=aggregator_id,
        persistor_id=persistor_id,
        shareable_generator_id=shareable_generator_id,
        train_task_name="train",
    )
    job.to_server(controller)
    job.to_server(IntimeModelSelector(key_metric="accuracy"))

    # Add clients
    for i in range(n_clients):
        executor = ScriptRunner(
            script=train_script,
            script_args="",
        )
        job.to(executor, f"site-{i + 1}")

    job.export_job("/tmp/nvflare/jobs/cifar10")
    job.simulator_run("/tmp/nvflare/workspaces/cifar10_fl/", gpu="0")
