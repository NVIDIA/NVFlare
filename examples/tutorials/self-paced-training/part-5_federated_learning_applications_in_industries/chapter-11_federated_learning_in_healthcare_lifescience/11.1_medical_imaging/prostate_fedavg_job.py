# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from src.learners.supervised_monai_prostate_learner import SupervisedMonaiProstateLearner
from src.unet import UNet

from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.executors.learner_executor import LearnerExecutor
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.job_config.api import FedJob

if __name__ == "__main__":
    n_clients = 4
    num_rounds = 3
    train_script = "src/monai_mednist_train.py"

    job = FedJob(name="prostate_fedavg")

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=[16, 32, 64, 128, 256],
        strides=[2, 2, 2, 2],
        num_res_units=2,
    )
    persistor = PTFileModelPersistor(model=model)
    job.to_server(persistor, id="persistor")

    shareable_generator = FullModelShareableGenerator()
    job.to_server(shareable_generator, id="shareable_generator")

    aggregator = InTimeAccumulateWeightedAggregator()
    job.to_server(aggregator, id="aggregator")

    model_selector = IntimeModelSelector(weigh_by_local_iter=True)
    job.to_server(model_selector, id="model_selector")

    # Add controller
    controller = ScatterAndGather(
        min_clients=n_clients,
        num_rounds=num_rounds,
        start_round=0,
        wait_time_after_min_received=10,
        aggregator_id="aggregator",
        persistor_id="persistor",
        shareable_generator_id="shareable_generator",
        train_task_name="train",
        train_timeout=0,
    )
    job.to_server(controller)

    # Add clients
    learner = SupervisedMonaiProstateLearner(
        train_config_filename="../custom/src/config/config_train.json", aggregation_epochs=10
    )
    job.to_clients(learner, id="prostate-learner")
    executor = LearnerExecutor(learner_id="prostate-learner")
    job.to_clients(executor)
    job.to_clients("src/config/config_train.json")

    job.export_job("/tmp/nvflare/jobs/")
    job.simulator_run("/tmp/nvflare/workspaces/prostate_fedavg", n_clients=n_clients, gpu="0")
