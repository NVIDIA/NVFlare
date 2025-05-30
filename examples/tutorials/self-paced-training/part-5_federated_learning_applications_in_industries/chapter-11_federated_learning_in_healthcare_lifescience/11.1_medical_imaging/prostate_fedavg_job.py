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

from nvflare.app_common.executors.learner_executor import LearnerExecutor
from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob

if __name__ == "__main__":
    n_clients = 4
    num_rounds = 3
    train_script = "src/monai_mednist_train.py"

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=[16, 32, 64, 128, 256],
        strides=[2, 2, 2, 2],
        num_res_units=2,
    )

    job = FedAvgJob(name="prostate_fedavg", n_clients=n_clients, num_rounds=num_rounds, initial_model=model)

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
