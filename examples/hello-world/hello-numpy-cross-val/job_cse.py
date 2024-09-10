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
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.np.np_formatter import NPFormatter
from nvflare.app_common.np.np_model_locator import NPModelLocator
from nvflare.app_common.np.np_trainer import NPTrainer
from nvflare.app_common.np.np_validator import NPValidator
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval

SERVER_MODEL_DIR = "/tmp/nvflare/server_pretrain_models"
CLIENT_MODEL_DIR = "/tmp/nvflare/client_pretrain_models"


if __name__ == "__main__":
    n_clients = 2

    job = FedJob(name="hello-numpy-cse", min_clients=n_clients)

    model_locator_id = job.to_server(
        NPModelLocator(
            model_dir="/tmp/nvflare/server_pretrain_models",
            model_name={"server_model_1": "server_1.npy", "server_model_2": "server_2.npy"},
        )
    )
    formatter_id = job.to_server(NPFormatter())
    job.to_server(ValidationJsonGenerator())

    # Define the controller workflow and send to server
    controller = CrossSiteModelEval(
        model_locator_id=model_locator_id,
        formatter_id=formatter_id,
    )
    job.to_server(controller)

    # Add clients
    trainer = NPTrainer(
        train_task_name=AppConstants.TASK_TRAIN,
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
        model_dir="/tmp/nvflare/client_pretrain_models",
    )
    job.to_clients(trainer, tasks=[AppConstants.TASK_SUBMIT_MODEL])
    validator = NPValidator(
        validate_task_name=AppConstants.TASK_VALIDATION,
    )
    job.to_clients(validator, tasks=[AppConstants.TASK_VALIDATION])

    job.export_job("/tmp/nvflare/jobs")
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0", n_clients=n_clients)
