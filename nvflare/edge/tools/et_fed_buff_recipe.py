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

import json
from typing import Dict

from nvflare.edge.models.model import DeviceModel
from nvflare.edge.tools.edge_fed_buff_recipe import (
    DeviceManagerConfig,
    EdgeFedBuffRecipe,
    EvaluatorConfig,
    ModelManagerConfig,
    SimulationConfig,
)
from nvflare.edge.tools.et_job import ETJob
from nvflare.job_config.file_source import FileSource

_TRAINER_NAME = "trainer"
_DEVICE_CONFIG_FILE_NAME = "device_config.json"


class ETFedBuffRecipe(EdgeFedBuffRecipe):

    def __init__(
        self,
        job_name: str,
        device_model: DeviceModel,
        input_shape,
        output_shape,
        model_manager_config: ModelManagerConfig,
        device_manager_config: DeviceManagerConfig,
        evaluator_config: EvaluatorConfig = None,
        simulation_config: SimulationConfig = None,
        device_training_params: Dict = None,
        custom_source_root: str = None,
    ):
        self.device_model = device_model
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.device_training_params = device_training_params

        pt_model = device_model.net
        EdgeFedBuffRecipe.__init__(
            self,
            job_name=job_name,
            model=pt_model,
            model_manager_config=model_manager_config,
            device_manager_config=device_manager_config,
            evaluator_config=evaluator_config,
            simulation_config=simulation_config,
            custom_source_root=custom_source_root,
        )

    def create_job(self):
        return ETJob(
            name=self.job_name,
            edge_method=self.method_name,
            device_model=self.device_model,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
        )

    def _configure_job(self, job):
        super()._configure_job(job)

        # add device training config file if specified
        if self.device_training_params:
            trainer_config = {"type": "Trainer.DLTrainer", "name": _TRAINER_NAME, "args": self.device_training_params}
            device_config = {"components": [trainer_config], "executors": {"train": f"@{_TRAINER_NAME}"}}

            with open(_DEVICE_CONFIG_FILE_NAME, "w") as f:
                json.dump(device_config, f, indent=2)

            job.to_server(FileSource(_DEVICE_CONFIG_FILE_NAME, app_folder_type="config"))
