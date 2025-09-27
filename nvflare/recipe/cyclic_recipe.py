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

from typing import Any

from pydantic import BaseModel

from nvflare import FedJob
from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
from nvflare.app_common.workflows.cyclic_ctl import CyclicController
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner
from nvflare.recipe.spec import Recipe


# Internal — not part of the public API
class _CyclicValidator(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str
    initial_model: Any
    num_rounds: int
    train_script: str
    train_args: str
    launch_external_process: bool = False
    command: str = "python3 -u"
    server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY
    params_transfer_type: TransferType = TransferType.FULL
    framework: FrameworkType = FrameworkType.NUMPY


class CyclicRecipe(Recipe):
    def __init__(
        self,
        *,
        name: str = "cyclic",
        initial_model: Any = None,
        num_rounds: int = 2,
        train_script: str,
        train_args: str = "",
        launch_external_process: bool = False,
        command: str = "python3 -u",
        framework: FrameworkType = FrameworkType.NUMPY,
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
    ):
        # Validate inputs internally
        v = _CyclicValidator(
            name=name,
            initial_model=initial_model,
            num_rounds=num_rounds,
            train_script=train_script,
            train_args=train_args,
            launch_external_process=launch_external_process,
            command=command,
            framework=framework,
            server_expected_format=server_expected_format,
            params_transfer_type=params_transfer_type,
        )

        self.name = v.name
        self.initial_model = v.initial_model
        self.num_rounds = v.num_rounds
        self.initial_model = v.initial_model
        self.train_script = v.train_script
        self.train_args = v.train_args
        self.launch_external_process = v.launch_external_process
        self.command = v.command
        self.framework = v.framework
        self.server_expected_format: ExchangeFormat = v.server_expected_format
        self.params_transfer_type: TransferType = v.params_transfer_type

        job = FedJob(name=name)
        # Define the controller workflow and send to server
        controller = CyclicController(
            num_rounds=num_rounds,
            task_assignment_timeout=10,
            persistor_id="persistor",
            shareable_generator_id="shareable_generator",
            task_name="train",
            task_check_period=0.5,
        )
        job.to(controller, "server")

        shareable_generator = FullModelShareableGenerator()
        job.to_server(shareable_generator, id="shareable_generator")

        # Define the initial global model and send to server
        job.to(self.initial_model, "server")

        executor = ScriptRunner(
            script=self.train_script,
            script_args=self.train_args,
            launch_external_process=self.launch_external_process,
            framework=self.framework,
            server_expected_format=self.server_expected_format,
            params_transfer_type=self.params_transfer_type,
        )
        job.to_clients(executor)

        super().__init__(job)
