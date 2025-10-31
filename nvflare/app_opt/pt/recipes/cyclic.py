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

from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.script_runner import FrameworkType
from nvflare.recipe.cyclic import CyclicRecipe as BaseCyclicRecipe


class CyclicRecipe(BaseCyclicRecipe):
    def __init__(
        self,
        *,
        name: str = "cyclic",
        initial_model: Any = None,
        num_rounds: int = 2,
        min_clients: int = 2,
        train_script: str,
        train_args: str = "",
        launch_external_process: bool = False,
        command: str = "python3 -u",
        framework: FrameworkType = FrameworkType.PYTORCH,
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
    ):
        if initial_model is None or isinstance(initial_model, PTModel):
            model_to_pass = initial_model
        else:
            model_to_pass = PTModel(initial_model)
        super().__init__(
            name=name,
            initial_model=model_to_pass,
            num_rounds=num_rounds,
            min_clients=min_clients,
            train_script=train_script,
            train_args=train_args,
            launch_external_process=launch_external_process,
            command=command,
            framework=framework,
            server_expected_format=server_expected_format,
            params_transfer_type=params_transfer_type,
        )
