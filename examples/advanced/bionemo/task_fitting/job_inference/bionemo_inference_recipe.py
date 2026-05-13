# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.app_common.workflows.cmd_task_controller import CmdTaskController
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe.spec import Recipe


class BioNeMoInferenceRecipe(Recipe):
    """A local BioNeMo recipe for one-round embedding inference."""

    TASK_NAME = "infer"

    def __init__(
        self,
        name: str,
        min_clients: int,
        task_script: str,
        task_args: str = "",
    ):
        job = FedJob(name=name, min_clients=min_clients)

        controller = CmdTaskController(task_name=self.TASK_NAME, persistor_id="")
        job.to_server(controller)

        executor = ScriptRunner(
            script=task_script,
            script_args=task_args,
            framework=FrameworkType.RAW,
            server_expected_format=ExchangeFormat.RAW,
            params_transfer_type=TransferType.FULL,
        )
        job.to_clients(executor, tasks=[self.TASK_NAME])

        super().__init__(job)
