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

"""Fixed executor for the signed CoCo hello-numpy validation trainer."""

from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor, ExecutionMode
from nvflare.client.config import ExchangeFormat, TransferType


class CoCoHelloNumpyExecutor(ClientAPIExecutor):
    """Run only the hello-numpy trainer baked into the signed CoCo image.

    The constructor intentionally accepts no job-controlled entry point or
    arguments. This makes the exact class safe to add to a participant's CoCo
    component allow-list without allowing a submitted configuration to select
    another executable path.
    """

    TRAINER_PATH = "/opt/nvflare/examples/hello-numpy/client.py"

    def __init__(self):
        super().__init__(
            execution_mode=ExecutionMode.IN_PROCESS,
            task_script_path=self.TRAINER_PATH,
            task_script_args="--update_type full",
            params_exchange_format=ExchangeFormat.NUMPY,
            server_expected_format=ExchangeFormat.NUMPY,
            params_transfer_type=TransferType.FULL,
        )
