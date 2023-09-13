# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import os

from omegaconf import OmegaConf

from nvflare.apis.client import Client
from nvflare.apis.dxo import DXO
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.response_processor import ResponseProcessor

from .constants import NemoDataKind


class ConfigSharerSFT(ResponseProcessor):
    def __init__(
        self,
        config_path: str = "config/megatron_gpt_sft.yaml",
    ):
        """Share the NeMo config files with the clients.

        Args:
            config_path: NeMo model config file
        """
        super().__init__()
        self.config_path = config_path

    def create_task_data(self, task_name: str, fl_ctx: FLContext) -> Shareable:
        """Create the data for the task to be sent to clients

        Args:
            task_name: name of the task
            fl_ctx: the FL context

        Returns: task data

        """
        # get app root
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)

        # Load model configuration to initialize training NeMo environment
        self.config_path = os.path.join(app_root, self.config_path)
        config = OmegaConf.load(self.config_path)
        self.log_info(fl_ctx, f"Load model configuration from {self.config_path}")

        configs = {
            NemoDataKind.NEMO_CONFIG: OmegaConf.to_container(config),
        }

        # convert omega conf to primitive dict
        dxo = DXO(data=configs, data_kind=NemoDataKind.CONFIGS)
        return dxo.to_shareable()

    def process_client_response(self, client: Client, task_name: str, response: Shareable, fl_ctx: FLContext) -> bool:
        """Process the weights submitted by a client.

        Args:
            client: the client that submitted the response
            task_name: name of the task
            response: submitted data from the client
            fl_ctx: FLContext

        Returns:
            boolean to indicate if the client data is acceptable.
            If not acceptable, the control flow will exit.

        """

        # We only check for client errors here
        if not isinstance(response, Shareable):
            self.log_error(
                fl_ctx,
                f"bad response from client {client.name}: " f"response must be Shareable but got {type(response)}",
            )
            return False

        if response.get_return_code() != ReturnCode.OK:
            self.log_exception(
                fl_ctx, f"bad response from client {client.name}: Got return code {response.get_return_code()}"
            )
            return False

        return True

    def final_process(self, fl_ctx: FLContext) -> bool:
        """Perform the final check. Do nothing.

        Args:
            fl_ctx: FLContext

        Returns:
            boolean indicating whether the final response processing is successful.
            If not successful, the control flow will exit.
        """

        # no final processing required for this task
        return True
