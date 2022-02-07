# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.client_engine_spec import ClientEngineSpec
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class ClientEngineExecutorSpec(ClientEngineSpec):
    """The ClientEngineExecutorSpec defines the ClientEngine APIs running in the child process."""

    def aux_send(self, topic: str, request: Shareable, timeout: float, fl_ctx: FLContext) -> Shareable:
        """Send the request to the Server.

        If reply is received, make sure to set peer_ctx into the reply shareable!

        Args:
            topic: topic of the request
            request: request Shareable to be sent
            timeout: number of secs to wait for reply. 0 means fire-and-forget.
            fl_ctx: fl context

        Returns: a reply.

        """
        pass

    def build_component(self, config_dict):
        """Build a component from the config_dict.

        Args:
            config_dict: config dict

        """

    def abort_app(self, run_number: int, fl_ctx: FLContext):
        """Abort the running FL App on the client.

        Args:
            run_number: current_run_number
            fl_ctx: FLContext

        """
        pass
