# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod

from nvflare.apis.client import Client
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class ResponseProcessor(FLComponent, ABC):
    @abstractmethod
    def create_task_data(self, task_name: str, fl_ctx: FLContext) -> Shareable:
        """Create the task data for the process request to clients
        This method is called at the beginning of the ResponseProcessor controller, e.g., in BroadcastAndProcess.
        The internal state of the processor should be reset here, if the processor is used multiple times.

        Args:
            task_name: name of the task
            fl_ctx: FL context

        Returns: task data as a shareable
        """
        pass

    @abstractmethod
    def process_client_response(self, client: Client, task_name: str, response: Shareable, fl_ctx: FLContext) -> bool:
        """Processes the response submitted by a client.
        This method is called every time a response is received from a client.

        Args:
            client: the client that submitted response
            task_name: name of the task that the response corresponds to
            response: client submitted response
            fl_ctx: FLContext

        Returns:
            boolean to indicate if the client data is acceptable.
            If not acceptable, the control flow will exit.

        """
        pass

    @abstractmethod
    def final_process(self, fl_ctx: FLContext) -> bool:
        """Perform the final process.
        This method is called after received responses from all clients.

        Args:
            fl_ctx: FLContext

        Returns:
            boolean indicating whether the final response processing is successful.
            If not successful, the control flow will exit.
        """
        pass
