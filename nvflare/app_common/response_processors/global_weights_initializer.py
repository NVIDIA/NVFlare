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

from nvflare.apis.client import Client
from nvflare.apis.dxo import DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.abstract.response_processor import ResponseProcessor
from nvflare.app_common.app_constant import AppConstants


class WeightMethod(object):

    FIRST = "first"
    CLIENT = "client"


class GlobalWeightsInitializer(ResponseProcessor):
    def __init__(
        self,
        weights_prop_name: str = AppConstants.GLOBAL_MODEL,
        weight_method: str = WeightMethod.FIRST,
        client_name: str = None,
    ):
        """Set global model weights based on specified weight setting method.

        Args:
            weights_prop_name: name of the prop to be set into fl_ctx for the determined global weights
            weight_method: the method to select final weights: one of "first", "client"
            client_name: the name of the client to be used as the weight provider

        If weight_method is "first", then use the weights reported from the first client;
        If weight_method is "client", then only use the weights reported from the specified client.
        """
        if weight_method not in [WeightMethod.FIRST, WeightMethod.CLIENT]:
            raise ValueError(f"invalid weight_method '{weight_method}'")
        if weight_method == WeightMethod.CLIENT and not client_name:
            raise ValueError(f"client name not provided for weight method '{WeightMethod.CLIENT}'")
        if weight_method == WeightMethod.CLIENT and not isinstance(client_name, str):
            raise ValueError(
                f"client name should be a single string for weight method '{WeightMethod.CLIENT}' but it is {client_name} "
            )

        ResponseProcessor.__init__(self)
        self.weights_prop_name = weights_prop_name
        self.weight_method = weight_method
        self.client_name = client_name
        self.final_weights = None

    def create_task_data(self, task_name: str, fl_ctx: FLContext) -> Shareable:
        """Create the data for the task to be sent to clients to collect their weights

        Args:
            task_name: name of the task
            fl_ctx: the FL context

        Returns: task data

        """
        # reset internal state in case this processor is used multiple times
        self.final_weights = None
        return Shareable()

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
        if not isinstance(response, Shareable):
            self.log_error(
                fl_ctx,
                f"bad response from client {client.name}: " f"response must be Shareable but got {type(response)}",
            )
            return False

        try:
            dxo = from_shareable(response)
        except Exception:
            self.log_exception(fl_ctx, f"bad response from client {client.name}: " f"it does not contain DXO")
            return False

        if dxo.data_kind != DataKind.WEIGHTS:
            self.log_error(
                fl_ctx,
                f"bad response from client {client.name}: "
                f"data_kind should be DataKind.WEIGHTS but got {dxo.data_kind}",
            )
            return False

        weights = dxo.data
        if not weights:
            self.log_error(fl_ctx, f"No model weights found from client {client.name}")
            return False

        if not self.final_weights and (
            self.weight_method == WeightMethod.FIRST
            or (self.weight_method == WeightMethod.CLIENT and client.name == self.client_name)
        ):
            self.final_weights = weights

        return True

    def final_process(self, fl_ctx: FLContext) -> bool:
        """Perform the final check on all the received weights from the clients.

        Args:
            fl_ctx: FLContext

        Returns:
            boolean indicating whether the final response processing is successful.
            If not successful, the control flow will exit.
        """
        if not self.final_weights:
            self.log_error(fl_ctx, "no weights available from clients")
            return False

        # must set sticky to True so other controllers can get it!
        fl_ctx.set_prop(self.weights_prop_name, make_model_learnable(self.final_weights, {}), private=True, sticky=True)
        return True
